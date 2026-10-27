# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Provider-neutral text generation for PReMISE.

The research implementation uses Amazon Bedrock by default. This module keeps
that path intact while providing small, dependency-free adapters for OpenAI,
Anthropic, Gemini, and OpenRouter.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from premise.config import (
    SUPPORTED_PROVIDER_NAMES,
    ConfigError,
    get_config,
    normalize_provider_name,
)

# Ordered for display in `--provider` choices; bedrock leads because it is the
# default. The set must match premise.config, so the check runs under -O too.
SUPPORTED_PROVIDERS = ("bedrock", "openai", "anthropic", "gemini", "openrouter")
if set(SUPPORTED_PROVIDERS) != SUPPORTED_PROVIDER_NAMES:
    raise ImportError(
        "SUPPORTED_PROVIDERS disagrees with premise.config.SUPPORTED_PROVIDER_NAMES: "
        f"{sorted(set(SUPPORTED_PROVIDERS) ^ SUPPORTED_PROVIDER_NAMES)}"
    )


class ProviderConfigurationError(ValueError):
    """Raised when a provider is unknown or its credentials are missing."""


class ProviderRequestError(RuntimeError):
    """Raised when a provider request fails after retries."""


class _EmptyProviderResponse(RuntimeError):
    """Internal error carrying whether an empty response is worth retrying."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider settings.

    API keys are intentionally excluded from ``repr`` so diagnostics cannot
    accidentally print credentials.
    """

    provider: str
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    site_url: str | None = None
    app_name: str | None = None
    reasoning_effort: str | None = None

    def __repr__(self) -> str:
        return (
            "ProviderConfig("
            f"provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, timeout={self.timeout!r}, "
            f"site_url={self.site_url!r}, app_name={self.app_name!r}, "
            f"reasoning_effort={self.reasoning_effort!r})"
        )


def provider_name(value: str | None = None) -> str:
    """Return a normalized provider name from arguments or active config."""
    configured = get_config().provider
    try:
        return normalize_provider_name(value or configured)
    except ConfigError as exc:
        raise ProviderConfigurationError(str(exc)) from None


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _key_env_names(config: Any) -> tuple[str, ...]:
    configured = config.get("api_key_env")
    if isinstance(configured, str):
        return (configured,)
    if isinstance(configured, (list, tuple)):
        return tuple(str(name) for name in configured if name)
    return ()


def load_provider_config(
    provider: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    require_key: bool = True,
) -> ProviderConfig:
    """Resolve provider configuration from explicit values and active config."""
    name = provider_name(provider)
    app_config = get_config()
    provider_config = app_config.provider_config(name)
    resolved_model = model or app_config.model(provider=name)
    resolved_base = base_url or provider_config.get("base_url")
    configured_timeout = provider_config.get("timeout")
    timeout = float(configured_timeout) if configured_timeout is not None else None
    key_env_names = _key_env_names(provider_config)
    api_key = _first_env(key_env_names)

    if name != "bedrock" and require_key and not api_key:
        expected = " or ".join(key_env_names)
        raise ProviderConfigurationError(
            f"{name} requires {expected}. Set it in your environment; "
            "PReMISE does not accept API keys as CLI flags."
        )

    headers = provider_config.get("headers", {})
    return ProviderConfig(
        provider=name,
        model=resolved_model,
        api_key=api_key,
        base_url=resolved_base,
        timeout=timeout,
        site_url=headers.get("site_url"),
        app_name=headers.get("app_name"),
        reasoning_effort=provider_config.get("reasoning_effort"),
    )


def configured_model(requested_model: str) -> str:
    """Map Bedrock-specific call sites to the active provider's default model."""
    if provider_name() != "bedrock":
        return get_config().model()
    return requested_model


def model_for_role(role: str, default: str | None = None) -> str:
    """Resolve a role-specific model from the active config.

    ``default`` is retained temporarily for source compatibility and is ignored.
    """
    return get_config().model(role, provider_name())


def parse_model_specs(values: list[str] | None) -> list[tuple[str, str]] | None:
    """Parse repeatable ``NAME=MODEL`` values used by ensemble audits."""
    if not values:
        return None
    parsed = []
    for value in values:
        if "=" in value:
            name, model = value.split("=", 1)
        else:
            name = model = value
        name, model = name.strip(), model.strip()
        if not name or not model:
            raise ProviderConfigurationError(
                f"Invalid model specification {value!r}; use NAME=MODEL."
            )
        parsed.append((name, model))
    return parsed


def local_hash_embedding(text: str, dimensions: int = 1024) -> list[float]:
    """Return a deterministic, dependency-free lexical embedding.

    This is a portability fallback for discovery runs outside Bedrock. It is
    not the embedding model used in the paper.
    """
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = sum(value * value for value in vector) ** 0.5
    if norm:
        vector = [value / norm for value in vector]
    return vector


class ProviderClient:
    """Minimal synchronous client for non-Bedrock providers."""

    def __init__(self, config: ProviderConfig):
        if config.provider == "bedrock":
            raise ProviderConfigurationError(
                "ProviderClient is for HTTP providers; use the Bedrock client."
            )
        self.config = config
        self.provider = config.provider

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system: str | None = None,
        max_retries: int = 3,
    ) -> str:
        """Generate text using the configured provider."""
        selected_model = model or self.config.model
        if not selected_model:
            raise ProviderConfigurationError(
                "No model configured. Set providers.<name>.models.default in "
                "config.json or pass --model."
            )

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                text = self._generate_once(
                    selected_model,
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=system,
                )
                if isinstance(text, str) and text.strip():
                    return text
                last_error = ValueError("provider returned an empty text response")
                if attempt == max_retries - 1:
                    break
                time.sleep(min(2**attempt, 8))
            except _EmptyProviderResponse as exc:
                last_error = exc
                if not exc.retryable or attempt == max_retries - 1:
                    break
                time.sleep(min(2**attempt, 8))
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {
                    408,
                    409,
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                if not retryable or attempt == max_retries - 1:
                    break
                time.sleep(min(2**attempt, 8))

        detail = _http_error_detail(last_error)
        raise ProviderRequestError(
            f"{self.provider} request failed for model {selected_model!r}: {detail}"
        ) from last_error

    def _generate_once(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        system: str | None,
    ) -> str:
        if self.provider == "openai":
            return self._openai(model, prompt, temperature, max_tokens, system)
        if self.provider == "anthropic":
            return self._anthropic(model, prompt, temperature, max_tokens, system)
        if self.provider == "gemini":
            return self._gemini(model, prompt, temperature, max_tokens, system)
        if self.provider == "openrouter":
            return self._openrouter(model, prompt, temperature, max_tokens, system)
        raise ProviderConfigurationError(f"Unsupported provider: {self.provider}")

    def _openai(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str | None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "input": messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        data = self._request(
            f"{self.config.base_url}/responses",
            payload,
            {"Authorization": f"Bearer {self.config.api_key}"},
        )
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        texts = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "".join(texts)

    def _anthropic(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        data = self._request(
            f"{self.config.base_url}/messages",
            payload,
            {
                "x-api-key": str(self.config.api_key),
                "anthropic-version": "2023-06-01",
            },
        )
        return "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )

    def _gemini(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str | None,
    ) -> str:
        quoted_model = urllib.parse.quote(model, safe="")
        url = f"{self.config.base_url}/models/{quoted_model}:generateContent"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        data = self._request(
            url,
            payload,
            {"x-goog-api-key": str(self.config.api_key)},
        )
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        return "".join(
            part.get("text", "")
            for part in candidates[0].get("content", {}).get("parts", [])
        )

    def _openrouter(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str | None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.app_name:
            headers["X-Title"] = self.config.app_name
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        data = self._request(
            f"{self.config.base_url}/chat/completions",
            payload,
            headers,
        )
        choices = data.get("choices", [])
        if not choices:
            raise _EmptyProviderResponse(
                "OpenRouter returned no choices",
                retryable=True,
            )
        choice = choices[0]
        content = choice.get("message", {}).get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        else:
            text = ""
        if text.strip():
            return text

        usage = data.get("usage", {})
        completion_details = usage.get("completion_tokens_details", {})
        finish_reason = choice.get("finish_reason")
        details = [
            f"finish_reason={finish_reason!r}",
            f"provider={data.get('provider')!r}",
            f"completion_tokens={usage.get('completion_tokens')!r}",
            f"reasoning_tokens={completion_details.get('reasoning_tokens')!r}",
        ]
        if self.config.reasoning_effort:
            details.append(f"reasoning_effort={self.config.reasoning_effort!r}")
        raise _EmptyProviderResponse(
            "OpenRouter returned empty visible content ("
            + ", ".join(details)
            + "). Reasoning tokens share max_tokens; lower reasoning_effort or "
            "increase the command's token budget.",
            retryable=finish_reason not in {"length", "max_tokens"},
        )

    def _request(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        if self.config.timeout is None:
            raise ProviderConfigurationError(
                f"Provider {self.provider!r} has no HTTP timeout configured."
            )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def get_provider_client(provider: str | None = None) -> ProviderClient:
    """Create a configured non-Bedrock provider client."""
    return ProviderClient(load_provider_config(provider))


def provider_status() -> list[dict[str, str | bool]]:
    """Return credential/configuration status without exposing key values."""
    rows = []
    active = provider_name()
    app_config = get_config()
    for name in app_config["providers"]:
        provider_config = app_config.provider_config(name)
        key_vars = _key_env_names(provider_config)
        rows.append(
            {
                "provider": name,
                "active": name == active,
                "credential": "AWS credential chain"
                if name == "bedrock"
                else " or ".join(key_vars),
                "configured": (name == "bedrock" or bool(_first_env(key_vars))),
            }
        )
    return rows


def _http_error_detail(error: Exception | None) -> str:
    if error is None:
        return "unknown error"
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return f"HTTP {error.code} {body[:500]}".strip()
    return str(error)
