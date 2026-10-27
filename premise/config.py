# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Validated, invocation-stable configuration for PReMISE."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

CONFIG_ENV = "PREMISE_CONFIG"
SCHEMA_VERSION = "1.0"
SUPPORTED_PROVIDER_NAMES = frozenset(
    {"bedrock", "openai", "anthropic", "gemini", "openrouter"}
)
_PROVIDER_ALIASES = {"aws": "bedrock", "google": "gemini"}
_ACTIVE_CONFIG: RuntimeConfig | None = None


class ConfigError(ValueError):
    """Raised when PReMISE configuration is missing or invalid."""


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class RuntimeConfig(Mapping[str, Any]):
    """An immutable, validated configuration snapshot."""

    _data: Mapping[str, Any]
    source: str

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def provider(self) -> str:
        """Return the selected provider name."""
        return str(self._data["provider"])

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable copy suitable for serialization or overrides."""
        return _thaw(self._data)

    def provider_config(self, provider: str | None = None) -> Mapping[str, Any]:
        """Return one provider block."""
        name = normalize_provider_name(provider or self.provider)
        providers = self._data["providers"]
        try:
            return providers[name]
        except KeyError:
            raise ConfigError(
                f"Provider {name!r} is not defined in configuration from {self.source}."
            ) from None

    def model(self, role: str = "default", provider: str | None = None) -> str:
        """Return a role-specific model, falling back to configured ``default``."""
        models = self.provider_config(provider)["models"]
        selected = models.get(role) or models.get("default")
        if not selected:
            name = normalize_provider_name(provider or self.provider)
            raise ConfigError(
                f"Provider {name!r} has no model for role {role!r} and no "
                f"'default' model in configuration from {self.source}."
            )
        return str(selected)

    def ensemble(
        self, name: str, provider: str | None = None
    ) -> tuple[Mapping[str, Any], ...]:
        """Return a named model ensemble."""
        provider_block = self.provider_config(provider)
        ensembles = provider_block.get("ensembles", {})
        try:
            return tuple(ensembles[name])
        except KeyError:
            selected = normalize_provider_name(provider or self.provider)
            raise ConfigError(
                f"Provider {selected!r} has no ensemble {name!r} in "
                f"configuration from {self.source}."
            ) from None

    def aws(self, key: str, *, required: bool = False) -> Any:
        """Return one AWS setting and optionally require a non-empty value."""
        aws_config = self._data["aws"]
        if key not in aws_config:
            raise ConfigError(f"Unknown AWS configuration key: {key!r}.")
        value = aws_config[key]
        if required and (value is None or value == ""):
            raise ConfigError(
                f"AWS setting 'aws.{key}' is required for this operation. "
                f"Set it in {self.source} before retrying."
            )
        return value

    def runtime(self, key: str) -> Any:
        """Return one runtime setting."""
        runtime_config = self._data["runtime"]
        if key not in runtime_config:
            raise ConfigError(f"Unknown runtime configuration key: {key!r}.")
        return runtime_config[key]


def normalize_provider_name(value: str) -> str:
    """Normalize and validate a provider name."""
    raw = value.strip().lower()
    name = _PROVIDER_ALIASES.get(raw, raw)
    if name not in SUPPORTED_PROVIDER_NAMES:
        choices = ", ".join(sorted(SUPPORTED_PROVIDER_NAMES))
        raise ConfigError(f"Unknown provider {raw!r}. Choose one of: {choices}.")
    return name


def _error(source: str, field: str, message: str) -> ConfigError:
    return ConfigError(f"Invalid PReMISE config {source}: {field} {message}.")


def _object(value: Any, source: str, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error(source, field, "must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    source: str,
    field: str,
    *,
    allowed: set[str],
    required: set[str] | None = None,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise _error(source, field, f"contains unknown field(s): {names}")
    missing = sorted((required or set()) - set(value))
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise _error(source, field, f"is missing required field(s): {names}")


def _string(value: Any, source: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(source, field, "must be a non-empty string")
    return value


def _nullable_string(value: Any, source: str, field: str) -> None:
    if value is not None:
        _string(value, source, field)


def _number(
    value: Any,
    source: str,
    field: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> None:
    valid_type = isinstance(value, int) and not isinstance(value, bool)
    if not integer:
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid_type:
        expected = "an integer" if integer else "a number"
        raise _error(source, field, f"must be {expected}")
    if minimum is not None and value < minimum:
        raise _error(source, field, f"must be at least {minimum}")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise _error(source, field, f"must be greater than {exclusive_minimum}")
    if maximum is not None and value > maximum:
        raise _error(source, field, f"must be at most {maximum}")


def _validate_provider(name: str, value: Any, source: str) -> None:
    field = f"providers.{name}"
    provider = _object(value, source, field)
    _exact_keys(
        provider,
        source,
        field,
        allowed={
            "base_url",
            "api_key_env",
            "timeout",
            "models",
            "ensembles",
            "headers",
            "reasoning_effort",
        },
        required={"models"},
    )
    models = _object(provider["models"], source, f"{field}.models")
    if not models:
        raise _error(source, f"{field}.models", "must contain at least one model")
    if "default" not in models:
        raise _error(source, f"{field}.models", "must define the 'default' role")
    for role, model in models.items():
        _string(role, source, f"{field}.models role")
        _string(model, source, f"{field}.models.{role}")

    if name != "bedrock":
        for required_field in ("base_url", "api_key_env", "timeout"):
            if required_field not in provider:
                raise _error(
                    source, field, f"is missing required field {required_field!r}"
                )
    if "base_url" in provider:
        _string(provider["base_url"], source, f"{field}.base_url")
    if "api_key_env" in provider:
        key_names = provider["api_key_env"]
        if isinstance(key_names, str):
            _string(key_names, source, f"{field}.api_key_env")
        elif isinstance(key_names, list) and key_names:
            for index, key_name in enumerate(key_names):
                _string(key_name, source, f"{field}.api_key_env[{index}]")
        else:
            raise _error(
                source,
                f"{field}.api_key_env",
                "must be a non-empty string or array of non-empty strings",
            )
    if "timeout" in provider:
        _number(
            provider["timeout"],
            source,
            f"{field}.timeout",
            exclusive_minimum=0,
        )
    if "reasoning_effort" in provider:
        if name != "openrouter":
            raise _error(
                source,
                f"{field}.reasoning_effort",
                "is currently supported only for the openrouter provider",
            )
        effort = _string(
            provider["reasoning_effort"],
            source,
            f"{field}.reasoning_effort",
        )
        supported_efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}
        if effort not in supported_efforts:
            choices = ", ".join(sorted(supported_efforts))
            raise _error(
                source,
                f"{field}.reasoning_effort",
                f"must be one of: {choices}",
            )

    if "ensembles" in provider:
        ensembles = _object(provider["ensembles"], source, f"{field}.ensembles")
        for ensemble_name, members in ensembles.items():
            _string(ensemble_name, source, f"{field}.ensembles name")
            if not isinstance(members, list) or not members:
                raise _error(
                    source,
                    f"{field}.ensembles.{ensemble_name}",
                    "must be a non-empty array",
                )
            for index, member_value in enumerate(members):
                member_field = f"{field}.ensembles.{ensemble_name}[{index}]"
                member = _object(member_value, source, member_field)
                _exact_keys(
                    member,
                    source,
                    member_field,
                    allowed={"name", "model"},
                    required={"name", "model"},
                )
                _string(member["name"], source, f"{member_field}.name")
                _string(member["model"], source, f"{member_field}.model")

    if "headers" in provider:
        headers = _object(provider["headers"], source, f"{field}.headers")
        _exact_keys(
            headers,
            source,
            f"{field}.headers",
            allowed={"site_url", "app_name"},
        )
        for header_name, header_value in headers.items():
            _nullable_string(header_value, source, f"{field}.headers.{header_name}")


def _validate_aws(value: Any, source: str) -> None:
    aws = _object(value, source, "aws")
    required = {"profile", "region", "s3_bucket", "batch_role_arn"}
    _exact_keys(aws, source, "aws", allowed=required, required=required)
    for name in ("profile", "s3_bucket", "batch_role_arn"):
        _nullable_string(aws[name], source, f"aws.{name}")
    _string(aws["region"], source, "aws.region")


def _validate_runtime(value: Any, source: str) -> None:
    runtime = _object(value, source, "runtime")
    runtime_fields = {"output_root", "annotation_output_dir", "bedrock_gate"}
    _exact_keys(
        runtime,
        source,
        "runtime",
        allowed=runtime_fields,
        required=runtime_fields,
    )
    _string(runtime["output_root"], source, "runtime.output_root")
    _string(runtime["annotation_output_dir"], source, "runtime.annotation_output_dir")
    gate = _object(runtime["bedrock_gate"], source, "runtime.bedrock_gate")
    gate_fields = {
        "enabled",
        "directory",
        "max_slots",
        "poll_seconds",
        "throttle_shrink_factor",
        "throttle_cooldown_seconds",
        "model_max_slots",
    }
    _exact_keys(
        gate,
        source,
        "runtime.bedrock_gate",
        allowed=gate_fields,
        required=gate_fields,
    )
    if not isinstance(gate["enabled"], bool):
        raise _error(source, "runtime.bedrock_gate.enabled", "must be a boolean")
    _string(gate["directory"], source, "runtime.bedrock_gate.directory")
    _number(
        gate["max_slots"],
        source,
        "runtime.bedrock_gate.max_slots",
        minimum=1,
        integer=True,
    )
    _number(
        gate["poll_seconds"],
        source,
        "runtime.bedrock_gate.poll_seconds",
        exclusive_minimum=0,
    )
    _number(
        gate["throttle_shrink_factor"],
        source,
        "runtime.bedrock_gate.throttle_shrink_factor",
        exclusive_minimum=0,
        maximum=1,
    )
    _number(
        gate["throttle_cooldown_seconds"],
        source,
        "runtime.bedrock_gate.throttle_cooldown_seconds",
        minimum=0,
    )
    model_limits = _object(
        gate["model_max_slots"], source, "runtime.bedrock_gate.model_max_slots"
    )
    for model, limit in model_limits.items():
        _string(model, source, "runtime.bedrock_gate.model_max_slots model")
        _number(
            limit,
            source,
            f"runtime.bedrock_gate.model_max_slots.{model}",
            minimum=1,
            integer=True,
        )


def validate_config(
    value: Mapping[str, Any], *, source: str = "<config object>"
) -> RuntimeConfig:
    """Validate a mapping and return an immutable configuration snapshot."""
    if not isinstance(value, dict):
        raise _error(source, "root", "must be an object")
    config = deepcopy(value)
    root_fields = {
        "$schema",
        "schema_version",
        "provider",
        "providers",
        "aws",
        "runtime",
    }
    _exact_keys(
        config,
        source,
        "root",
        allowed=root_fields,
        required={"schema_version", "provider", "providers", "aws", "runtime"},
    )
    if "$schema" in config:
        _string(config["$schema"], source, "$schema")
    if config["schema_version"] != SCHEMA_VERSION:
        raise _error(
            source,
            "schema_version",
            f"must be {SCHEMA_VERSION!r}, got {config['schema_version']!r}",
        )

    providers = _object(config["providers"], source, "providers")
    if not providers:
        raise _error(source, "providers", "must contain at least one provider")
    for raw_name, provider in providers.items():
        name = normalize_provider_name(raw_name)
        if name != raw_name:
            raise _error(
                source,
                f"providers.{raw_name}",
                f"must use canonical provider name {name!r}",
            )
        _validate_provider(name, provider, source)

    selected = normalize_provider_name(_string(config["provider"], source, "provider"))
    if selected not in providers:
        raise _error(
            source,
            "provider",
            f"selects {selected!r}, which is not defined under 'providers'",
        )
    config["provider"] = selected
    _validate_aws(config["aws"], source)
    _validate_runtime(config["runtime"], source)
    return RuntimeConfig(_freeze(config), source)


def _packaged_config() -> tuple[str, str]:
    resource = files("premise").joinpath("data", "default_config.json")
    if resource.is_file():
        return resource.read_text(encoding="utf-8"), "packaged default_config.json"
    repository_copy = Path(__file__).resolve().parents[1] / "config.json"
    if repository_copy.is_file():
        return (
            repository_copy.read_text(encoding="utf-8"),
            str(repository_copy.resolve()),
        )
    raise ConfigError(
        "PReMISE's packaged default_config.json is missing. Reinstall the package "
        "or pass an explicit configuration path."
    )


def resolve_config_path(
    path: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Resolve explicit path, ``PREMISE_CONFIG``, then local ``config.json``."""
    explicit = path or os.environ.get(CONFIG_ENV)
    if explicit:
        resolved = Path(explicit).expanduser().resolve()
        if not resolved.is_file():
            raise ConfigError(f"PReMISE config file not found: {resolved}")
        return resolved
    local = Path.cwd() / "config.json"
    return local.resolve() if local.is_file() else None


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    config: Mapping[str, Any] | RuntimeConfig | None = None,
) -> RuntimeConfig:
    """Load and validate a fresh config from a mapping, file, or package data."""
    if path is not None and config is not None:
        raise ConfigError("Pass either a config path or a config object, not both.")
    if isinstance(config, RuntimeConfig):
        return config
    if config is not None:
        return validate_config(config)

    resolved = resolve_config_path(path)
    if resolved is None:
        text, source = _packaged_config()
    else:
        source = str(resolved)
        text = resolved.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON in PReMISE config {source}: "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from None
    return validate_config(raw, source=source)


def _apply_overrides(
    base: RuntimeConfig,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    output_root: str | None = None,
) -> RuntimeConfig:
    if not any((provider, model, base_url, output_root)):
        return base
    value = base.to_dict()
    selected = normalize_provider_name(provider or base.provider)
    value["provider"] = selected
    if selected not in value["providers"]:
        raise ConfigError(
            f"Provider {selected!r} is not defined in configuration from {base.source}."
        )
    selected_provider = value["providers"][selected]
    if model:
        selected_provider["models"] = {
            role: model for role in selected_provider["models"]
        }
        selected_provider["models"]["default"] = model
    if base_url:
        selected_provider["base_url"] = base_url
    if output_root:
        value["runtime"]["output_root"] = output_root
    return validate_config(value, source=f"{base.source} with explicit overrides")


def configure(
    path: str | os.PathLike[str] | None = None,
    *,
    config: Mapping[str, Any] | RuntimeConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    output_root: str | None = None,
) -> RuntimeConfig:
    """Set the stable configuration snapshot for the current invocation."""
    global _ACTIVE_CONFIG
    loaded = load_config(path, config=config)
    _ACTIVE_CONFIG = _apply_overrides(
        loaded,
        provider=provider,
        model=model,
        base_url=base_url,
        output_root=output_root,
    )
    return _ACTIVE_CONFIG


def get_config() -> RuntimeConfig:
    """Return the active snapshot, loading it once when necessary."""
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        _ACTIVE_CONFIG = load_config()
    return _ACTIVE_CONFIG


def reset_config() -> None:
    """Clear the active snapshot. Intended for tests and embedding applications."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = None


@contextmanager
def config_context(
    path: str | os.PathLike[str] | None = None,
    *,
    config: Mapping[str, Any] | RuntimeConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    output_root: str | None = None,
) -> Iterator[RuntimeConfig]:
    """Temporarily install a validated configuration snapshot."""
    global _ACTIVE_CONFIG
    previous = _ACTIVE_CONFIG
    try:
        yield configure(
            path,
            config=config,
            provider=provider,
            model=model,
            base_url=base_url,
            output_root=output_root,
        )
    finally:
        _ACTIVE_CONFIG = previous


def get_provider_config(
    config: Mapping[str, Any] | RuntimeConfig,
    provider: str | None = None,
) -> Mapping[str, Any]:
    """Compatibility helper returning one validated provider block."""
    runtime_config = (
        config if isinstance(config, RuntimeConfig) else validate_config(config)
    )
    return runtime_config.provider_config(provider)
