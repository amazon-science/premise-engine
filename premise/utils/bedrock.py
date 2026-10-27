# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""AWS Bedrock utilities and the legacy provider-compatible inference shim."""

import hashlib
import json
import os
import time

import boto3
import botocore.exceptions

from premise.config import get_config

# Transitional aliases for existing call signatures. Values come exclusively
# from the active, validated config snapshot and contain no Python fallbacks.
_CONFIG = get_config()
DEFAULT_REGION = _CONFIG.aws("region", required=True)
DEFAULT_S3_BUCKET = _CONFIG.aws("s3_bucket")
DEFAULT_BATCH_ROLE_ARN = _CONFIG.aws("batch_role_arn")


def require_aws_setting(key: str, value=None):
    """Return an explicit AWS value or require it from active configuration."""
    if value is not None and value != "":
        return value
    return get_config().aws(key, required=True)


def resolve_batch_settings(
    s3_bucket: str | None = None, role_arn: str | None = None
) -> tuple[str, str]:
    """Resolve and validate resources required by Bedrock batch inference."""
    return (
        str(require_aws_setting("s3_bucket", s3_bucket)),
        str(require_aws_setting("batch_role_arn", role_arn)),
    )


def resolve_profile() -> str | None:
    """Return the configured profile, or let boto3 use its credential chain."""
    return get_config().aws("profile")


def make_session(region: str = DEFAULT_REGION):
    """A boto3 Session for `region`, with a readable error for a bad profile."""
    profile = resolve_profile()
    if profile is None:
        return boto3.Session(region_name=region)
    try:
        return boto3.Session(profile_name=profile, region_name=region)
    except botocore.exceptions.ProfileNotFound:
        raise SystemExit(
            f"AWS profile {profile!r} does not exist in your AWS config.\n"
            f"Update aws.profile in {get_config().source}, or set it to null to "
            "use the standard AWS credential chain."
        ) from None


def _get_session(region: str = DEFAULT_REGION):
    """Get a boto3 session from the configured profile or credential chain."""
    return make_session(region)


def get_bedrock_client(region: str = DEFAULT_REGION):
    """Get the active inference client.

    The historical name is retained for compatibility. When the active provider
    is Bedrock this returns a Bedrock Runtime client; otherwise it returns the
    configured HTTP provider adapter.
    """
    from premise.providers import get_provider_client, provider_name

    if provider_name() != "bedrock":
        return get_provider_client()
    return _get_session(region).client("bedrock-runtime")


def get_bedrock_management_client(region: str = DEFAULT_REGION):
    """Get Bedrock management client (for batch job APIs)."""
    return _get_session(region).client("bedrock")


def get_s3_client(region: str = DEFAULT_REGION):
    """Get S3 client."""
    return _get_session(region).client("s3")


def invoke_model(
    client,
    model_id: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_retries: int = 3,
    system: str | None = None,
    return_reasoning: bool = False,
):
    """Invoke the active provider with retry logic.

    Calls acquire a slot from the shared bedrock_gate token bucket (if enabled
    via BEDROCK_MAX_SLOTS > 0), so cross-process Bedrock concurrency stays
    bounded when many pipelines run in parallel.

    When `return_reasoning=True`, returns (text, reasoning_text) — useful
    for reasoning-capable models like DeepSeek-R1 whose Converse output
    contains a separate `reasoningContent` block. For non-reasoning models
    the second value is an empty string.
    """
    from premise.providers import ProviderClient, configured_model

    if isinstance(client, ProviderClient):
        text = client.generate(
            configured_model(model_id),
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            system=system,
        )
        if return_reasoning:
            return text, ""
        return text

    messages = [{"role": "user", "content": [{"text": prompt}]}]
    kwargs = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]

    # Per-model gate: acquire a slot for this specific model before calling.
    gate_enabled = bool(get_config().runtime("bedrock_gate")["enabled"])
    _gate_acquire = _gate_release = _gate_throttle = None
    if gate_enabled:
        try:
            from premise.utils.bedrock_gate import (
                acquire as _gate_acquire,
            )
            from premise.utils.bedrock_gate import (
                release as _gate_release,
            )
            from premise.utils.bedrock_gate import (
                report_throttle as _gate_throttle,
            )

            _gate_acquire(model_id)
        except Exception:
            gate_enabled = False  # fall through silently if gate misconfigured

    try:
        for attempt in range(max_retries):
            try:
                response = client.converse(**kwargs)
                content = response["output"]["message"]["content"]
                text = ""
                reasoning = ""
                for block in content:
                    if "text" in block:
                        text += block["text"]
                    elif "reasoningContent" in block:
                        reasoning += (
                            block["reasoningContent"]
                            .get("reasoningText", {})
                            .get("text", "")
                        )
                if return_reasoning:
                    return text, reasoning
                return text or (content[0].get("text", "") if content else "")
            except client.exceptions.ThrottlingException:
                # Inform the gate so other callers on this model back off too.
                if gate_enabled and _gate_throttle:
                    try:
                        _gate_throttle(model_id)
                    except Exception:
                        pass
                wait = 2**attempt * 5
                print(f"  Throttled on {model_id}, waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Error: {e}, retrying...")
                    time.sleep(2**attempt)
                else:
                    print(f"  Failed after {max_retries} attempts: {e}")
                    return (None, "") if return_reasoning else None
        return (None, "") if return_reasoning else None
    finally:
        if gate_enabled and _gate_release:
            try:
                _gate_release(model_id)
            except Exception:
                pass


# ============================================================
# Batch Inference Utilities
# ============================================================


def get_batch_model_input(
    model_id: str, prompt: str, temperature: float = 0.1, max_tokens: int = 2000
) -> dict:
    """Build the model-specific modelInput dict for Bedrock batch inference."""
    mid = model_id.lower()

    if "anthropic" in mid or "claude" in mid:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
    elif "deepseek" in mid:
        # DeepSeek V3/R1 on Bedrock batch require messages format
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
    elif "llama" in mid or "meta" in mid:
        return {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
    else:
        # Qwen, Mistral, and other OpenAI-compatible models
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }


def extract_batch_model_response(
    model_output: dict,
    model_id: str,
    prefer_reasoning: bool = False,
) -> str | None:
    """Extract text from a batch inference modelOutput dict.

    When `prefer_reasoning=True`, returns the model's reasoning trace
    (e.g. DeepSeek-R1's `reasoning_content` field, or text inside
    `<think>...</think>` / `<reasoning>...</reasoning>` tags) when present.
    Falls back to the visible answer otherwise.
    """
    if not model_output:
        return None

    # Converse-style (Claude)
    if "content" in model_output:
        content = model_output.get("content", [])
        if content and isinstance(content, list):
            if prefer_reasoning:
                for blk in content:
                    if isinstance(blk, dict) and "reasoningContent" in blk:
                        rt = blk["reasoningContent"].get("reasoningText", {})
                        if isinstance(rt, dict) and rt.get("text"):
                            return rt["text"]
            return content[0].get("text", "")

    # OpenAI-style (DeepSeek, Qwen, Mistral, Kimi)
    if "choices" in model_output:
        choices = model_output["choices"]
        if choices:
            msg = choices[0].get("message", {})
            # DeepSeek-R1 batch output exposes reasoning_content alongside content.
            if prefer_reasoning:
                rc = msg.get("reasoning_content") or msg.get("reasoningContent")
                if isinstance(rc, str) and rc.strip():
                    return rc
            text = msg.get("content") or ""
            # Handle <think>...</think> or <reasoning>...</reasoning> tags.
            import re as _re

            for tag in ("think", "reasoning"):
                pattern_open = f"<{tag}>"
                if pattern_open in text:
                    if prefer_reasoning:
                        match = _re.search(
                            rf"<{tag}>(.*?)</{tag}>",
                            text,
                            flags=_re.DOTALL,
                        )
                        if match:
                            return match.group(1).strip()
                    else:
                        outside = _re.sub(
                            rf"<{tag}>.*?</{tag}>\s*",
                            "",
                            text,
                            flags=_re.DOTALL,
                        ).strip()
                        if outside:
                            text = outside
                        else:
                            match = _re.search(
                                rf"<{tag}>(.*?)</{tag}>",
                                text,
                                flags=_re.DOTALL,
                            )
                            text = match.group(1).strip() if match else text
                    break
            return text

    # Raw text field (some models)
    if "generation" in model_output:
        return model_output["generation"]

    # Fallback: try to find any text
    for key in ("output", "text", "completion"):
        if key in model_output:
            val = model_output[key]
            if isinstance(val, str):
                return val

    return json.dumps(model_output)


def upload_to_s3(
    local_path: str,
    bucket: str | None,
    key: str,
    region: str = DEFAULT_REGION,
    max_retries: int = 10,
) -> str:
    """Upload a local file to S3. Returns the s3:// URI.

    Uses boto3 with explicit multipart config (5MB parts, single-threaded)
    for reliable uploads on constrained networks. Falls back to AWS CLI
    for very large files if boto3 also fails.
    """
    import random

    import boto3
    from boto3.s3.transfer import TransferConfig

    bucket = str(require_aws_setting("s3_bucket", bucket))
    s3_uri = f"s3://{bucket}/{key}"
    # 5MB parts, single-thread to avoid connection exhaustion
    config = TransferConfig(
        multipart_threshold=5 * 1024 * 1024,
        multipart_chunksize=5 * 1024 * 1024,
        max_concurrency=1,
        use_threads=False,
    )
    for attempt in range(max_retries):
        try:
            s3_client = boto3.client("s3", region_name=region)
            s3_client.upload_file(local_path, bucket, key, Config=config)
            return s3_uri
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1) + random.uniform(0, 5)
                print(
                    f"  S3 upload failed (attempt {attempt + 1}): {e}, retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                raise
    return s3_uri


def download_batch_results(
    bucket: str, prefix: str, local_dir: str, region: str = DEFAULT_REGION
) -> list:
    """Download all .jsonl.out files under an S3 prefix. Returns local file paths."""
    s3 = get_s3_client(region)
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = []
    os.makedirs(local_dir, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl.out"):
                continue
            local_path = os.path.join(local_dir, os.path.basename(key))
            s3.download_file(bucket, key, local_path)
            downloaded.append(local_path)
    return downloaded


def compute_file_hash(file_path: str) -> str:
    """MD5 hash of a file for cache keying."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def submit_batch_job(
    bedrock_client,
    s3_input_uri: str,
    s3_output_uri: str,
    job_name: str,
    model_id: str,
    role_arn: str | None,
) -> str:
    """Submit a Bedrock batch inference job. Returns the job ARN.

    Idempotent: if a job with the same name already exists, returns its ARN.
    """
    role_arn = str(require_aws_setting("batch_role_arn", role_arn))
    try:
        response = bedrock_client.create_model_invocation_job(
            jobName=job_name,
            modelId=model_id,
            roleArn=role_arn,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": s3_input_uri,
                    "s3InputFormat": "JSONL",
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": s3_output_uri,
                }
            },
        )
        return response["jobArn"]
    except bedrock_client.exceptions.ConflictException:
        # Job with this name already exists — find and return it
        print(f"  Job '{job_name}' already exists, looking up ARN...")
        paginator = bedrock_client.get_paginator("list_model_invocation_jobs")
        for page in paginator.paginate(nameContains=job_name):
            for job in page.get("invocationJobSummaries", []):
                if job["jobName"] == job_name:
                    print(
                        f"  Found existing job: {job['jobArn']} (status: {job['status']})"
                    )
                    return job["jobArn"]
        raise


def refresh_credentials(region: str = DEFAULT_REGION):
    """Recreate the Bedrock client from the standard AWS credential chain."""
    try:
        return get_bedrock_management_client(region)
    except Exception as e:
        print(f"  [credential reload failed: {e}]")
        return None


def wait_for_batch_job(
    bedrock_client,
    job_arn: str,
    poll_interval: int = 60,
    timeout: int = 7200,
    region: str = DEFAULT_REGION,
) -> dict:
    """Poll a batch job until completion. Returns the job response dict.

    Auto-refreshes credentials on ExpiredTokenException.
    Raises RuntimeError on failure or timeout.
    """
    start = time.time()
    while True:
        try:
            response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        except Exception as e:
            if "ExpiredToken" in str(e):
                print("  Token expired, refreshing credentials...")
                new_client = refresh_credentials(region)
                if new_client:
                    bedrock_client = new_client
                    continue
            raise

        status = response["status"]
        elapsed = int(time.time() - start)

        if status == "Completed":
            print(f"  Batch job completed in {elapsed}s")
            return response
        elif status in ("Failed", "Stopped"):
            msg = response.get("message", "unknown error")
            raise RuntimeError(f"Batch job {status}: {msg}")
        else:
            print(f"  Batch job status: {status} ({elapsed}s elapsed)")
            if elapsed > timeout:
                raise RuntimeError(
                    f"Batch job timed out after {timeout}s (status: {status})"
                )
            time.sleep(poll_interval)
