# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Stable fingerprints used to validate resumable workflow artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from premise.config import RuntimeConfig, get_config


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with stable key ordering and separators."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_config_identity(
    config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    """Return non-secret provider details plus a full effective-config digest."""
    selected = config or get_config()
    provider_config = selected.provider_config()
    return {
        "provider": selected.provider,
        "models": {
            str(role): str(model)
            for role, model in sorted(provider_config["models"].items())
        },
        "config_sha256": canonical_sha256(selected.to_dict()),
    }


def changed_identity_fields(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[str]:
    """List top-level identity fields that differ or are missing."""
    previous = previous or {}
    return sorted(
        key
        for key in set(previous) | set(current)
        if previous.get(key) != current.get(key)
    )
