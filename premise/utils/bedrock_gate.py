# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Per-model file-based token buckets for Bedrock API concurrency control.

Rationale: Bedrock quotas are per-model, so a single global cap is wasteful.
We maintain one slot file per model and cap each independently. Adaptive
throttle handling shrinks a model's cap on `ThrottlingException` and auto-
recovers after a cooldown window.

Usage:
    from premise.utils.bedrock_gate import bedrock_gate, report_throttle
    with bedrock_gate(model_id):
        # call Bedrock
        ...
    # on ThrottlingException:
    report_throttle(model_id)

All limits and paths come from ``runtime.bedrock_gate`` in the active config.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import time

from premise.config import get_config


def _settings():
    return get_config().runtime("bedrock_gate")


def _model_key(model_id: str) -> str:
    """Produce a filesystem-safe key from a Bedrock model id."""
    if not model_id:
        return "default"
    stripped = re.sub(r"^(us|global)\.", "", model_id.lower())
    stripped = stripped.split(":")[0]
    stripped = re.sub(r"-v\d+$", "", stripped)
    return re.sub(r"[^a-z0-9]+", "_", stripped).strip("_") or "default"


def _slot_path(model_id: str) -> str:
    directory = str(_settings()["directory"])
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"slots.{_model_key(model_id)}.json")


def _max_slots(model_id: str) -> int:
    settings = _settings()
    overrides = settings["model_max_slots"]
    configured = overrides.get(model_id) or overrides.get(_model_key(model_id))
    return int(configured or settings["max_slots"])


def _read_state(fh) -> dict:
    fh.seek(0)
    raw = fh.read().strip()
    if not raw:
        return {"count": 0, "throttle_until": 0.0, "throttle_factor": 1.0}
    try:
        d = json.loads(raw)
        return {
            "count": int(d.get("count", 0)),
            "throttle_until": float(d.get("throttle_until", 0.0)),
            "throttle_factor": float(d.get("throttle_factor", 1.0)),
        }
    except Exception:
        return {"count": 0, "throttle_until": 0.0, "throttle_factor": 1.0}


def _write_state(fh, state: dict):
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps(state))
    fh.flush()


def _ensure_file(path):
    if not os.path.exists(path):
        try:
            with open(path, "x") as f:
                f.write(
                    json.dumps(
                        {"count": 0, "throttle_until": 0.0, "throttle_factor": 1.0}
                    )
                )
        except FileExistsError:
            pass


def _effective_max(model_id: str, state: dict) -> int:
    base = _max_slots(model_id)
    # If in cooldown, apply shrink
    if time.time() < state.get("throttle_until", 0):
        return max(1, int(base * state.get("throttle_factor", 1.0)))
    # Cooldown expired — reset factor opportunistically on next write
    return base


def acquire(model_id: str | None = None, timeout: float = 300.0):
    if not _settings()["enabled"]:
        return
    path = _slot_path(model_id or "default")
    _ensure_file(path)
    deadline = time.time() + timeout
    while True:
        with open(path, "r+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                state = _read_state(fh)
                eff_max = _effective_max(model_id, state)
                # Reset shrink factor if cooldown expired
                if (
                    time.time() >= state.get("throttle_until", 0)
                    and state.get("throttle_factor", 1.0) < 1.0
                ):
                    state["throttle_factor"] = 1.0
                    state["throttle_until"] = 0.0
                    _write_state(fh, state)
                if state["count"] < eff_max:
                    state["count"] += 1
                    _write_state(fh, state)
                    return
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        if time.time() > deadline:
            raise TimeoutError(f"bedrock_gate acquire timeout model={model_id}")
        time.sleep(float(_settings()["poll_seconds"]))


def release(model_id: str | None = None):
    if not _settings()["enabled"]:
        return
    path = _slot_path(model_id or "default")
    _ensure_file(path)
    with open(path, "r+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            state = _read_state(fh)
            state["count"] = max(0, state["count"] - 1)
            _write_state(fh, state)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def report_throttle(model_id: str | None = None):
    """Shrink the model's effective cap temporarily in response to Bedrock throttling."""
    settings = _settings()
    if not settings["enabled"]:
        return
    path = _slot_path(model_id or "default")
    _ensure_file(path)
    with open(path, "r+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            state = _read_state(fh)
            # Shrink the factor further, but never below 0.1 of base
            new_factor = max(
                0.1,
                state.get("throttle_factor", 1.0)
                * float(settings["throttle_shrink_factor"]),
            )
            state["throttle_factor"] = new_factor
            state["throttle_until"] = time.time() + float(
                settings["throttle_cooldown_seconds"]
            )
            _write_state(fh, state)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def bedrock_gate(model_id: str | None = None, timeout: float = 300.0):
    acquire(model_id, timeout)
    try:
        yield
    finally:
        release(model_id)


def reset(model_id: str | None = None):
    path = _slot_path(model_id or "default")
    _ensure_file(path)
    with open(path, "r+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            _write_state(
                fh, {"count": 0, "throttle_until": 0.0, "throttle_factor": 1.0}
            )
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def status_all() -> dict:
    """Dump every slot file found in the gate dir."""
    out = {}
    directory = str(_settings()["directory"])
    if not os.path.exists(directory):
        return out
    for fn in os.listdir(directory):
        if not fn.startswith("slots.") or not fn.endswith(".json"):
            continue
        key = fn[len("slots.") : -len(".json")]
        with open(os.path.join(directory, fn)) as fh:
            try:
                d = json.load(fh)
            except Exception:
                d = {}
        out[key] = d
    return out


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        st = status_all()
        if not st:
            print("(no slot files)")
        else:
            print(
                f"{'model_key':<25} {'count':>6} {'factor':>8} {'throttle_until':>18}"
            )
            print("-" * 60)
            for k, v in sorted(st.items()):
                tu = v.get("throttle_until", 0)
                tu_s = (
                    "—" if tu == 0 or time.time() >= tu else f"{tu - time.time():+.0f}s"
                )
                print(
                    f"{k:<25} {v.get('count', '?'):>6} {v.get('throttle_factor', '?'):>8} {tu_s:>18}"
                )
    elif cmd == "reset":
        target = sys.argv[2] if len(sys.argv) > 2 else None
        if target:
            reset(target)
            print(f"reset {target}")
        else:
            # reset all
            st = status_all()
            for k in st:
                path = os.path.join(str(_settings()["directory"]), f"slots.{k}.json")
                with open(path, "w") as fh:
                    fh.write(
                        json.dumps(
                            {"count": 0, "throttle_until": 0.0, "throttle_factor": 1.0}
                        )
                    )
            print(f"reset {len(st)} slot files")
    else:
        print(f"unknown: {cmd}")
