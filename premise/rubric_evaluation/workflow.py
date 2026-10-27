# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Resumable quick and paper-protocol rubric audit workflows."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from premise.provenance import (
    active_config_identity,
    changed_identity_fields,
    file_sha256,
)

STATE_FILE = "audit-workflow.json"
WORKFLOW_REVISION = 2


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _invoke_module(module: str, argv: list[str]) -> None:
    saved = sys.argv
    try:
        sys.argv = [f"premise audit {module.rsplit('.', 1)[-1]}", *argv]
        try:
            runpy.run_module(module, run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError(f"{module} exited with status {exc.code}") from exc
    finally:
        sys.argv = saved


def _run_structural_budget(context: dict[str, Any]) -> None:
    from premise.api import audit_structural

    result = audit_structural(context["policy_rubrics_file"])
    _write_json(Path(context["output"]) / "structural-budget.json", result)


def _run_structural_adequacy(context: dict[str, Any]) -> None:
    _invoke_module(
        "premise.rubric_evaluation.meta_trait",
        [
            "--traits",
            context["policy_rubrics_file"],
            "--output",
            str(Path(context["output"]) / "structural-adequacy"),
        ],
    )


def _run_pref_fit(context: dict[str, Any]) -> None:
    argv = [
        "--traits",
        context["policy_rubrics_file"],
        "--samples",
        context["input_preference_data"],
        "--output",
        str(Path(context["output"]) / "pref-fit"),
        "--max-samples",
        str(context["max_samples"]),
    ]
    if context["no_batch"]:
        argv.append("--no-batch")
    _invoke_module("premise.rubric_evaluation.pref_fit", argv)


def _run_consistency(context: dict[str, Any]) -> None:
    argv = [
        "--traits",
        context["policy_rubrics_file"],
        "--samples",
        context["input_preference_data"],
        "--output",
        str(Path(context["output"]) / "consistency"),
        "--n-runs",
        str(context["n_runs"]),
        "--max-samples",
        str(context["max_samples"]),
    ]
    if context["no_batch"]:
        argv.append("--no-batch")
    if context["no_reasoning"]:
        argv.append("--no-reasoning")
    _invoke_module("premise.rubric_evaluation.consistency", argv)


def _run_adversarial(context: dict[str, Any]) -> None:
    argv = [
        "--traits",
        context["policy_rubrics_file"],
        "--samples",
        context["input_preference_data"],
        "--output",
        str(Path(context["output"]) / "adversarial"),
        "--n-attempts",
        str(context["n_attempts"]),
        "--workers",
        str(context["workers"]),
    ]
    if context["no_batch"]:
        argv.append("--no-batch")
    _invoke_module("premise.rubric_evaluation.adversarial", argv)


def _run_ira(context: dict[str, Any]) -> None:
    argv = [
        "--traits",
        context["policy_rubrics_file"],
        "--samples",
        context["input_preference_data"],
        "--output",
        str(Path(context["output"]) / "ira"),
        "--max-samples",
        str(context["max_samples"]),
    ]
    if context["no_batch"]:
        argv.append("--no-batch")
    _invoke_module("premise.rubric_evaluation.ira", argv)


def _run_paraphrase(context: dict[str, Any]) -> None:
    argv = [
        "--traits",
        context["policy_rubrics_file"],
        "--samples",
        context["input_preference_data"],
        "--output",
        str(Path(context["output"]) / "paraphrase"),
        "--max-samples",
        str(context["max_samples"]),
        "--k",
        "5",
    ]
    if context["no_batch"]:
        argv.append("--no-batch")
    _invoke_module("premise.rubric_evaluation.paraphrase", argv)


def _run_bt_scoring(context: dict[str, Any]) -> None:
    argv = [
        "--traits",
        context["policy_rubrics_file"],
        "--battles",
        context["input_preference_data"],
        "--output",
        str(Path(context["output"]) / "bt-scoring"),
        "--max-samples",
        str(context["bt_max_samples"]),
    ]
    if context["no_batch"]:
        argv.append("--no-batch")
    _invoke_module("premise.rubric_evaluation.bt_scoring", argv)


def _run_bt_preference_fit(context: dict[str, Any]) -> None:
    _invoke_module(
        "premise.rubric_evaluation.pref_fit_analysis",
        [
            "--bt-scores",
            str(Path(context["output"]) / "bt-scoring" / "bt_scores.jsonl"),
            "--battles",
            context["input_preference_data"],
            "--output",
            str(Path(context["output"]) / "preference-fit.json"),
        ],
    )


_STAGE_RUNNERS = {
    "structural_budget": _run_structural_budget,
    "structural_adequacy": _run_structural_adequacy,
    "preference_fit_direct": _run_pref_fit,
    "consistency": _run_consistency,
    "ira": _run_ira,
    "paraphrase": _run_paraphrase,
    "bt_scoring": _run_bt_scoring,
    "bt_preference_fit": _run_bt_preference_fit,
    "adversarial_vfr": _run_adversarial,
}


def run_audit_workflow(
    *,
    policy_rubrics_file: str,
    input_preference_data: str,
    output: str,
    paper_protocol: bool = False,
    include_adversarial: bool = False,
    n_runs: int = 5,
    n_attempts: int | None = None,
    max_samples: int | None = None,
    bt_max_samples: int | None = None,
    workers: int = 10,
    no_batch: bool = False,
    no_reasoning: bool = False,
) -> dict[str, Any]:
    """Run required audits, resuming after the last completed stage."""
    if n_attempts is None:
        n_attempts = 500 if paper_protocol else 50
    if max_samples is None:
        max_samples = 500 if paper_protocol else 200
    if bt_max_samples is None:
        bt_max_samples = 1000 if paper_protocol else max_samples
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    state_path = output_path / STATE_FILE
    policy_path = Path(policy_rubrics_file).expanduser().resolve()
    preferences_path = Path(input_preference_data).expanduser().resolve()
    identity = {
        "workflow_revision": WORKFLOW_REVISION,
        "policy_rubrics_file": str(policy_path),
        "policy_rubrics_sha256": file_sha256(policy_path),
        "input_preference_data": str(preferences_path),
        "input_preference_data_sha256": file_sha256(preferences_path),
        "include_adversarial": include_adversarial,
        "paper_protocol": paper_protocol,
        "n_runs": n_runs,
        "n_attempts": n_attempts,
        "max_samples": max_samples,
        "bt_max_samples": bt_max_samples,
        "workers": workers,
        "no_batch": no_batch,
        "no_reasoning": no_reasoning,
        **active_config_identity(),
    }
    if paper_protocol:
        stages = [
            "structural_budget",
            "structural_adequacy",
            "ira",
            "consistency",
            "paraphrase",
            "bt_scoring",
            "bt_preference_fit",
            "adversarial_vfr",
        ]
    else:
        stages = ["structural_budget", "preference_fit_direct", "consistency"]
    if include_adversarial and "adversarial_vfr" not in stages:
        stages.append("adversarial_vfr")

    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            changed = ", ".join(
                changed_identity_fields(state.get("identity"), identity)
            )
            raise ValueError(
                f"{state_path} does not match the current audit invocation "
                f"(changed: {changed}); choose a new --output directory"
            )
    else:
        state = {
            "schema_version": "1.1",
            "kind": "audit",
            "identity": identity,
            "stages": {name: {"status": "pending"} for name in stages},
            "created_at": _now(),
        }
        _write_json(state_path, state)

    context = {**identity, "output": str(output_path)}
    for name in stages:
        stage = state["stages"].setdefault(name, {"status": "pending"})
        if stage.get("status") == "completed":
            print(f"[resume] {name}: already completed")
            continue
        stage.update({"status": "running", "started_at": _now()})
        stage.pop("error", None)
        state["updated_at"] = _now()
        _write_json(state_path, state)
        print(f"\n=== Audit: {name} ===")
        try:
            _STAGE_RUNNERS[name](context)
        except Exception as exc:
            stage.update({"status": "failed", "error": str(exc), "updated_at": _now()})
            state["updated_at"] = _now()
            _write_json(state_path, state)
            raise
        stage.update({"status": "completed", "completed_at": _now()})
        state["updated_at"] = _now()
        _write_json(state_path, state)

    state["status"] = "completed"
    state["completed_at"] = _now()
    _write_json(state_path, state)
    print(f"\nAudit workflow complete. State: {state_path}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the required rubric audits as one resumable workflow."
    )
    parser.add_argument("--policy-rubrics-file", required=True)
    parser.add_argument("--input-preference-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help="Run the full paper audit: model-panel structure, all three "
        "reliability checks, BT preference fit, and 500-attempt VFR.",
    )
    parser.add_argument(
        "--include-adversarial",
        action="store_true",
        help="Also run adversarial Verified Fool Rate; optional because it is costly.",
    )
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-attempts", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--bt-max-samples",
        type=int,
        default=None,
        help="Preference pairs for paper BT scoring (default: 1000 in "
        "--paper-protocol).",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--no-batch", action="store_true")
    parser.add_argument("--no-reasoning", action="store_true")
    args = parser.parse_args()
    n_attempts = args.n_attempts
    if n_attempts is None:
        n_attempts = 500 if args.paper_protocol else 50
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = 500 if args.paper_protocol else 200
    try:
        run_audit_workflow(
            policy_rubrics_file=args.policy_rubrics_file,
            input_preference_data=args.input_preference_data,
            output=args.output,
            paper_protocol=args.paper_protocol,
            include_adversarial=args.include_adversarial,
            n_runs=args.n_runs,
            n_attempts=n_attempts,
            max_samples=max_samples,
            bt_max_samples=args.bt_max_samples,
            workers=args.workers,
            no_batch=args.no_batch,
            no_reasoning=args.no_reasoning,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"premise audit run: error: {exc}\n")


if __name__ == "__main__":
    main()
