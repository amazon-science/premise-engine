# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Resumable one-command adversarial rubric repair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from premise.rubric_refinement.adversarial.operators import OPERATORS

STATE_FILE = "repair-workflow.json"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _call_main(function, prog: str, argv: list[str]) -> None:
    saved = sys.argv
    try:
        sys.argv = [prog, *argv]
        function()
    finally:
        sys.argv = saved


def _run_seed(context: dict[str, Any]) -> None:
    from premise.rubric_refinement.adversarial.seed import seed

    seed(
        root=str(Path(context["output"]).parent),
        dataset=Path(context["input_preference_data"]).stem,
        dataset_path=context["output"],
        traits_file=context["policy_rubrics_file"],
        battles_file=context["input_preference_data"],
        vfr_file=context["vfr_results"],
        high_vfr_only=context["high_vfr_only"],
        vfr_threshold=context["vfr_threshold"],
        seed_value=context["seed"],
        n_screen=context["n_screen"],
        n_full=context["n_full"],
        n_eval=context["n_eval"],
    )


def _run_predict(context: dict[str, Any]) -> None:
    from premise.rubric_refinement.adversarial.repair import predict_prep_main

    argv = [
        "--root",
        context["output"],
        "--iter",
        "1",
        "--max-depth",
        str(len(context["operators"])),
        "--traits-per-bundle",
        str(context["traits_per_bundle"]),
        "--temperature",
        str(context["temperature"]),
        "--max-tokens",
        str(context["max_tokens"]),
        "--execute",
    ]
    for operator in context["operators"]:
        argv += ["--operator", operator]
    if context["model"]:
        argv += ["--model", context["model"]]
    if context["allow_large_bundles"]:
        argv.append("--allow-large-bundles")
    _call_main(
        predict_prep_main,
        "premise refine adversarial predict",
        argv,
    )


def _run_merge(context: dict[str, Any]) -> None:
    from premise.rubric_refinement.adversarial.repair import merge_bundles_main

    _call_main(
        merge_bundles_main,
        "premise refine adversarial merge",
        ["--root", context["output"], "--iter", "1"],
    )


def _rubrics_for(tree: dict[str, Any], operators: list[str]) -> list[dict[str, str]]:
    wanted = sorted(operators)
    rubrics = []
    trait_ids = sorted({node["trait_id"] for node in tree.get("nodes", [])})
    for trait_id in trait_ids:
        node = next(
            (
                candidate
                for candidate in tree["nodes"]
                if candidate["trait_id"] == trait_id
                and sorted(candidate.get("ops", [])) == wanted
            ),
            None,
        )
        if node is None:
            raise RuntimeError(
                f"repair output is missing operators {wanted} for trait {trait_id}"
            )
        payload = json.loads(Path(node["rubric_path"]).read_text())
        source = payload["rubrics"][0]
        rubrics.append({"id": source["id"], "description": source["description"]})
    return rubrics


def _materialize(context: dict[str, Any]) -> None:
    output = Path(context["output"])
    tree = json.loads((output / "tree.json").read_text())
    cumulative: list[str] = []

    before = {
        "schema_version": "1.0.0",
        "operators": [],
        "rubrics": _rubrics_for(tree, []),
    }
    _write_json(output / "before-rubric.json", before)

    for index, operator in enumerate(context["operators"], start=1):
        cumulative.append(operator)
        payload = {
            "schema_version": "1.0.0",
            "operators": list(cumulative),
            "rubrics": _rubrics_for(tree, cumulative),
        }
        _write_json(output / f"repair-{index}-{operator}.json", payload)
        if index == len(context["operators"]):
            _write_json(output / "final-rubric.json", payload)


_STAGE_RUNNERS = {
    "seed": _run_seed,
    "predict": _run_predict,
    "merge": _run_merge,
    "materialize": _materialize,
}


def run_repair_workflow(
    *,
    policy_rubrics_file: str,
    input_preference_data: str,
    output: str,
    operators: list[str],
    vfr_results: str | None = None,
    high_vfr_only: bool = False,
    vfr_threshold: float = 0.5,
    seed: int = 42,
    n_screen: int = 20,
    n_full: int = 200,
    n_eval: int = 50,
    traits_per_bundle: int = 10,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 16000,
    allow_large_bundles: bool = False,
) -> dict[str, Any]:
    """Run seed, predict, merge, and final-rubric materialization."""
    if not operators:
        raise ValueError("at least one repair operator is required")
    if len(set(operators)) != len(operators):
        raise ValueError("repair operators must not be repeated")

    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    state_path = output_path / STATE_FILE
    identity = {
        "policy_rubrics_file": str(Path(policy_rubrics_file).resolve()),
        "input_preference_data": str(Path(input_preference_data).resolve()),
        "vfr_results": str(Path(vfr_results).resolve()) if vfr_results else None,
        "operators": list(operators),
        "high_vfr_only": high_vfr_only,
        "vfr_threshold": vfr_threshold,
        "seed": seed,
        "n_screen": n_screen,
        "n_full": n_full,
        "n_eval": n_eval,
        "traits_per_bundle": traits_per_bundle,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "allow_large_bundles": allow_large_bundles,
    }
    stages = ["seed", "predict", "merge", "materialize"]

    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise ValueError(
                f"{state_path} belongs to different inputs or options; "
                "choose a new --output directory"
            )
    else:
        state = {
            "schema_version": "1.0",
            "kind": "adversarial_repair",
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
        print(f"\n=== Repair: {name} ===")
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
    state["final_rubric"] = str(output_path / "final-rubric.json")
    _write_json(state_path, state)
    print(f"\nRepair workflow complete. Final rubric: {state['final_rubric']}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run adversarial rubric repair as one resumable workflow."
    )
    parser.add_argument("--policy-rubrics-file", required=True)
    parser.add_argument("--input-preference-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--operator",
        dest="operators",
        action="append",
        choices=OPERATORS,
        required=True,
        help="Repair operator; repeat to define the ordered repair path.",
    )
    parser.add_argument(
        "--vfr-results",
        default=None,
        help="Optional prior adversarial results. Repair does not require them.",
    )
    parser.add_argument("--high-vfr-only", action="store_true")
    parser.add_argument("--vfr-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-screen", type=int, default=20)
    parser.add_argument("--n-full", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--traits-per-bundle", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--allow-large-bundles", action="store_true")
    args = parser.parse_args()
    run_repair_workflow(
        policy_rubrics_file=args.policy_rubrics_file,
        input_preference_data=args.input_preference_data,
        output=args.output,
        operators=args.operators,
        vfr_results=args.vfr_results,
        high_vfr_only=args.high_vfr_only,
        vfr_threshold=args.vfr_threshold,
        seed=args.seed,
        n_screen=args.n_screen,
        n_full=args.n_full,
        n_eval=args.n_eval,
        traits_per_bundle=args.traits_per_bundle,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        allow_large_bundles=args.allow_large_bundles,
    )


if __name__ == "__main__":
    main()
