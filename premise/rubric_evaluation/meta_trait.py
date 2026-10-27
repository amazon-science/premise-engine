# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Meta-Trait / Structural Analysis Experiment.

Evaluates whether rubric definitions satisfy 5 design criteria via LLM panel (majority vote).
No sample data needed — operates on rubric definitions alone.

Usage:
    premise audit meta-trait --traits <traits.json> --output results/meta_trait/
"""

import argparse
import json
import math
import os

from premise.config import get_config
from premise.utils import DEFAULT_REGION, get_bedrock_client, invoke_model, load_traits
from premise.utils.metrics import parse_verdict
from premise.utils.prompts import META_TRAIT_JUDGE_PROMPT

META_TRAITS = [
    "no_conflicting_requirements",
    "unambiguous_scope",
    "response_observable",
    "atomic",
    "operationalizable",
]

META_TRAIT_DESCRIPTIONS = {
    "no_conflicting_requirements": "The rubric does not contain internally contradictory rules. No requirement in the definition makes it impossible to satisfy another requirement in the same definition.",
    "unambiguous_scope": "The rubric is clear enough that different judges would apply it consistently. The boundaries of what counts as adherence vs. violation are well-defined.",
    "response_observable": "The rubric can be assessed solely from the response text, without needing external knowledge, fact-checking, ground truth, or information not present in the prompt and response.",
    "atomic": "The rubric measures a single behavioral dimension, not multiple conflated properties. It does not combine distinct qualities (e.g., 'clear AND concise' would fail atomicity).",
    "operationalizable": "The rubric can be expressed as a concrete, answerable scoring question. A judge could turn this into a specific yes/no or scaled question without needing further interpretation.",
}


def run_meta_trait_experiment(
    traits_file: str,
    output_dir: str,
    judge_models: list | None = None,
    region: str = DEFAULT_REGION,
):
    if judge_models is None:
        from premise.providers import model_for_role, provider_name

        if provider_name() == "bedrock":
            judge_models = [
                (str(member["name"]), str(member["model"]))
                for member in get_config().ensemble("structural")
            ]
        else:
            selected = model_for_role("judge")
            judge_models = [(selected, selected)]

    os.makedirs(output_dir, exist_ok=True)
    client = get_bedrock_client(region)
    traits = load_traits(traits_file)

    results = {"traits": {}}
    total_calls = len(traits) * len(META_TRAITS) * len(judge_models)
    call_count = 0

    for trait in traits:
        trait_id = trait["id"]
        trait_desc = trait["description"]
        print(f"\nEvaluating: {trait_id}")

        per_judge_meta = {}
        per_judge_scores = {}

        for judge_name, judge_model_id in judge_models:
            per_judge_meta[judge_name] = {}
            pass_count = 0

            for mt in META_TRAITS:
                call_count += 1
                print(f"  [{call_count}/{total_calls}] {judge_name} → {mt}")

                prompt = META_TRAIT_JUDGE_PROMPT.format(
                    trait_id=trait_id,
                    trait_description=trait_desc,
                    meta_trait_name=mt,
                    meta_trait_description=META_TRAIT_DESCRIPTIONS[mt],
                )

                response = invoke_model(client, judge_model_id, prompt, temperature=0.0)
                verdict = parse_verdict(response)
                per_judge_meta[judge_name][mt] = verdict
                if verdict == "PASS":
                    pass_count += 1

            per_judge_scores[judge_name] = f"{pass_count}/5"

        # Majority vote per meta-trait
        majority_meta = {}
        for mt in META_TRAITS:
            votes = [per_judge_meta[j][mt] for j in per_judge_meta]
            pass_votes = sum(1 for v in votes if v == "PASS")
            required = math.floor(len(votes) / 2) + 1
            majority_meta[mt] = "PASS" if pass_votes >= required else "FAIL"

        majority_score = sum(1 for v in majority_meta.values() if v == "PASS")

        results["traits"][trait_id] = {
            "per_judge_scores": per_judge_scores,
            "majority_score": majority_score,
            "meta_traits": majority_meta,
            "per_judge_meta_traits": per_judge_meta,
        }

    output_file = os.path.join(output_dir, "meta_trait_results_merged.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n=== Summary ===")
    for tid, r in results["traits"].items():
        score = r["majority_score"]
        flag = "PASS" if score == 5 else ("REVIEW" if score >= 3 else "FAIL")
        print(f"  {tid}: {score}/5 [{flag}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PReMISE meta-trait analysis")
    parser.add_argument("--traits", required=True, help="Path to traits JSON file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--judge-model",
        action="append",
        default=[],
        metavar="NAME=MODEL",
        help="Judge model; repeat for an ensemble.",
    )
    args = parser.parse_args()
    from premise.providers import parse_model_specs

    run_meta_trait_experiment(
        args.traits,
        args.output,
        judge_models=parse_model_specs(args.judge_model),
        region=args.region,
    )
