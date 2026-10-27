# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Rubric Applicability Evaluation.

For a given rubric, measures what fraction of prompts from a dataset each
trait meaningfully applies to. Designed to run both in-domain (source dataset)
and out-of-domain (foreign dataset) to measure the generality gap.

Uses Bedrock batch inference with DeepSeek-V3 as the judge.

For each (trait, prompt) pair, the judge answers: "Is this trait relevant
for evaluating a response to this prompt?" → yes / no.

Metrics:
    in_domain_applicability  = mean fraction of source-dataset prompts each trait applies to
    out_domain_applicability = mean fraction of foreign-dataset prompts each trait applies to
    generality_gap           = in_domain - out_domain (positive = appropriate specificity)

Usage:
    python -m premise.rubric_evaluation.applicability_eval \
        --rubric <traits.json> \
        --in-domain-battles <source_battles.jsonl> \
        --out-domain-battles <foreign_battles.jsonl> \
        --max-prompts 200 \
        --output results/applicability/
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from premise.config import get_config
from premise.utils import (
    DEFAULT_BATCH_ROLE_ARN,
    DEFAULT_REGION,
    DEFAULT_S3_BUCKET,
    compute_file_hash,
    download_batch_results,
    extract_batch_model_response,
    get_batch_model_input,
    get_bedrock_management_client,
    resolve_batch_settings,
    submit_batch_job,
    upload_to_s3,
    wait_for_batch_job,
)
from premise.utils.samples import _extract_conversation_text


def _judge_model() -> str:
    return get_config().model("judge")


APPLICABILITY_PROMPT = """You are evaluating whether an evaluation trait is RELEVANT for scoring a response to a given user prompt.

TRAIT:
- Name: {trait_id}
- Description: {trait_description}

USER PROMPT:
{user_prompt}

QUESTION: Could a judge meaningfully score a response to this user prompt on the above trait? A trait is relevant if a response could demonstrably succeed or fail on it given this prompt type.

Answer with ONLY one word: "yes" or "no"."""


def load_rubric(rubric_path: str) -> list[dict]:
    """Load rubric traits from JSON."""
    data = json.loads(Path(rubric_path).read_text())
    rubrics = data.get("rubrics") or data.get("rules") or []
    traits = []
    for i, r in enumerate(rubrics):
        if isinstance(r, str):
            traits.append({"id": f"rule_{i}", "description": r})
        else:
            traits.append(
                {
                    "id": r.get("id") or r.get("name") or f"trait_{i}",
                    "description": r.get("description") or r.get("text") or "",
                }
            )
    return traits


def load_prompts(
    battles_path: str, max_prompts: int = 200, seed: int = 42
) -> list[str]:
    """Load user prompts from a battles JSONL file, sampled deterministically."""
    prompts = []
    with open(battles_path) as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            conv_a = b.get("conversation_a", [])
            conv_b = b.get("conversation_b", [])
            p, _ = _extract_conversation_text(conv_a)
            if not p:
                p, _ = _extract_conversation_text(conv_b)
            if p:
                prompts.append(p)

    # Deterministic subsample
    if len(prompts) > max_prompts:
        import random

        rng = random.Random(seed)
        prompts = rng.sample(prompts, max_prompts)
    return prompts


def prepare_batch_input(
    traits: list[dict],
    prompts: list[str],
    output_dir: str,
    filename: str = "applicability_batch_input.jsonl",
) -> str:
    """Build batch input: one record per (trait, prompt) pair."""
    batch_path = os.path.join(output_dir, filename)
    written = 0
    with open(batch_path, "w") as f:
        for t_idx, trait in enumerate(traits):
            for p_idx, prompt in enumerate(prompts):
                record_id = f"{t_idx:04d}__{p_idx:04d}"
                eval_prompt = APPLICABILITY_PROMPT.format(
                    trait_id=trait["id"],
                    trait_description=trait["description"],
                    user_prompt=prompt[:1000],
                )
                record = {
                    "recordId": record_id,
                    "modelInput": get_batch_model_input(
                        _judge_model(),
                        eval_prompt,
                        temperature=0.0,
                        max_tokens=10,
                    ),
                }
                f.write(json.dumps(record) + "\n")
                written += 1
    print(
        f"  Wrote {written} records ({len(traits)} traits × {len(prompts)} prompts) "
        f"to {batch_path}"
    )
    return batch_path


def parse_batch_results(
    result_files: list[str],
    n_traits: int,
    n_prompts: int,
) -> dict:
    """Parse batch output into per-trait applicability counts.

    Returns {trait_idx: n_applicable} where n_applicable = number of prompts
    the judge said "yes" for that trait.
    """
    applicable = {}  # (trait_idx, prompt_idx) -> bool
    for rf in result_files:
        with open(rf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_id = rec.get("recordId", "")
                parts = record_id.split("__")
                if len(parts) != 2:
                    continue
                t_idx = int(parts[0])
                p_idx = int(parts[1])
                text = (
                    extract_batch_model_response(
                        rec.get("modelOutput", {}),
                        _judge_model(),
                    )
                    or ""
                )
                text = text.strip().lower()
                is_yes = text.startswith("yes")
                applicable[(t_idx, p_idx)] = is_yes

    # Aggregate per trait
    per_trait = {}
    for t_idx in range(n_traits):
        count = sum(
            1 for p_idx in range(n_prompts) if applicable.get((t_idx, p_idx), False)
        )
        per_trait[t_idx] = count
    return per_trait


def run_applicability_eval(
    rubric_path: str,
    in_domain_battles: str,
    out_domain_battles: str | None,
    output_dir: str,
    max_prompts: int = 200,
    region: str = DEFAULT_REGION,
    s3_bucket: str = DEFAULT_S3_BUCKET,
    role_arn: str = DEFAULT_BATCH_ROLE_ARN,
    no_batch: bool = False,
) -> dict:
    """Run applicability evaluation.

    Submits batch jobs for in-domain (and optionally out-of-domain) scoring.
    """
    if not no_batch:
        s3_bucket, role_arn = resolve_batch_settings(s3_bucket, role_arn)
    os.makedirs(output_dir, exist_ok=True)
    traits = load_rubric(rubric_path)
    print(f"Loaded {len(traits)} traits from {rubric_path}")

    in_prompts = load_prompts(in_domain_battles, max_prompts)
    print(f"Loaded {len(in_prompts)} in-domain prompts from {in_domain_battles}")

    out_prompts = None
    if out_domain_battles:
        out_prompts = load_prompts(out_domain_battles, max_prompts)
        print(
            f"Loaded {len(out_prompts)} out-of-domain prompts from {out_domain_battles}"
        )

    results = {"traits": [t["id"] for t in traits], "n_traits": len(traits)}

    # --- In-domain ---
    print("\n=== In-domain applicability ===")
    in_batch = prepare_batch_input(
        traits, in_prompts, output_dir, "in_domain_batch.jsonl"
    )

    if no_batch:
        # Sync mode for small runs
        from premise.utils import get_bedrock_client, invoke_model

        client = get_bedrock_client(region)
        in_counts = {}
        for t_idx, trait in enumerate(traits):
            count = 0
            for prompt in in_prompts:
                p = APPLICABILITY_PROMPT.format(
                    trait_id=trait["id"],
                    trait_description=trait["description"],
                    user_prompt=prompt[:1000],
                )
                resp = invoke_model(
                    client, _judge_model(), p, temperature=0.0, max_tokens=10
                )
                if resp and resp.strip().lower().startswith("yes"):
                    count += 1
            in_counts[t_idx] = count
            if (t_idx + 1) % 5 == 0:
                print(f"  {t_idx + 1}/{len(traits)} traits scored")
    else:
        file_hash = compute_file_hash(in_batch)[:12]
        job_name = f"applicability-indomain-{file_hash}"
        job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]
        s3_key = f"premise/applicability/{job_name}/input.jsonl"
        s3_output = f"s3://{s3_bucket}/premise/applicability/{job_name}/output/"

        print(f"  Uploading to s3://{s3_bucket}/{s3_key}")
        upload_to_s3(in_batch, s3_bucket, s3_key, region)

        bedrock = get_bedrock_management_client(region)
        print(f"  Submitting batch job: {job_name}")
        job_arn = submit_batch_job(
            bedrock,
            f"s3://{s3_bucket}/{s3_key}",
            s3_output,
            job_name,
            _judge_model(),
            role_arn,
        )
        print("  Waiting for completion...")
        result = wait_for_batch_job(bedrock, job_arn, poll_interval=60, timeout=7200)

        output_s3 = (
            result.get("outputDataConfig", {})
            .get("s3OutputDataConfig", {})
            .get("s3Uri", s3_output)
        )
        output_prefix = output_s3.replace(f"s3://{s3_bucket}/", "")
        local_results = os.path.join(output_dir, "in_domain_output")
        result_files = download_batch_results(
            s3_bucket, output_prefix, local_results, region
        )
        in_counts = parse_batch_results(result_files, len(traits), len(in_prompts))

    # Compute in-domain metrics
    in_rates = {
        traits[i]["id"]: in_counts[i] / len(in_prompts) for i in range(len(traits))
    }
    mean_in = sum(in_rates.values()) / len(in_rates)
    results["in_domain"] = {
        "n_prompts": len(in_prompts),
        "per_trait": in_rates,
        "mean_applicability": round(mean_in, 4),
    }
    print(f"  Mean in-domain applicability: {mean_in:.1%}")

    # --- Out-of-domain ---
    if out_prompts:
        print("\n=== Out-of-domain applicability ===")
        out_batch = prepare_batch_input(
            traits, out_prompts, output_dir, "out_domain_batch.jsonl"
        )

        if no_batch:
            out_counts = {}
            for t_idx, trait in enumerate(traits):
                count = 0
                for prompt in out_prompts:
                    p = APPLICABILITY_PROMPT.format(
                        trait_id=trait["id"],
                        trait_description=trait["description"],
                        user_prompt=prompt[:1000],
                    )
                    resp = invoke_model(
                        client, _judge_model(), p, temperature=0.0, max_tokens=10
                    )
                    if resp and resp.strip().lower().startswith("yes"):
                        count += 1
                out_counts[t_idx] = count
                if (t_idx + 1) % 5 == 0:
                    print(f"  {t_idx + 1}/{len(traits)} traits scored")
        else:
            file_hash = compute_file_hash(out_batch)[:12]
            job_name = f"applicability-outdomain-{file_hash}"
            job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]
            s3_key = f"premise/applicability/{job_name}/input.jsonl"
            s3_output = f"s3://{s3_bucket}/premise/applicability/{job_name}/output/"

            print(f"  Uploading to s3://{s3_bucket}/{s3_key}")
            upload_to_s3(out_batch, s3_bucket, s3_key, region)

            bedrock = get_bedrock_management_client(region)
            print(f"  Submitting batch job: {job_name}")
            job_arn = submit_batch_job(
                bedrock,
                f"s3://{s3_bucket}/{s3_key}",
                s3_output,
                job_name,
                _judge_model(),
                role_arn,
            )
            print("  Waiting for completion...")
            result = wait_for_batch_job(
                bedrock, job_arn, poll_interval=60, timeout=7200
            )

            output_s3 = (
                result.get("outputDataConfig", {})
                .get("s3OutputDataConfig", {})
                .get("s3Uri", s3_output)
            )
            output_prefix = output_s3.replace(f"s3://{s3_bucket}/", "")
            local_results = os.path.join(output_dir, "out_domain_output")
            result_files = download_batch_results(
                s3_bucket,
                output_prefix,
                local_results,
                region,
            )
            out_counts = parse_batch_results(
                result_files, len(traits), len(out_prompts)
            )

        out_rates = {
            traits[i]["id"]: out_counts[i] / len(out_prompts)
            for i in range(len(traits))
        }
        mean_out = sum(out_rates.values()) / len(out_rates)
        gap = mean_in - mean_out
        results["out_domain"] = {
            "n_prompts": len(out_prompts),
            "per_trait": out_rates,
            "mean_applicability": round(mean_out, 4),
        }
        results["generality_gap"] = round(gap, 4)
        print(f"  Mean out-of-domain applicability: {mean_out:.1%}")
        print(f"  Generality gap (in - out): {gap:+.1%}")

    # Save results
    output_path = os.path.join(output_dir, "applicability_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate rubric trait applicability across datasets"
    )
    parser.add_argument("--rubric", required=True, help="Path to rubric JSON")
    parser.add_argument(
        "--in-domain-battles",
        required=True,
        help="Battles JSONL from the source dataset",
    )
    parser.add_argument(
        "--out-domain-battles",
        default=None,
        help="Battles JSONL from a foreign dataset (optional)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=200,
        help="Max prompts to sample per dataset (default 200)",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--role-arn", default=DEFAULT_BATCH_ROLE_ARN)
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Use sync calls instead of batch (for small tests)",
    )

    args = parser.parse_args()

    run_applicability_eval(
        rubric_path=args.rubric,
        in_domain_battles=args.in_domain_battles,
        out_domain_battles=args.out_domain_battles,
        output_dir=args.output,
        max_prompts=args.max_prompts,
        region=args.region,
        s3_bucket=args.s3_bucket,
        role_arn=args.role_arn,
        no_batch=args.no_batch,
    )
