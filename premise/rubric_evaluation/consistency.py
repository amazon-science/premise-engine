# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Intra-Rater Consistency Experiment.

Measures score stability under non-deterministic settings (temperature > 0).
Primary metric: flip rate — percentage of samples where the winner changes across runs.

Default mode: Bedrock batch inference (server-side parallel).
Fallback: --no-batch for sequential Converse API calls.

Usage:
    # Batch mode (default)
    premise audit consistency --traits <traits.json> \
        --samples <samples.jsonl> --output results/consistency/

    # Sequential mode
    premise audit consistency --traits <traits.json> \
        --samples <samples.jsonl> --output results/consistency/ --no-batch
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

from premise.utils import (
    DEFAULT_BATCH_ROLE_ARN,
    DEFAULT_REGION,
    DEFAULT_S3_BUCKET,
    compute_file_hash,
    download_batch_results,
    extract_batch_model_response,
    get_batch_model_input,
    get_bedrock_client,
    get_bedrock_management_client,
    invoke_model,
    load_samples,
    load_traits,
    resolve_batch_settings,
    submit_batch_job,
    upload_to_s3,
    wait_for_batch_job,
)
from premise.utils.metrics import parse_score, score_to_winner
from premise.utils.prompts import (
    PAIRWISE_JUDGE_PROMPT,
    PAIRWISE_JUDGE_PROMPT_NO_REASONING,
)

BATCH_MAX_RECORDS = 10_000


# ============================================================
# Batch Inference Mode
# ============================================================


def prepare_consistency_batch(
    traits,
    samples,
    n_runs,
    temperature,
    judge_model_id,
    output_dir,
    prompt_template=None,
):
    """Build JSONL batch input files for all (trait, sample, run) combinations.

    recordId format: "{global_idx}|{trait_id}|{sample_idx}|{run_idx}"
    """
    if prompt_template is None:
        prompt_template = PAIRWISE_JUDGE_PROMPT
    records = []
    for trait in traits:
        for s_idx, sample in enumerate(samples):
            prompt = prompt_template.format(
                trait_id=trait["id"],
                trait_description=trait["description"],
                user_prompt=sample.get("prompt", sample.get("question", ""))[:2000],
                response_a=sample.get("response_a", sample.get("candidate", ""))[:3000],
                response_b=sample.get("response_b", sample.get("baseline", ""))[:3000],
            )
            for run_idx in range(n_runs):
                records.append(
                    {
                        "trait_id": trait["id"],
                        "sample_idx": s_idx,
                        "run_idx": run_idx,
                        "prompt": prompt,
                    }
                )

    print(
        f"  Total records: {len(records)} "
        f"({len(traits)} traits x {len(samples)} samples x {n_runs} runs)"
    )

    # Split into chunks
    chunks = [
        records[i : i + BATCH_MAX_RECORDS]
        for i in range(0, len(records), BATCH_MAX_RECORDS)
    ]
    print(f"  Splitting into {len(chunks)} batch file(s)")

    batch_paths = []
    for chunk_idx, chunk in enumerate(chunks):
        suffix = f"_{chunk_idx}" if len(chunks) > 1 else ""
        batch_path = os.path.join(output_dir, f"batch_input{suffix}.jsonl")

        with open(batch_path, "w") as f:
            for i, rec in enumerate(chunk):
                global_idx = chunk_idx * BATCH_MAX_RECORDS + i
                record_id = (
                    f"{global_idx}|{rec['trait_id']}"
                    f"|{rec['sample_idx']}|{rec['run_idx']}"
                )
                batch_record = {
                    "recordId": record_id,
                    "modelInput": get_batch_model_input(
                        judge_model_id,
                        rec["prompt"],
                        temperature=temperature,
                        max_tokens=500,
                    ),
                }
                f.write(json.dumps(batch_record) + "\n")

        print(f"  Chunk {chunk_idx}: {len(chunk)} records -> {batch_path}")
        batch_paths.append(batch_path)

    return batch_paths


def parse_consistency_batch_results(result_files, judge_model_id):
    """Parse batch output into {trait_id: {sample_idx: {run_idx: score}}}.

    Returns: nested dict of scores.
    """
    scores = defaultdict(lambda: defaultdict(dict))
    parsed = 0
    errors = 0

    for fpath in result_files:
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    record_id = rec.get("recordId", "")
                    parts = record_id.split("|", 3)
                    if len(parts) < 4:
                        errors += 1
                        continue

                    _, trait_id, sample_idx, run_idx = parts
                    sample_idx = int(sample_idx)
                    run_idx = int(run_idx)

                    model_output = rec.get("modelOutput", {})
                    text = extract_batch_model_response(model_output, judge_model_id)
                    score = parse_score(text) if text else None

                    scores[trait_id][sample_idx][run_idx] = score
                    if score is not None:
                        parsed += 1
                    else:
                        errors += 1

                except (json.JSONDecodeError, ValueError):
                    errors += 1

    print(f"  Parsed {parsed} scores, {errors} errors")
    return scores


def run_batch_consistency(
    traits,
    samples,
    n_runs,
    temperature,
    judge_model_id,
    output_dir,
    s3_bucket,
    role_arn,
    region,
    prompt_template=None,
):
    """Run consistency experiment via Bedrock batch inference."""

    # Check for already-downloaded results (resume)
    local_results_dir = os.path.join(output_dir, "batch_output")
    existing_results = [
        os.path.join(local_results_dir, f)
        for f in (
            os.listdir(local_results_dir) if os.path.isdir(local_results_dir) else []
        )
        if f.endswith(".jsonl.out")
    ]
    if existing_results:
        print(
            f"  Found {len(existing_results)} existing batch result files, "
            f"skipping upload/submit"
        )
        return parse_consistency_batch_results(existing_results, judge_model_id)

    # Prepare batch inputs
    batch_paths = prepare_consistency_batch(
        traits,
        samples,
        n_runs,
        temperature,
        judge_model_id,
        output_dir,
        prompt_template=prompt_template,
    )

    bedrock = get_bedrock_management_client(region)

    # Parallel upload + submit of all chunks (bounded by max_workers to stay within S3/API limits)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _upload_and_submit(chunk_idx, batch_path):
        file_hash = compute_file_hash(batch_path)[:12]
        suffix = f"-{chunk_idx}" if len(batch_paths) > 1 else ""
        job_name = f"premise-consistency-{file_hash}{suffix}"
        job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]
        s3_key = f"premise/consistency/{job_name}/input.jsonl"
        s3_output_prefix = f"s3://{s3_bucket}/premise/consistency/{job_name}/output/"
        print(
            f"  [chunk {chunk_idx}] uploading ({os.path.getsize(batch_path) / 1e6:.1f} MB)",
            flush=True,
        )
        s3_input_uri = upload_to_s3(batch_path, s3_bucket, s3_key, region)
        print(f"  [chunk {chunk_idx}] submitting {job_name}", flush=True)
        job_arn = submit_batch_job(
            bedrock,
            s3_input_uri,
            s3_output_prefix,
            job_name,
            judge_model_id,
            role_arn,
        )
        print(f"  [chunk {chunk_idx}] submitted → {job_arn}", flush=True)
        return (chunk_idx, job_name, job_arn, s3_output_prefix)

    max_parallel_uploads = min(8, len(batch_paths))
    print(
        f"\n  Parallel upload+submit with {max_parallel_uploads} workers for {len(batch_paths)} chunks"
    )
    job_arns = []
    with ThreadPoolExecutor(max_workers=max_parallel_uploads) as ex:
        futs = [ex.submit(_upload_and_submit, i, p) for i, p in enumerate(batch_paths)]
        for fut in as_completed(futs):
            job_arns.append(fut.result())
    # Sort by chunk_idx to keep waiting loop ordered
    job_arns.sort(key=lambda x: x[0])

    # Wait for all jobs and download results
    all_result_files = []
    for chunk_idx, job_name, job_arn, s3_output_prefix in job_arns:
        print(f"\n  Waiting for chunk {chunk_idx} ({job_name})...")
        result = wait_for_batch_job(
            bedrock, job_arn, poll_interval=60, timeout=14400, region=region
        )

        output_s3_uri = (
            result.get("outputDataConfig", {})
            .get("s3OutputDataConfig", {})
            .get("s3Uri", s3_output_prefix)
        )
        output_prefix = output_s3_uri.replace(f"s3://{s3_bucket}/", "")
        chunk_results_dir = os.path.join(local_results_dir, f"chunk_{chunk_idx}")
        print(f"  Downloading results from {output_s3_uri}")
        result_files = download_batch_results(
            s3_bucket, output_prefix, chunk_results_dir, region
        )
        print(f"  Downloaded {len(result_files)} result files for chunk {chunk_idx}")
        all_result_files.extend(result_files)

    print(
        f"\n  All {len(job_arns)} batch jobs complete. "
        f"Total result files: {len(all_result_files)}"
    )
    return parse_consistency_batch_results(all_result_files, judge_model_id)


# ============================================================
# Sequential Mode (fallback)
# ============================================================


def run_sequential_consistency(
    traits, samples, n_runs, temperature, judge_model_id, region, prompt_template=None
):
    """Run consistency via sequential Converse API calls.

    Returns: {trait_id: {sample_idx: {run_idx: score}}}
    """
    if prompt_template is None:
        prompt_template = PAIRWISE_JUDGE_PROMPT
    client = get_bedrock_client(region)
    total_calls = len(traits) * len(samples) * n_runs
    print(f"  Total API calls: {total_calls}")

    scores = defaultdict(lambda: defaultdict(dict))
    call_count = 0

    for trait in traits:
        trait_id = trait["id"]
        trait_desc = trait["description"]
        print(f"\n=== {trait_id} ===")

        for run_idx in range(n_runs):
            for s_idx, sample in enumerate(samples):
                call_count += 1
                if call_count % 100 == 0:
                    print(f"  [{call_count}/{total_calls}]")

                prompt = prompt_template.format(
                    trait_id=trait_id,
                    trait_description=trait_desc,
                    user_prompt=sample.get("prompt", sample.get("question", ""))[:2000],
                    response_a=sample.get("response_a", sample.get("candidate", ""))[
                        :3000
                    ],
                    response_b=sample.get("response_b", sample.get("baseline", ""))[
                        :3000
                    ],
                )
                response = invoke_model(
                    client, judge_model_id, prompt, temperature=temperature
                )
                scores[trait_id][s_idx][run_idx] = parse_score(response)

            print(f"  Run {run_idx + 1}/{n_runs} complete")

    return scores


# ============================================================
# Metrics computation
# ============================================================


def compute_consistency_metrics(scores, traits, n_runs):
    """Compute flip rate and raw sigma from score matrix.

    Args:
        scores: {trait_id: {sample_idx: {run_idx: score}}}
    Returns:
        dict of per-trait results.
    """
    results = {}

    for trait in traits:
        trait_id = trait["id"]
        trait_scores = scores.get(trait_id, {})

        n_samples_valid = 0
        n_flips = 0
        score_sigmas = []

        for s_idx in sorted(trait_scores.keys()):
            run_scores = trait_scores[s_idx]
            valid_scores = [
                run_scores[r]
                for r in range(n_runs)
                if r in run_scores and run_scores[r] is not None
            ]
            if len(valid_scores) < 2:
                continue

            n_samples_valid += 1
            score_sigmas.append(np.std(valid_scores))
            winners = {score_to_winner(s) for s in valid_scores} - {"unknown"}
            if len(winners) > 1:
                n_flips += 1

        flip_rate = (n_flips / n_samples_valid * 100) if n_samples_valid > 0 else 0
        raw_sigma = float(np.mean(score_sigmas)) if score_sigmas else 0

        results[trait_id] = {
            "raw_sigma": round(raw_sigma, 3),
            "win_flip_rate": round(flip_rate, 1),
            "n_samples": n_samples_valid,
        }

    return results


# ============================================================
# Main
# ============================================================


def run_consistency_experiment(
    traits_file: str,
    samples_file: str,
    output_dir: str,
    n_runs: int = 5,
    temperature: float = 0.7,
    max_samples: int = 200,
    judge_model: str | None = None,
    no_batch: bool = False,
    no_reasoning: bool = False,
    region: str = DEFAULT_REGION,
    s3_bucket: str = DEFAULT_S3_BUCKET,
    role_arn: str = DEFAULT_BATCH_ROLE_ARN,
):
    if judge_model is None:
        from premise.providers import model_for_role

        judge_model = model_for_role("judge")
    if not no_batch:
        s3_bucket, role_arn = resolve_batch_settings(s3_bucket, role_arn)

    prompt_template = (
        PAIRWISE_JUDGE_PROMPT_NO_REASONING if no_reasoning else PAIRWISE_JUDGE_PROMPT
    )

    os.makedirs(output_dir, exist_ok=True)
    traits = load_traits(traits_file)
    samples = load_samples(samples_file, max_samples)

    print(f"Loaded {len(traits)} traits, {len(samples)} samples")
    print(f"Running {n_runs} iterations at temperature={temperature}")
    print(f"Judge model: {judge_model}")
    print(f"Reasoning: {'disabled' if no_reasoning else 'enabled'}")
    print(f"Batch inference: {'disabled' if no_batch else 'enabled'}")
    total_calls = len(traits) * len(samples) * n_runs
    print(f"Total API calls: {total_calls}")

    # Run scoring
    if no_batch:
        scores = run_sequential_consistency(
            traits,
            samples,
            n_runs,
            temperature,
            judge_model,
            region,
            prompt_template=prompt_template,
        )
    else:
        scores = run_batch_consistency(
            traits,
            samples,
            n_runs,
            temperature,
            judge_model,
            output_dir,
            s3_bucket,
            role_arn,
            region,
            prompt_template=prompt_template,
        )

    # Compute metrics
    results = compute_consistency_metrics(scores, traits, n_runs)

    output_file = os.path.join(output_dir, "consistency_scores.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n=== Summary ===")
    for tid, r in sorted(results.items(), key=lambda x: x[1]["win_flip_rate"]):
        flip = r["win_flip_rate"]
        flag = "PASS" if flip < 5 else ("REVIEW" if flip < 15 else "FAIL")
        print(f"  {tid}: flip={flip:.1f}% [{flag}]")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PReMISE consistency experiment")
    parser.add_argument("--traits", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--judge-model",
        help="Judge model ID (default: config role 'judge')",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch inference, use sequential Converse API",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Use score-only prompt without chain-of-thought reasoning",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--role-arn", default=DEFAULT_BATCH_ROLE_ARN)
    args = parser.parse_args()
    run_consistency_experiment(
        args.traits,
        args.samples,
        args.output,
        n_runs=args.n_runs,
        temperature=args.temperature,
        max_samples=args.max_samples,
        judge_model=args.judge_model,
        no_batch=args.no_batch,
        no_reasoning=args.no_reasoning,
        region=args.region,
        s3_bucket=args.s3_bucket,
        role_arn=args.role_arn,
    )
