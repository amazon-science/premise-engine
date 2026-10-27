# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Inter-Rater Agreement (IRA) Experiment.

Measures whether multiple LLM judges produce consistent pairwise ratings.
Computes ordinal Krippendorff's alpha on the original 0--10 scores per rubric,
matching the paper protocol. Raw output also includes the derived
win/loss/tie outcome for compatibility with older analysis scripts.

Default mode: Bedrock batch inference (one batch job per judge model).
Fallback: --no-batch for sequential Converse API calls.

Usage:
    # Batch mode (default)
    premise audit ira --traits <traits.json> \
        --samples <samples.jsonl> --output results/ira/

    # Sequential mode
    premise audit ira --traits <traits.json> \
        --samples <samples.jsonl> --output results/ira/ --no-batch
"""

import argparse
import json
import os
import re
from collections import defaultdict

from premise.config import get_config
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
from premise.utils.metrics import (
    compute_krippendorff_alpha,
    parse_score,
    score_to_outcome,
)
from premise.utils.prompts import PAIRWISE_JUDGE_PROMPT

BATCH_MAX_RECORDS = 10_000


# ============================================================
# Batch Inference Mode
# ============================================================


def prepare_ira_batch(traits, samples, judge_model_id, judge_name, output_dir):
    """Build JSONL batch input files for one judge model.

    recordId format: "{global_idx}|{judge_name}|{trait_id}|{sample_idx}"
    """
    records = []
    for trait in traits:
        for s_idx, sample in enumerate(samples):
            prompt = PAIRWISE_JUDGE_PROMPT.format(
                trait_id=trait["id"],
                trait_description=trait["description"],
                user_prompt=sample.get("prompt", sample.get("question", "")),
                response_a=sample.get("response_a", sample.get("candidate", "")),
                response_b=sample.get("response_b", sample.get("baseline", "")),
            )
            records.append(
                {
                    "trait_id": trait["id"],
                    "sample_idx": s_idx,
                    "prompt": prompt,
                }
            )

    # Split into chunks
    chunks = [
        records[i : i + BATCH_MAX_RECORDS]
        for i in range(0, len(records), BATCH_MAX_RECORDS)
    ]

    safe_judge = re.sub(r"[^a-zA-Z0-9]", "-", judge_name.lower())
    batch_paths = []
    for chunk_idx, chunk in enumerate(chunks):
        suffix = f"_{chunk_idx}" if len(chunks) > 1 else ""
        batch_path = os.path.join(output_dir, f"batch_input_{safe_judge}{suffix}.jsonl")

        with open(batch_path, "w") as f:
            for i, rec in enumerate(chunk):
                global_idx = chunk_idx * BATCH_MAX_RECORDS + i
                record_id = (
                    f"{global_idx}|{judge_name}|{rec['trait_id']}|{rec['sample_idx']}"
                )
                batch_record = {
                    "recordId": record_id,
                    "modelInput": get_batch_model_input(
                        judge_model_id,
                        rec["prompt"],
                        temperature=0.0,
                        max_tokens=500,
                    ),
                }
                f.write(json.dumps(batch_record) + "\n")

        batch_paths.append(batch_path)

    print(f"  {judge_name}: {len(records)} records -> {len(chunks)} chunk(s)")
    return batch_paths


def parse_ira_batch_results(result_files, judge_model_id):
    """Parse batch output into {judge_name: {trait_id: {sample_idx: score}}}.

    Returns: nested dict of 0--10 scores.
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

                    _, judge_name, trait_id, sample_idx = parts
                    sample_idx = int(sample_idx)

                    model_output = rec.get("modelOutput", {})
                    text = extract_batch_model_response(model_output, judge_model_id)
                    score = parse_score(text) if text else None
                    scores[judge_name][trait_id][sample_idx] = score
                    if score is not None:
                        parsed += 1
                    else:
                        errors += 1

                except (json.JSONDecodeError, ValueError):
                    errors += 1

    print(f"  Parsed {parsed} scores, {errors} errors")
    return scores


def run_batch_ira(
    traits, samples, judge_models, output_dir, s3_bucket, role_arn, region
):
    """Run IRA experiment via Bedrock batch inference.

    Submits one batch job per judge model, waits for all, then downloads.
    """
    # Check for already-downloaded results (resume)
    local_results_dir = os.path.join(output_dir, "batch_output")
    existing_results = []
    if os.path.isdir(local_results_dir):
        for root, dirs, files in os.walk(local_results_dir):
            for f in files:
                if f.endswith(".jsonl.out"):
                    existing_results.append(os.path.join(root, f))

    if existing_results:
        print(
            f"  Found {len(existing_results)} existing batch result files, "
            f"skipping upload/submit"
        )
        # Need to parse with each judge's model_id — use first judge as default
        # since extract_batch_model_response handles common formats
        return parse_ira_batch_results(existing_results, judge_models[0][1])

    bedrock = get_bedrock_management_client(region)
    all_job_info = []  # (judge_name, judge_model_id, chunk_idx, job_name, job_arn, s3_output)

    # Submit batch jobs for each judge
    for judge_name, judge_model_id in judge_models:
        print(f"\n  Preparing batch for {judge_name} ({judge_model_id})")
        batch_paths = prepare_ira_batch(
            traits, samples, judge_model_id, judge_name, output_dir
        )

        for chunk_idx, batch_path in enumerate(batch_paths):
            file_hash = compute_file_hash(batch_path)[:12]
            safe_judge = re.sub(r"[^a-zA-Z0-9]", "-", judge_name.lower())
            suffix = f"-{chunk_idx}" if len(batch_paths) > 1 else ""
            job_name = f"premise-ira-{safe_judge}-{file_hash}{suffix}"
            job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]

            s3_key = f"premise/ira/{job_name}/input.jsonl"
            s3_output_prefix = f"s3://{s3_bucket}/premise/ira/{job_name}/output/"

            print(f"  Uploading to s3://{s3_bucket}/{s3_key}")
            s3_input_uri = upload_to_s3(batch_path, s3_bucket, s3_key, region)

            print(f"  Submitting batch job: {job_name}")
            job_arn = submit_batch_job(
                bedrock,
                s3_input_uri,
                s3_output_prefix,
                job_name,
                judge_model_id,
                role_arn,
            )
            print(f"  Job ARN: {job_arn}")
            all_job_info.append(
                (
                    judge_name,
                    judge_model_id,
                    chunk_idx,
                    job_name,
                    job_arn,
                    s3_output_prefix,
                )
            )

    # Wait for all jobs and download results
    all_result_files = []
    for (
        judge_name,
        judge_model_id,
        chunk_idx,
        job_name,
        job_arn,
        s3_output_prefix,
    ) in all_job_info:
        print(f"\n  Waiting for {judge_name} chunk {chunk_idx} ({job_name})...")
        result = wait_for_batch_job(
            bedrock, job_arn, poll_interval=60, timeout=14400, region=region
        )

        output_s3_uri = (
            result.get("outputDataConfig", {})
            .get("s3OutputDataConfig", {})
            .get("s3Uri", s3_output_prefix)
        )
        output_prefix = output_s3_uri.replace(f"s3://{s3_bucket}/", "")
        safe_judge = re.sub(r"[^a-zA-Z0-9]", "-", judge_name.lower())
        chunk_results_dir = os.path.join(
            local_results_dir, f"{safe_judge}_chunk_{chunk_idx}"
        )
        print(f"  Downloading results from {output_s3_uri}")
        result_files = download_batch_results(
            s3_bucket, output_prefix, chunk_results_dir, region
        )
        print(f"  Downloaded {len(result_files)} result files")
        all_result_files.extend(result_files)

    print(
        f"\n  All {len(all_job_info)} batch jobs complete. "
        f"Total result files: {len(all_result_files)}"
    )

    # Parse with first judge model id (format is the same across all)
    return parse_ira_batch_results(all_result_files, judge_models[0][1])


# ============================================================
# Sequential Mode (fallback)
# ============================================================


def run_sequential_ira(traits, samples, judge_models, output_dir, region):
    """Run IRA via sequential Converse API calls.

    Returns: {judge_name: {trait_id: {sample_idx: score}}}
    """
    client = get_bedrock_client(region)
    total_calls = len(traits) * len(samples) * len(judge_models)
    print(f"  Total API calls: {total_calls}")

    scores = defaultdict(lambda: defaultdict(dict))
    call_count = 0

    for trait in traits:
        trait_id = trait["id"]
        trait_desc = trait["description"]
        print(f"\n=== {trait_id} ===")

        for judge_name, judge_model_id in judge_models:
            for s_idx, sample in enumerate(samples):
                call_count += 1
                if call_count % 50 == 0:
                    print(f"  [{call_count}/{total_calls}]")

                prompt = PAIRWISE_JUDGE_PROMPT.format(
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
                response = invoke_model(client, judge_model_id, prompt, temperature=0.0)
                score = parse_score(response)
                scores[judge_name][trait_id][s_idx] = score

            print(f"  {judge_name} done")

    return scores


# ============================================================
# Metrics computation
# ============================================================


def compute_ira_metrics(scores, traits, samples, judge_models):
    """Compute Krippendorff's alpha and per-judge win rates from score matrix.

    Args:
        scores: {judge_name: {trait_id: {sample_idx: score}}}
    Returns:
        dict of per-trait results.
    """
    results = {}
    n_samples = len(samples)

    for trait in traits:
        trait_id = trait["id"]

        all_judgments = []
        per_judge_winrates = {}

        for judge_name, _ in judge_models:
            judge_scores = []
            wins = 0
            for s_idx in range(n_samples):
                score = scores.get(judge_name, {}).get(trait_id, {}).get(s_idx)
                judge_scores.append(score)
                if score is not None and score > 5:
                    wins += 1

            all_judgments.append(judge_scores)
            winrate = (wins / n_samples) * 100 if n_samples > 0 else 0
            per_judge_winrates[judge_name] = round(winrate, 1)

        alpha = compute_krippendorff_alpha(all_judgments, level="ordinal")

        results[trait_id] = {
            "krippendorff_alpha": round(alpha, 3),
            "per_judge_winrates": per_judge_winrates,
            "n_samples": n_samples,
        }

    return results


# ============================================================
# Main
# ============================================================


def run_ira_experiment(
    traits_file: str,
    samples_file: str,
    output_dir: str,
    max_samples: int = 200,
    judge_models: list | None = None,
    no_batch: bool = False,
    region: str = DEFAULT_REGION,
    s3_bucket: str = DEFAULT_S3_BUCKET,
    role_arn: str = DEFAULT_BATCH_ROLE_ARN,
):
    if judge_models is None:
        from premise.providers import model_for_role, provider_name

        if provider_name() == "bedrock":
            judge_models = [
                (str(member["name"]), str(member["model"]))
                for member in get_config().ensemble("ira")
            ]
        else:
            selected = model_for_role("judge")
            judge_models = [(selected, selected)]
    if not no_batch:
        s3_bucket, role_arn = resolve_batch_settings(s3_bucket, role_arn)

    os.makedirs(output_dir, exist_ok=True)
    traits = load_traits(traits_file)
    samples = load_samples(samples_file, max_samples)

    print(
        f"Loaded {len(traits)} traits, {len(samples)} samples, {len(judge_models)} judges"
    )
    print(f"Batch inference: {'disabled' if no_batch else 'enabled'}")
    total_calls = len(traits) * len(samples) * len(judge_models)
    print(f"Total API calls: {total_calls}")

    # Run scoring
    if no_batch:
        scores = run_sequential_ira(traits, samples, judge_models, output_dir, region)
    else:
        scores = run_batch_ira(
            traits, samples, judge_models, output_dir, s3_bucket, role_arn, region
        )

    # Compute metrics
    results = compute_ira_metrics(scores, traits, samples, judge_models)

    output_file = os.path.join(output_dir, "ira_scores.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Save raw per-sample judgments
    for trait in traits:
        trait_id = trait["id"]
        raw_file = os.path.join(output_dir, f"raw_{trait_id}.jsonl")
        with open(raw_file, "w") as rf:
            for s_idx, sample in enumerate(samples):
                sample_id = sample.get("id", sample.get("question_id", str(s_idx)))
                for judge_name, _ in judge_models:
                    score = scores.get(judge_name, {}).get(trait_id, {}).get(s_idx)
                    rf.write(
                        json.dumps(
                            {
                                "sample_id": str(sample_id),
                                "judge": judge_name,
                                "rubric": trait_id,
                                "score": score,
                                "outcome": score_to_outcome(score),
                            }
                        )
                        + "\n"
                    )

    print("\n=== Summary ===")
    for tid, r in sorted(results.items(), key=lambda x: -x[1]["krippendorff_alpha"]):
        alpha = r["krippendorff_alpha"]
        flag = "PASS" if alpha >= 0.6 else ("REVIEW" if alpha >= 0.3 else "FAIL")
        print(f"  {tid}: alpha={alpha:.3f} [{flag}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PReMISE IRA experiment")
    parser.add_argument("--traits", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch inference, use sequential Converse API",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--role-arn", default=DEFAULT_BATCH_ROLE_ARN)
    parser.add_argument(
        "--judge-model",
        action="append",
        default=[],
        metavar="NAME=MODEL",
        help="Judge model; repeat for an ensemble.",
    )
    args = parser.parse_args()
    from premise.providers import parse_model_specs

    run_ira_experiment(
        args.traits,
        args.samples,
        args.output,
        max_samples=args.max_samples,
        judge_models=parse_model_specs(args.judge_model),
        no_batch=args.no_batch,
        region=args.region,
        s3_bucket=args.s3_bucket,
        role_arn=args.role_arn,
    )
