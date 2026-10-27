# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Bradley-Terry Preference-Fit Scoring.

Scores preference battles on every rubric through Bedrock Batch Inference or a
synchronous provider. Outputs per-sample per-trait scores for
`premise audit bt-regress` and `premise audit pref-fit-analysis`.

Uses the configured judge model (DeepSeek-V3 in the paper profile) and the
pairwise judge prompt (0--10 scale, 5=equal).

Usage:
    # Score 2000 battles on a discovered rubric
    premise audit bt-score \
        --battles <battles.jsonl> \
        --traits <traits_top_20.json> \
        --max-samples 2000 \
        --output results/bt_scoring/discovered/

    # Score with a specific judge and run ID
    premise audit bt-score \
        --battles <battles.jsonl> \
        --traits <traits_top_20.json> \
        --max-samples 2000 \
        --judge-model <model-id> \
        --run-id my-run \
        --output results/bt_scoring/baseline/
"""

import argparse
import json
import os
import re

from premise.providers import model_for_role
from premise.utils import (
    get_bedrock_client,
    invoke_model,
    load_samples,
    load_traits,
)
from premise.utils.bedrock import (
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
from premise.utils.metrics import parse_score
from premise.utils.prompts import PAIRWISE_JUDGE_PROMPT

BATCH_MAX_RECORDS = 10_000


def prepare_batch(traits, samples, judge_model_id, output_dir, run_id):
    """Build JSONL batch input for all (sample, trait) pairs.

    recordId format: "{global_idx}|{sample_idx}|{trait_id}|{sample_id}"
    Returns list of local batch file paths.
    """
    records = []
    for s_idx, sample in enumerate(samples):
        sample_id = str(sample.get("id", sample.get("question_id", s_idx)))
        prompt_text = sample.get("prompt", sample.get("question", ""))[:2000]
        resp_a = sample.get("response_a", sample.get("candidate", ""))[:3000]
        resp_b = sample.get("response_b", sample.get("baseline", ""))[:3000]

        for trait in traits:
            judge_prompt = PAIRWISE_JUDGE_PROMPT.format(
                trait_id=trait["id"],
                trait_description=trait["description"],
                user_prompt=prompt_text,
                response_a=resp_a,
                response_b=resp_b,
            )
            records.append(
                {
                    "sample_idx": s_idx,
                    "sample_id": sample_id,
                    "trait_id": trait["id"],
                    "prompt": judge_prompt,
                }
            )

    # Split into chunks
    chunks = [
        records[i : i + BATCH_MAX_RECORDS]
        for i in range(0, len(records), BATCH_MAX_RECORDS)
    ]

    batch_paths = []
    for chunk_idx, chunk in enumerate(chunks):
        suffix = f"_{chunk_idx}" if len(chunks) > 1 else ""
        batch_path = os.path.join(output_dir, f"batch_input{suffix}.jsonl")

        with open(batch_path, "w") as f:
            for i, rec in enumerate(chunk):
                global_idx = chunk_idx * BATCH_MAX_RECORDS + i
                record_id = (
                    f"{global_idx}|{rec['sample_idx']}"
                    f"|{rec['trait_id']}|{rec['sample_id']}"
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

    print(f"  {len(records)} records -> {len(chunks)} chunk(s)")
    return batch_paths


def submit_and_wait(
    batch_paths, judge_model_id, output_dir, s3_bucket, role_arn, region, run_id
):
    """Upload, submit, wait, and download batch results.

    Returns list of local result file paths.
    """
    bedrock = get_bedrock_management_client(region)
    local_results_dir = os.path.join(output_dir, "batch_output")

    # Resume: check for existing results
    existing_results = []
    if os.path.isdir(local_results_dir):
        for root, _, files in os.walk(local_results_dir):
            for f in files:
                if f.endswith(".jsonl.out"):
                    existing_results.append(os.path.join(root, f))

    if existing_results:
        print(f"  Found {len(existing_results)} existing result files, skipping submit")
        return existing_results

    all_job_info = []
    for chunk_idx, batch_path in enumerate(batch_paths):
        file_hash = compute_file_hash(batch_path)[:12]
        suffix = f"-{chunk_idx}" if len(batch_paths) > 1 else ""
        safe_run = re.sub(r"[^a-zA-Z0-9]", "-", run_id)[:20]
        job_name = f"premise-bt-{safe_run}-{file_hash}{suffix}"
        job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]

        s3_key = f"premise/bt-scoring/{job_name}/input.jsonl"
        s3_output_prefix = f"s3://{s3_bucket}/premise/bt-scoring/{job_name}/output/"

        print(f"  Uploading chunk {chunk_idx} to s3://{s3_bucket}/{s3_key}")
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
        all_job_info.append((chunk_idx, job_name, job_arn, s3_output_prefix))

    # Wait and download
    all_result_files = []
    for chunk_idx, job_name, job_arn, s3_output_prefix in all_job_info:
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
        chunk_dir = os.path.join(local_results_dir, f"chunk_{chunk_idx}")
        print(f"  Downloading results from {output_s3_uri}")
        result_files = download_batch_results(
            s3_bucket, output_prefix, chunk_dir, region
        )
        print(f"  Downloaded {len(result_files)} result files")
        all_result_files.extend(result_files)

    print(
        f"\n  All {len(all_job_info)} batch jobs complete. "
        f"Total result files: {len(all_result_files)}"
    )
    return all_result_files


def parse_results(result_files, judge_model_id, judge_name, output_dir):
    """Parse batch outputs into bt_scores.jsonl.

    Output format per line:
    {"sample_idx": 0, "sample_id": "...", "trait_id": "...", "score": 7, "judge": "..."}

    Returns: {(sample_idx, trait_id): score}
    """
    scores = {}
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

                    _, sample_idx_str, trait_id, sample_id = parts
                    sample_idx = int(sample_idx_str)

                    model_output = rec.get("modelOutput", {})
                    text = extract_batch_model_response(model_output, judge_model_id)
                    score = parse_score(text) if text else None

                    if score is not None:
                        scores[(sample_idx, trait_id)] = {
                            "sample_idx": sample_idx,
                            "sample_id": sample_id,
                            "trait_id": trait_id,
                            "score": score,
                            "judge": judge_name,
                        }
                        parsed += 1
                    else:
                        errors += 1
                except (json.JSONDecodeError, ValueError):
                    errors += 1

    print(f"  Parsed {parsed} scores ({errors} errors)")

    # Write bt_scores.jsonl
    out_path = os.path.join(output_dir, "bt_scores.jsonl")
    with open(out_path, "w") as f:
        f.writelines(json.dumps(scores[key]) + "\n" for key in sorted(scores.keys()))
    print(f"  Scores written to {out_path}")

    return {k: v["score"] for k, v in scores.items()}


def run_sequential(traits, samples, judge_model_id, judge_name, output_dir, region):
    """Score all sample/rubric pairs through the active synchronous provider."""
    client = get_bedrock_client(region)
    total = len(traits) * len(samples)
    completed = 0
    rows = []

    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("id", sample.get("question_id", sample_index)))
        for trait in traits:
            prompt = PAIRWISE_JUDGE_PROMPT.format(
                trait_id=trait["id"],
                trait_description=trait["description"],
                user_prompt=sample.get("prompt", sample.get("question", ""))[:2000],
                response_a=sample.get("response_a", sample.get("candidate", ""))[:3000],
                response_b=sample.get("response_b", sample.get("baseline", ""))[:3000],
            )
            response = invoke_model(
                client,
                judge_model_id,
                prompt,
                temperature=0.0,
                max_tokens=500,
            )
            score = parse_score(response)
            if score is not None:
                rows.append(
                    {
                        "sample_idx": sample_index,
                        "sample_id": sample_id,
                        "trait_id": trait["id"],
                        "score": score,
                        "judge": judge_name,
                    }
                )
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  [{completed}/{total}]")

    out_path = os.path.join(output_dir, "bt_scores.jsonl")
    with open(out_path, "w") as output_file:
        output_file.writelines(json.dumps(row) + "\n" for row in rows)
    print(f"  Scores written to {out_path} ({len(rows)}/{total} parsed)")


def main():
    parser = argparse.ArgumentParser(
        description="Score preference battles for paper-style preference fit"
    )
    parser.add_argument(
        "--battles", required=True, help="JSONL file with pairwise battles"
    )
    parser.add_argument("--traits", required=True, help="JSON traits file")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max battles to score (default: 2000)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model ID (default: config role 'judge')",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for job naming (default: auto from output dir)",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--role-arn", default=DEFAULT_BATCH_ROLE_ARN)
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Use synchronous inference; required for non-Bedrock providers.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    judge_model_id = args.judge_model or model_for_role("judge")
    judge_name = "judge"
    run_id = args.run_id or os.path.basename(args.output.rstrip("/"))

    traits = load_traits(args.traits)
    samples = load_samples(args.battles, max_samples=args.max_samples)
    print(f"Loaded {len(traits)} traits, {len(samples)} samples")
    print(f"Judge: {judge_name} ({judge_model_id})")
    print(f"Total records: {len(traits) * len(samples)}")

    # Check for existing scores
    scores_path = os.path.join(args.output, "bt_scores.jsonl")
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            n_lines = sum(1 for _ in f)
        expected = len(traits) * len(samples)
        if n_lines >= expected * 0.95:
            print(f"  bt_scores.jsonl already has {n_lines}/{expected} lines, skipping")
            return

    if args.no_batch:
        run_sequential(
            traits,
            samples,
            judge_model_id,
            judge_name,
            args.output,
            args.region,
        )
    else:
        args.s3_bucket, args.role_arn = resolve_batch_settings(
            args.s3_bucket, args.role_arn
        )
        batch_paths = prepare_batch(
            traits, samples, judge_model_id, args.output, run_id
        )
        result_files = submit_and_wait(
            batch_paths,
            judge_model_id,
            args.output,
            args.s3_bucket,
            args.role_arn,
            args.region,
            run_id,
        )
        parse_results(result_files, judge_model_id, judge_name, args.output)

    # Save config for reproducibility
    config = {
        "battles": args.battles,
        "traits": args.traits,
        "max_samples": args.max_samples,
        "judge": judge_name,
        "judge_model_id": judge_model_id,
        "n_traits": len(traits),
        "n_samples": len(samples),
        "n_records": len(traits) * len(samples),
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
