# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Paraphrase Stability Experiment.

Measures whether rewording a rubric definition changes evaluation scores.

Default mode: Bedrock batch inference (server-side parallel).
Fallback: --no-batch for sequential Converse API calls.

Usage:
    # Batch mode (default)
    premise audit paraphrase --traits <traits.json> \
        --samples <samples.jsonl> --output results/paraphrase/

    # Sequential mode
    premise audit paraphrase --traits <traits.json> \
        --samples <samples.jsonl> --output results/paraphrase/ --no-batch
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

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
from premise.utils.metrics import parse_score
from premise.utils.prompts import PAIRWISE_JUDGE_PROMPT, PARAPHRASE_PROMPT

BATCH_MAX_RECORDS = 10_000


def generate_paraphrases(client, model_id: str, description: str, k: int = 5) -> list:
    paraphrases = []
    for _ in range(k):
        prompt = PARAPHRASE_PROMPT.format(definition=description)
        response = invoke_model(client, model_id, prompt, temperature=0.8)
        if response:
            paraphrases.append(response.strip().strip('"').strip("'"))
    return paraphrases


# ============================================================
# Batch Inference Mode
# ============================================================


def prepare_paraphrase_batch(
    traits, samples, paraphrases_data, k_paraphrases, judge_model_id, output_dir
):
    """Build JSONL batch input files for all (trait, sample, variant) combos.

    recordId format: "{global_idx}|{trait_id}|{sample_idx}|{variant_idx}"
    variant_idx 0 = original definition, 1..k = paraphrases.
    """
    records = []
    for trait in traits:
        trait_id = trait["id"]
        original_desc = paraphrases_data[trait_id]["original"]
        variant_descs = [original_desc] + paraphrases_data[trait_id]["paraphrases"][
            :k_paraphrases
        ]

        for v_idx, desc in enumerate(variant_descs):
            for s_idx, sample in enumerate(samples):
                prompt = PAIRWISE_JUDGE_PROMPT.format(
                    trait_id=trait_id,
                    trait_description=desc,
                    user_prompt=sample.get("prompt", sample.get("question", "")),
                    response_a=sample.get("response_a", sample.get("candidate", ""))[
                        :3000
                    ],
                    response_b=sample.get("response_b", sample.get("baseline", ""))[
                        :3000
                    ],
                )
                records.append(
                    {
                        "trait_id": trait_id,
                        "sample_idx": s_idx,
                        "variant_idx": v_idx,
                        "prompt": prompt,
                    }
                )

    print(
        f"  Total records: {len(records)} "
        f"({len(traits)} traits x {len(samples)} samples x {1 + k_paraphrases} variants)"
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
                    f"|{rec['sample_idx']}|{rec['variant_idx']}"
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

        print(f"  Chunk {chunk_idx}: {len(chunk)} records -> {batch_path}")
        batch_paths.append(batch_path)

    return batch_paths


def parse_paraphrase_batch_results(result_files, judge_model_id):
    """Parse batch output into {trait_id: {variant_idx: {sample_idx: score}}}.

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

                    _, trait_id, sample_idx, variant_idx = parts
                    sample_idx = int(sample_idx)
                    variant_idx = int(variant_idx)

                    model_output = rec.get("modelOutput", {})
                    text = extract_batch_model_response(model_output, judge_model_id)
                    score = parse_score(text) if text else None

                    scores[trait_id][variant_idx][sample_idx] = score
                    if score is not None:
                        parsed += 1
                    else:
                        errors += 1

                except (json.JSONDecodeError, ValueError):
                    errors += 1

    print(f"  Parsed {parsed} scores, {errors} errors")
    return scores


def run_batch_paraphrase(
    traits,
    samples,
    paraphrases_data,
    k_paraphrases,
    judge_model_id,
    output_dir,
    s3_bucket,
    role_arn,
    region,
):
    """Run paraphrase scoring via Bedrock batch inference."""

    # Check for already-downloaded results (resume)
    local_results_dir = os.path.join(output_dir, "batch_output")
    existing_results = [
        os.path.join(local_results_dir, f)
        for f in (
            os.listdir(local_results_dir) if os.path.isdir(local_results_dir) else []
        )
        if f.endswith(".jsonl.out")
    ]
    if not existing_results and os.path.isdir(local_results_dir):
        # Check subdirectories (chunk_0/, chunk_1/, ...)
        for sub in os.listdir(local_results_dir):
            sub_path = os.path.join(local_results_dir, sub)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.endswith(".jsonl.out"):
                        existing_results.append(os.path.join(sub_path, f))

    if existing_results:
        print(
            f"  Found {len(existing_results)} existing batch result files, "
            f"skipping upload/submit"
        )
        return parse_paraphrase_batch_results(existing_results, judge_model_id)

    # Prepare batch inputs
    batch_paths = prepare_paraphrase_batch(
        traits,
        samples,
        paraphrases_data,
        k_paraphrases,
        judge_model_id,
        output_dir,
    )

    bedrock = get_bedrock_management_client(region)
    job_arns = []

    for chunk_idx, batch_path in enumerate(batch_paths):
        file_hash = compute_file_hash(batch_path)[:12]
        suffix = f"-{chunk_idx}" if len(batch_paths) > 1 else ""
        job_name = f"premise-paraphrase-{file_hash}{suffix}"
        job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]

        s3_key = f"premise/paraphrase/{job_name}/input.jsonl"
        s3_output_prefix = f"s3://{s3_bucket}/premise/paraphrase/{job_name}/output/"

        print(f"\n  --- Chunk {chunk_idx}/{len(batch_paths) - 1} ---")
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
        job_arns.append((chunk_idx, job_name, job_arn, s3_output_prefix))

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
    return parse_paraphrase_batch_results(all_result_files, judge_model_id)


# ============================================================
# Metrics computation
# ============================================================


def compute_paraphrase_metrics(scores, traits, k_paraphrases):
    """Compute Spearman rho between original and each paraphrase variant.

    Args:
        scores: {trait_id: {variant_idx: {sample_idx: score}}}
    Returns:
        dict of per-trait results.
    """
    results = {}

    for trait in traits:
        trait_id = trait["id"]
        trait_scores = scores.get(trait_id, {})
        original_scores = trait_scores.get(0, {})

        per_paraphrase_rho = []
        for v_idx in range(1, k_paraphrases + 1):
            variant_scores = trait_scores.get(v_idx, {})

            # Gather valid pairs
            valid_orig = []
            valid_var = []
            for s_idx in sorted(
                set(original_scores.keys()) & set(variant_scores.keys())
            ):
                o = original_scores[s_idx]
                v = variant_scores[s_idx]
                if o is not None and v is not None:
                    valid_orig.append(o)
                    valid_var.append(v)

            if len(valid_orig) >= 10:
                rho, _ = stats.spearmanr(valid_orig, valid_var)
                per_paraphrase_rho.append(round(rho, 3))
            else:
                per_paraphrase_rho.append(None)

        valid_rhos = [r for r in per_paraphrase_rho if r is not None]
        raw_rho = round(np.mean(valid_rhos), 3) if valid_rhos else 0.0

        n_orig_valid = sum(1 for v in original_scores.values() if v is not None)
        results[trait_id] = {
            "raw_rho": raw_rho,
            "winrate_rho": raw_rho,
            "n_variants": len(valid_rhos),
            "n_samples": n_orig_valid,
            "per_variant_rho": per_paraphrase_rho,
        }

    return results


# ============================================================
# Sequential Mode (fallback)
# ============================================================


def run_sequential_paraphrase(
    traits, samples, paraphrases_data, k_paraphrases, judge_model_id, region
):
    """Run paraphrase scoring via sequential Converse API calls.

    Returns: {trait_id: {variant_idx: {sample_idx: score}}}
    """
    client = get_bedrock_client(region)
    total_calls = len(traits) * len(samples) * (1 + k_paraphrases)
    print(f"  Total API calls: {total_calls}")

    scores = defaultdict(lambda: defaultdict(dict))
    call_count = 0

    for trait in traits:
        trait_id = trait["id"]
        original_desc = paraphrases_data[trait_id]["original"]
        variant_descs = [original_desc] + paraphrases_data[trait_id]["paraphrases"][
            :k_paraphrases
        ]
        print(f"\n=== {trait_id} ===")

        for v_idx, desc in enumerate(variant_descs):
            for s_idx, sample in enumerate(samples):
                call_count += 1
                if call_count % 50 == 0:
                    print(f"  [{call_count}/{total_calls}]")

                prompt = PAIRWISE_JUDGE_PROMPT.format(
                    trait_id=trait_id,
                    trait_description=desc,
                    user_prompt=sample.get("prompt", sample.get("question", ""))[:2000],
                    response_a=sample.get("response_a", sample.get("candidate", ""))[
                        :3000
                    ],
                    response_b=sample.get("response_b", sample.get("baseline", ""))[
                        :3000
                    ],
                )
                response = invoke_model(client, judge_model_id, prompt, temperature=0.0)
                scores[trait_id][v_idx][s_idx] = parse_score(response)

            print(f"  Variant {v_idx}/{len(variant_descs) - 1} complete")

    return scores


# ============================================================
# Main
# ============================================================


def run_paraphrase_experiment(
    traits_file: str,
    samples_file: str,
    output_dir: str,
    k_paraphrases: int = 5,
    max_samples: int = 200,
    judge_model: str | None = None,
    no_batch: bool = False,
    region: str = DEFAULT_REGION,
    s3_bucket: str = DEFAULT_S3_BUCKET,
    role_arn: str = DEFAULT_BATCH_ROLE_ARN,
):
    if judge_model is None:
        from premise.providers import model_for_role

        judge_model = model_for_role("judge")
    if not no_batch:
        s3_bucket, role_arn = resolve_batch_settings(s3_bucket, role_arn)

    os.makedirs(output_dir, exist_ok=True)
    client = get_bedrock_client(region)
    traits = load_traits(traits_file)
    samples = load_samples(samples_file, max_samples)

    print(f"Loaded {len(traits)} traits, {len(samples)} samples, k={k_paraphrases}")
    print(f"Judge model: {judge_model}")
    print(f"Batch inference: {'disabled' if no_batch else 'enabled'}")
    total_calls = len(traits) * len(samples) * (1 + k_paraphrases)
    print(f"Total scoring calls: {total_calls}")

    # Step 1: Generate paraphrases (or load existing)
    para_file = os.path.join(output_dir, "paraphrases.json")
    if os.path.isfile(para_file):
        print(f"\n--- Loading existing paraphrases from {para_file} ---")
        with open(para_file) as f:
            paraphrases_data = json.load(f)
        # Verify all traits present
        missing = [t["id"] for t in traits if t["id"] not in paraphrases_data]
        if missing:
            print(
                f"  WARNING: {len(missing)} traits missing paraphrases, regenerating..."
            )
            for trait in traits:
                if trait["id"] not in paraphrases_data:
                    print(f"    {trait['id']}...")
                    variants = generate_paraphrases(
                        client, judge_model, trait["description"], k_paraphrases
                    )
                    paraphrases_data[trait["id"]] = {
                        "original": trait["description"],
                        "paraphrases": variants,
                    }
            with open(para_file, "w") as f:
                json.dump(paraphrases_data, f, indent=2)
    else:
        print("\n--- Generating paraphrases ---")
        paraphrases_data = {}
        for trait in traits:
            trait_id = trait["id"]
            print(f"  {trait_id}...")
            variants = generate_paraphrases(
                client, judge_model, trait["description"], k_paraphrases
            )
            paraphrases_data[trait_id] = {
                "original": trait["description"],
                "paraphrases": variants,
            }
        with open(para_file, "w") as f:
            json.dump(paraphrases_data, f, indent=2)

    print(f"  Paraphrases ready for {len(paraphrases_data)} traits")

    # Step 2: Score all variants
    print("\n--- Scoring ---")
    if no_batch:
        scores = run_sequential_paraphrase(
            traits,
            samples,
            paraphrases_data,
            k_paraphrases,
            judge_model,
            region,
        )
    else:
        scores = run_batch_paraphrase(
            traits,
            samples,
            paraphrases_data,
            k_paraphrases,
            judge_model,
            output_dir,
            s3_bucket,
            role_arn,
            region,
        )

    # Step 3: Compute metrics
    results = compute_paraphrase_metrics(scores, traits, k_paraphrases)

    output_file = os.path.join(output_dir, "paraphrase_scores.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n=== Summary ===")
    for tid, r in sorted(results.items(), key=lambda x: -x[1]["raw_rho"]):
        rho = r["raw_rho"]
        flag = "PASS" if rho >= 0.85 else ("REVIEW" if rho >= 0.7 else "FAIL")
        print(f"  {tid}: ρ={rho:.3f} [{flag}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PReMISE paraphrase experiment")
    parser.add_argument("--traits", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
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
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--role-arn", default=DEFAULT_BATCH_ROLE_ARN)
    args = parser.parse_args()
    run_paraphrase_experiment(
        args.traits,
        args.samples,
        args.output,
        k_paraphrases=args.k,
        max_samples=args.max_samples,
        judge_model=args.judge_model,
        no_batch=args.no_batch,
        region=args.region,
        s3_bucket=args.s3_bucket,
        role_arn=args.role_arn,
    )
