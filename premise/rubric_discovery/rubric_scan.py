# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Rubric Applicability Scanner.

For a given set of pre-defined rubrics — any rubric file in the standard schema,
however it was authored — scans battle samples to identify which rubrics are
applicable per prompt. Aggregates counts and outputs top-N rubrics in
discovered-traits format, so a hand-written rubric set can be fed to the same
evaluation commands as a discovered one.

Supports Bedrock batch inference for scale.

Usage:
    # Scan battles against a pre-defined rubric set
    premise discover scan \
        --rubrics <rubrics.json> \
        --input-preference-data <preferences.jsonl> \
        --output results/rubric_scan/predefined/ \
        --max-samples 500

    # Same, at scale, via batch inference
    premise discover scan \
        --rubrics <rubrics.json> \
        --input-preference-data <preferences.jsonl> \
        --output results/rubric_scan/predefined/ \
        --batch --max-samples 2000

    # Resume from existing batch output
    premise discover scan \
        --rubrics <rubrics.json> \
        --input-preference-data <preferences.jsonl> \
        --output results/rubric_scan/predefined/ \
        --skip-inference
"""

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict

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
    resolve_batch_settings,
    submit_batch_job,
)

BATCH_MAX_RECORDS = 10_000
BATCH_MAX_BYTES = 50_000_000  # 50MB — keeps uploads fast and reliable

# ============================================================
# Prompt
# ============================================================

RUBRIC_SCAN_PROMPT = """You are an evaluation design expert. Given a user prompt and two model responses from a pairwise comparison, identify which of the following pre-defined rubrics are APPLICABLE for evaluating the quality difference between the responses.

A rubric is applicable if it captures a meaningful dimension along which the two responses could differ in quality for this specific prompt.

USER PROMPT:
{prompt}

RESPONSE A (excerpt):
{response_a}

RESPONSE B (excerpt):
{response_b}

RUBRICS:
{rubric_list}

YOUR TASK:
Select ALL rubrics that are applicable for evaluating these responses. A rubric is applicable if:
1. The dimension it measures is relevant to this type of prompt
2. The two responses could meaningfully differ along this dimension

OUTPUT FORMAT (strict JSON):
{{
  "applicable_rubrics": [
    {{
      "rubric_id": "the_rubric_id",
      "reason": "brief reason why this rubric applies (1 sentence)"
    }}
  ]
}}

IMPORTANT:
- Return ONLY the JSON object
- Only include rubrics that are genuinely relevant — do not include all rubrics by default
- Be selective: typically 3-8 rubrics apply to any given prompt
- Use the exact rubric IDs from the list above"""


# ============================================================
# Helpers
# ============================================================


def load_rubrics(rubrics_file: str) -> list[dict]:
    """Load rubric definitions from JSON file."""
    with open(rubrics_file) as f:
        data = json.load(f)
    return data.get("rubrics", data.get("traits", []))


def load_battles(arena_file: str, max_samples: int | None = None) -> list[dict]:
    """Load battle samples from JSONL."""
    battles = []
    with open(arena_file) as f:
        for line in f:
            if line.strip():
                battles.append(json.loads(line))
                if max_samples and len(battles) >= max_samples:
                    break
    return battles


def extract_conversation_text(conversation: list) -> tuple[str, str]:
    """Extract prompt and response from conversation structure."""
    prompt = ""
    response = ""
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if isinstance(content, list) and content:
            text = (
                content[0].get("text", "")
                if isinstance(content[0], dict)
                else str(content[0])
            )
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        if role == "user" and not prompt:
            prompt = text or ""
        elif role == "assistant" and not response:
            response = text or ""
    return prompt, response


def format_rubric_list(rubrics: list[dict]) -> str:
    """Format rubrics into a numbered list for the prompt."""
    lines = []
    for i, r in enumerate(rubrics, 1):
        rid = r.get("id", r.get("trait_id", f"rubric_{i}"))
        title = r.get("title", "")
        desc = r.get("description", r.get("trait_description", ""))
        # Truncate long descriptions
        if len(desc) > 300:
            desc = desc[:300] + "..."
        title_str = f" ({title})" if title else ""
        lines.append(f"{i}. **{rid}**{title_str}: {desc}")
    return "\n".join(lines)


def build_prompt(battle: dict, rubrics: list[dict]) -> str | None:
    """Build the scan prompt for a single battle."""
    conv_a = battle.get("conversation_a", [])
    conv_b = battle.get("conversation_b", [])
    prompt_a, response_a = extract_conversation_text(conv_a)
    prompt_b, response_b = extract_conversation_text(conv_b)
    prompt = prompt_a or prompt_b

    if not prompt:
        return None

    return RUBRIC_SCAN_PROMPT.format(
        prompt=prompt[:3000],
        response_a=(response_a or "")[:3000],
        response_b=(response_b or "")[:3000],
        rubric_list=format_rubric_list(rubrics),
    )


def parse_scan_response(response_text: str, valid_ids: set) -> list[str]:
    """Parse LLM response to extract applicable rubric IDs."""
    if not response_text:
        return []
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            applicable = parsed.get("applicable_rubrics", [])
            ids = []
            for item in applicable:
                if isinstance(item, dict):
                    rid = item.get("rubric_id", "")
                elif isinstance(item, str):
                    rid = item
                else:
                    continue
                if rid in valid_ids:
                    ids.append(rid)
            return ids
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return []


# ============================================================
# Sequential inference
# ============================================================


def run_sequential(
    client,
    model_id: str,
    battles: list[dict],
    rubrics: list[dict],
    output_dir: str,
    valid_ids: set,
) -> list[dict]:
    """Run rubric scan sequentially (small datasets)."""
    results = []
    results_file = os.path.join(output_dir, "scan_results.jsonl")

    # Resume from partial results
    done_ids = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done_ids.add(r["battle_id"])
        print(f"  Resuming: {len(done_ids)} already done")

    with open(results_file, "a") as fout:
        for i, battle in enumerate(battles):
            bid = battle.get("id", f"battle_{i}")
            if bid in done_ids:
                continue

            prompt = build_prompt(battle, rubrics)
            if not prompt:
                continue

            response = invoke_model(
                client, model_id, prompt, temperature=0.0, max_tokens=2000
            )
            applicable = parse_scan_response(response, valid_ids)

            entry = {"battle_id": bid, "applicable_rubrics": applicable}
            results.append(entry)
            fout.write(json.dumps(entry) + "\n")
            fout.flush()

            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(battles)}] scanned")

    return results


# ============================================================
# Batch inference
# ============================================================


def prepare_batch_input(
    battles: list[dict],
    rubrics: list[dict],
    model_id: str,
    output_dir: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> list[str]:
    """Prepare batch input JSONL files (split at BATCH_MAX_RECORDS)."""
    records = []
    for i, battle in enumerate(battles):
        bid = battle.get("id", f"battle_{i}")
        prompt = build_prompt(battle, rubrics)
        if not prompt:
            continue
        model_input = get_batch_model_input(
            model_id, prompt, temperature=temperature, max_tokens=max_tokens
        )
        records.append({"recordId": bid, "modelInput": model_input})

    # Split into chunks respecting both record count and file size limits
    chunks = []
    current_chunk = []
    current_size = 0
    for rec in records:
        rec_line = json.dumps(rec) + "\n"
        rec_size = len(rec_line.encode("utf-8"))
        if current_chunk and (
            len(current_chunk) >= BATCH_MAX_RECORDS
            or current_size + rec_size > BATCH_MAX_BYTES
        ):
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(rec)
        current_size += rec_size
    if current_chunk:
        chunks.append(current_chunk)

    paths = []
    for ci, chunk in enumerate(chunks):
        fname = f"batch_input_{ci}.jsonl" if len(chunks) > 1 else "batch_input.jsonl"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            for rec in chunk:
                f.write(json.dumps(rec) + "\n")
        size_mb = os.path.getsize(path) / 1_000_000
        paths.append(path)
        print(f"  Wrote {len(chunk)} records to {fname} ({size_mb:.1f} MB)")

    return paths


def _upload_s3_cli(
    local_path: str, s3_uri: str, region: str, max_retries: int = 3
) -> bool:
    """Upload file to S3 using aws CLI with retries."""
    for attempt in range(max_retries):
        result = subprocess.run(
            ["aws", "s3", "cp", local_path, s3_uri, "--region", region],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return True
        if attempt < max_retries - 1:
            wait = 2 ** (attempt + 1)
            print(f"    Upload attempt {attempt + 1} failed, retrying in {wait}s...")
            import time

            time.sleep(wait)
    print(
        f"  ERROR: S3 upload failed after {max_retries} attempts: {result.stderr.strip()}"
    )
    return False


def run_batch(
    battles: list[dict],
    rubrics: list[dict],
    model_id: str,
    output_dir: str,
    valid_ids: set,
    s3_bucket: str,
    role_arn: str,
    region: str,
) -> list[dict]:
    """Run rubric scan via Bedrock batch inference.

    Upload and submit all batch jobs first, then wait for all to complete.
    """
    batch_dir = os.path.join(output_dir, "batch_output")
    os.makedirs(batch_dir, exist_ok=True)

    input_paths = prepare_batch_input(battles, rubrics, model_id, output_dir)

    mgmt_client = get_bedrock_management_client(region)

    # Phase 1: Upload all files and submit all jobs
    jobs = []  # (job_name, job_arn, s3_output_prefix)
    for pi, local_path in enumerate(input_paths):
        file_hash = compute_file_hash(local_path)[:12]
        job_name = f"premise-rubric-scan-{file_hash}"
        job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]

        s3_key = f"premise/rubric_scan/{job_name}/input.jsonl"
        s3_input_uri = f"s3://{s3_bucket}/{s3_key}"
        s3_output_uri = f"s3://{s3_bucket}/premise/rubric_scan/{job_name}/output/"

        size_mb = os.path.getsize(local_path) / 1_000_000
        print(f"  [{pi + 1}/{len(input_paths)}] Uploading {size_mb:.1f} MB: {job_name}")
        if not _upload_s3_cli(local_path, s3_input_uri, region):
            continue

        print(f"  [{pi + 1}/{len(input_paths)}] Submitting: {job_name}")
        job_arn = submit_batch_job(
            mgmt_client, s3_input_uri, s3_output_uri, job_name, model_id, role_arn
        )
        output_prefix = f"premise/rubric_scan/{job_name}/output/"
        jobs.append((job_name, job_arn, output_prefix))
        print(f"    ARN: {job_arn}")

    if not jobs:
        print("  ERROR: No jobs submitted")
        return []

    print(f"\n  All {len(jobs)} jobs submitted, waiting for completion...")

    # Phase 2: Wait for all jobs to complete
    import time

    pending = {name: (arn, prefix) for name, arn, prefix in jobs}
    completed = {}
    poll_interval = 60
    elapsed = 0
    timeout = 7200

    while pending and elapsed < timeout:
        still_pending = {}
        for name, (arn, prefix) in pending.items():
            try:
                response = mgmt_client.get_model_invocation_job(jobIdentifier=arn)
                status = response.get("status", "Unknown")
            except Exception as e:
                print(f"    Error checking {name}: {e}")
                still_pending[name] = (arn, prefix)
                continue

            if status == "Completed":
                print(f"  ✓ {name} completed ({elapsed}s)")
                completed[name] = (arn, prefix)
            elif status in ("Failed", "Stopped"):
                print(f"  ✗ {name} {status}: {response.get('message', '')}")
            else:
                still_pending[name] = (arn, prefix)

        pending = still_pending
        if pending:
            names = ", ".join(pending.keys())
            print(
                f"  Waiting... {len(pending)} jobs pending ({elapsed}s): {names[:100]}"
            )
            time.sleep(poll_interval)
            elapsed += poll_interval

    if pending:
        print(f"  WARNING: {len(pending)} jobs timed out")

    # Phase 3: Download and parse all completed results
    all_results = []
    for name, (arn, prefix) in completed.items():
        local_files = download_batch_results(s3_bucket, prefix, batch_dir, region)
        print(f"  Downloaded {len(local_files)} files from {name}")
        for fpath in local_files:
            with open(fpath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    bid = rec.get("recordId", "")
                    model_output = rec.get("modelOutput", {})
                    response_text = extract_batch_model_response(model_output, model_id)
                    applicable = parse_scan_response(response_text, valid_ids)
                    all_results.append(
                        {"battle_id": bid, "applicable_rubrics": applicable}
                    )

    # Save parsed results
    results_file = os.path.join(output_dir, "scan_results.jsonl")
    with open(results_file, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in all_results)
    print(f"  Saved {len(all_results)} scan results")

    return all_results


def parse_existing_results(
    output_dir: str, valid_ids: set, model_id: str
) -> list[dict]:
    """Parse results from existing batch output files."""
    results_file = os.path.join(output_dir, "scan_results.jsonl")
    if os.path.exists(results_file):
        results = []
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    # Fallback: parse from batch_output/
    batch_dir = os.path.join(output_dir, "batch_output")
    if not os.path.exists(batch_dir):
        return []

    results = []
    for fname in sorted(os.listdir(batch_dir)):
        if not fname.endswith(".jsonl") and not fname.endswith(".jsonl.out"):
            continue
        with open(os.path.join(batch_dir, fname)) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                bid = rec.get("recordId", "")
                model_output = rec.get("modelOutput", {})
                response_text = extract_batch_model_response(model_output, model_id)
                applicable = parse_scan_response(response_text, valid_ids)
                results.append({"battle_id": bid, "applicable_rubrics": applicable})

    if results:
        out_path = os.path.join(output_dir, "scan_results.jsonl")
        with open(out_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)

    return results


# ============================================================
# Aggregation
# ============================================================


def aggregate_results(
    results: list[dict],
    rubrics: list[dict],
    top_n: int = 20,
) -> tuple[list[dict], dict]:
    """Aggregate scan results into per-rubric counts and top-N output."""
    rubric_by_id = {r.get("id", r.get("trait_id", "")): r for r in rubrics}

    # Count applicability
    counts = Counter()
    source_ids = defaultdict(list)
    for r in results:
        for rid in r["applicable_rubrics"]:
            counts[rid] += 1
            if len(source_ids[rid]) < 50:
                source_ids[rid].append(r["battle_id"])

    # Build full list sorted by count
    all_rubrics = []
    for rid, count in counts.most_common():
        rubric = rubric_by_id.get(rid, {})
        all_rubrics.append(
            {
                "id": rid,
                "title": rubric.get("title", ""),
                "category": "external",
                "description": rubric.get(
                    "description", rubric.get("trait_description", "")
                ),
                "source_count": count,
                "source_ids": source_ids[rid][:10],
                "applicability_rate": round(count / len(results) * 100, 1)
                if results
                else 0,
            }
        )

    top = all_rubrics[:top_n]

    # Summary
    summary = {
        "n_samples_scanned": len(results),
        "n_rubrics_total": len(rubrics),
        "n_rubrics_matched": len(counts),
        "n_rubrics_unmatched": len(rubrics) - len(counts),
        "top_n": top_n,
        "top_rubrics": [
            {
                "id": r["id"],
                "source_count": r["source_count"],
                "applicability_rate": r["applicability_rate"],
                "description": r["description"][:100] + "..."
                if len(r["description"]) > 100
                else r["description"],
            }
            for r in top
        ],
    }

    return all_rubrics, summary


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Rubric Applicability Scanner")
    parser.add_argument(
        "--rubrics", required=True, help="JSON file with rubric definitions"
    )
    parser.add_argument(
        "--input-preference-data",
        dest="arena_data",
        help="JSONL file with pairwise preferences",
    )
    parser.add_argument("--arena-data", dest="arena_data", help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--judge-model", default=None, help="Model ID (default: deepseek_v3)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to scan"
    )
    parser.add_argument(
        "--top-n", type=int, default=20, help="Number of top rubrics to output"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Use Bedrock batch inference"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference, parse existing results",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--role-arn", default=DEFAULT_BATCH_ROLE_ARN)
    args = parser.parse_args()
    if not args.arena_data:
        parser.error("--input-preference-data is required")

    os.makedirs(args.output, exist_ok=True)

    # Load rubrics
    rubrics = load_rubrics(args.rubrics)
    valid_ids = {r.get("id", r.get("trait_id", "")) for r in rubrics}
    print(f"Loaded {len(rubrics)} rubrics from {args.rubrics}")

    # Resolve model
    from premise.providers import model_for_role

    model_id = args.judge_model or model_for_role("judge")
    print(f"Model: {model_id}")

    # Load battles
    battles = load_battles(args.arena_data, args.max_samples)
    print(f"Loaded {len(battles)} battles from {args.arena_data}")

    # Run scan
    if args.skip_inference:
        print("\n=== Parsing existing results ===")
        results = parse_existing_results(args.output, valid_ids, model_id)
        print(f"  Found {len(results)} existing results")
    elif args.batch:
        args.s3_bucket, args.role_arn = resolve_batch_settings(
            args.s3_bucket, args.role_arn
        )
        print("\n=== Running batch inference ===")
        results = run_batch(
            battles,
            rubrics,
            model_id,
            args.output,
            valid_ids,
            args.s3_bucket,
            args.role_arn,
            args.region,
        )
    else:
        print("\n=== Running sequential inference ===")
        client = get_bedrock_client(args.region)
        results = run_sequential(
            client, model_id, battles, rubrics, args.output, valid_ids
        )

    if not results:
        print("ERROR: No results to aggregate")
        return

    # Aggregate
    print(f"\n=== Aggregating {len(results)} results ===")
    all_rubrics, summary = aggregate_results(results, rubrics, args.top_n)

    # Write outputs
    # 1. Full rubric counts (discovered_traits format)
    all_rubrics_file = os.path.join(args.output, "rubric_applicability.json")
    with open(all_rubrics_file, "w") as f:
        json.dump({"schema_version": "1.0", "rubrics": all_rubrics}, f, indent=2)
    print(f"  Wrote {len(all_rubrics)} rubrics to rubric_applicability.json")

    # 2. Top-N (same format as traits_top_N.json)
    top_file = os.path.join(args.output, f"rubrics_top_{args.top_n}.json")
    top_rubrics = all_rubrics[: args.top_n]
    with open(top_file, "w") as f:
        json.dump({"schema_version": "1.0", "rubrics": top_rubrics}, f, indent=2)
    print(f"  Wrote top {len(top_rubrics)} rubrics to rubrics_top_{args.top_n}.json")

    # 3. Summary
    summary_file = os.path.join(args.output, "scan_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print("  Wrote summary to scan_summary.json")

    # Print top rubrics
    print(f"\n=== Top {args.top_n} Rubrics by Applicability ===")
    for i, r in enumerate(top_rubrics, 1):
        print(
            f"  {i:2d}. {r['id']:<40s} count={r['source_count']:5d}  rate={r['applicability_rate']:5.1f}%"
        )

    unmatched = [
        r
        for r in rubrics
        if r.get("id", r.get("trait_id", "")) not in {x["id"] for x in all_rubrics}
    ]
    if unmatched:
        print(f"\n  Unmatched rubrics ({len(unmatched)}):")
        for r in unmatched:
            print(f"    - {r.get('id', r.get('trait_id', ''))}")


if __name__ == "__main__":
    main()
