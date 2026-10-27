# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Preference-Fit Audit.

Holistic rubric-conditioned preference prediction. For each normalized
battle (prompt, response_a, response_b, winner), the judge sees the
*full* rubric set in a single prompt and picks A or B. Position is
randomized per-pair (deterministic by sample id) to remove order bias.

Metric:
    accuracy = #(judge picks ground-truth winner) / N_valid

Inputs match the rest of the audit suite:
    --traits   <traits.json>       (canonical PReMISE schema)
    --samples  <battles.jsonl>
    --output   results/pref_fit/

Default mode: Bedrock batch inference. Fallback: --no-batch.

Usage:
    premise audit pref-fit \\
        --traits <traits.json> \\
        --samples <battles.jsonl> \\
        --output results/pref_fit/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

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
    resolve_batch_settings,
    submit_batch_job,
    upload_to_s3,
    wait_for_batch_job,
)
from premise.utils.prompts import PREFERENCE_FIT_JUDGE_PROMPT

BATCH_MAX_RECORDS = 10_000


def format_rubric_block(rubric_obj) -> str:
    """Format an arbitrary rubric file payload into a numbered block.

    Accepts PReMISE's own schema plus the other rubric shapes seen in the
    wild, so an externally authored rubric can be audited without conversion:
      - {"rubrics_text": "..."}                     (use verbatim)
      - {"rubrics": [{"id"|"category", "description"}, ...]}
      - {"traits": [...]}                            (alias of rubrics)
      - {"rules": [str | {"name", "description"}, ...]}
      - {"theme_tips": [{"theme", "tips": [...]}, ...]}
      - bare list[str] | list[dict]
    """
    if isinstance(rubric_obj, dict):
        if rubric_obj.get("rubrics_text"):
            return str(rubric_obj["rubrics_text"])
        if rubric_obj.get("theme_tips"):
            lines = []
            for i, cat in enumerate(rubric_obj["theme_tips"], 1):
                lines.append(f"{i}. {cat.get('theme', '')}".rstrip())
                for tip in cat.get("tips", []):
                    lines.append(f"   - {tip}")
            return "\n".join(lines)
        items = (
            rubric_obj.get("rubrics")
            or rubric_obj.get("traits")
            or rubric_obj.get("rules")
            or []
        )
    else:
        items = rubric_obj or []

    lines = []
    for i, t in enumerate(items, 1):
        if isinstance(t, str):
            lines.append(f"{i}. {t}")
            continue
        if isinstance(t, dict):
            cat = (t.get("category") or "").strip()
            # "discovered" is a placeholder PReMISE writes for every trait; it
            # is not a meaningful category and pollutes the judge prompt.
            label = (
                cat if cat and cat.lower() != "discovered" else (t.get("name") or "")
            )
            desc = t.get("description") or t.get("definition") or ""
            if label and desc:
                lines.append(f"{i}. {label}: {desc}")
            elif desc:
                lines.append(f"{i}. {desc}")
            elif label:
                lines.append(f"{i}. {label}")
            continue
        lines.append(f"{i}. {t}")
    return "\n".join(lines)


def count_rubric_items(rubric_obj) -> int:
    """Best-effort count of rubric items (for tracker n_rubrics)."""
    if isinstance(rubric_obj, dict):
        if rubric_obj.get("rubrics_text"):
            return 1
        if rubric_obj.get("theme_tips"):
            return sum(len(c.get("tips", [])) or 1 for c in rubric_obj["theme_tips"])
        return len(
            rubric_obj.get("rubrics")
            or rubric_obj.get("traits")
            or rubric_obj.get("rules")
            or []
        )
    return len(rubric_obj or [])


def _swap_for(sample_id: str) -> bool:
    """Deterministic per-sample coin flip — keeps batch + sequential aligned
    and lets re-runs reproduce the same A/B assignment."""
    h = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return (h[0] & 1) == 1


def _ground_truth(sample: dict) -> str:
    """Map normalized battle winner to 'A' or 'B' in the *unswapped* frame
    (i.e. before per-pair randomization). Returns '' for ties / unknown."""
    w = (sample.get("winner") or "").strip().lower()
    if w == "model_a":
        return "A"
    if w == "model_b":
        return "B"
    return ""


def _sample_key(sample: dict, idx: int) -> str:
    return str(sample.get("id") or sample.get("question_id") or idx)


def _parse_judge_answer(text: str) -> str:
    """Extract 'A' or 'B' from the judge response. Returns '' if invalid.

    Many judges emit verdicts in formats other than the requested
    <verdict>A</verdict> XML — bracketed [[A]], markdown's **Winner:** A, or
    plain English ("Response A is better"). Without these fallbacks a large
    share of otherwise-correct verdicts silently score as wrong, so the
    fallbacks matter more to measured accuracy than the prompt does.
    """
    if not text:
        return ""
    # Canonical: <verdict>A</verdict>
    m = re.search(r"<verdict>\s*([AB])\s*</verdict>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # JSON: {"answer": "A"} or {"verdict": "A"}
    try:
        obj = json.loads(text)
        ans = (obj.get("answer") or obj.get("verdict") or "").strip().upper()
        if ans in {"A", "B"}:
            return ans
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r'"(?:answer|verdict)"\s*:\s*"([AB])"', text)
    if m:
        return m.group(1).upper()
    # Bracketed verdict residue from other judge harnesses: [[A]] / [[B]]
    m = re.search(r"\[\[\s*([AB])\s*\]\]", text)
    if m:
        return m.group(1).upper()
    # Markdown: **Winner:** A  or  **Winner: A**  or  **Winner:** Response A
    m = re.search(
        r"\*\*Winner:?\s*\*?\*?[:\s]*(?:Response\s+)?([AB])\b", text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()
    m = re.search(r"\bwinner[:\s]*(?:is\s+)?response\s*([AB])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # English: "Response A is better/stronger/wins"
    m = re.search(
        r"Response\s+([AB])\s+(?:is|appears|seems|provides|delivers|wins)\s+(?:better|stronger|more|the\s+best|the\s+winner)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    m = re.search(
        r"(?:better|preferred|chosen|stronger)\s+response\s+is\s+(?:Response\s+)?([AB])\b",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    # Verdict-like context anywhere
    m = re.search(
        r"(?:verdict|answer|choice|pick|better)[:\s]*([AB])\b", text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()
    # Last resort: "Therefore, A" near end
    tail = text[-200:]
    m = re.search(r"\bTherefore[^A-Za-z]*([AB])\b", tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


def _filter_labeled(samples: list) -> list:
    """Keep only samples with a model_a / model_b label (ties dropped)."""
    return [s for s in samples if _ground_truth(s) in {"A", "B"}]


# ============================================================
# Batch Inference Mode
# ============================================================


def prepare_pref_fit_batch(rubric_obj, samples, judge_model_id, output_dir):
    """Build JSONL batch input file. recordId = '{global_idx}|{sample_key}|{swapped}'."""
    rubric_block = format_rubric_block(rubric_obj)
    records = []
    for s_idx, sample in enumerate(samples):
        key = _sample_key(sample, s_idx)
        swap = _swap_for(key)
        a_text = sample.get("response_b", "") if swap else sample.get("response_a", "")
        b_text = sample.get("response_a", "") if swap else sample.get("response_b", "")
        prompt = PREFERENCE_FIT_JUDGE_PROMPT.format(
            rubric_block=rubric_block,
            user_prompt=(sample.get("prompt") or sample.get("question") or "")[:2000],
            response_a=(a_text or "")[:3000],
            response_b=(b_text or "")[:3000],
        )
        records.append((key, swap, prompt))

    chunks = [
        records[i : i + BATCH_MAX_RECORDS]
        for i in range(0, len(records), BATCH_MAX_RECORDS)
    ]

    batch_paths = []
    for chunk_idx, chunk in enumerate(chunks):
        suffix = f"_{chunk_idx}" if len(chunks) > 1 else ""
        batch_path = os.path.join(output_dir, f"batch_input{suffix}.jsonl")
        with open(batch_path, "w") as f:
            for i, (key, swap, prompt) in enumerate(chunk):
                global_idx = chunk_idx * BATCH_MAX_RECORDS + i
                # Sample keys can contain '|' — escape with %7C so we can split safely.
                safe_key = str(key).replace("|", "%7C")
                record_id = f"{global_idx}|{safe_key}|{int(swap)}"
                model_input = get_batch_model_input(
                    judge_model_id,
                    prompt,
                    temperature=0.0,
                    max_tokens=4096,
                )
                # Add system message to anchor evaluator role
                if "messages" in model_input:
                    model_input["messages"].insert(
                        0,
                        {
                            "role": "system",
                            "content": "You are a pairwise response evaluator. Read the rubric and two responses, then output your verdict.",
                        },
                    )
                f.write(
                    json.dumps(
                        {
                            "recordId": record_id,
                            "modelInput": model_input,
                        }
                    )
                    + "\n"
                )
        batch_paths.append(batch_path)

    print(f"  {len(records)} records -> {len(chunks)} chunk(s)")
    return batch_paths


def parse_pref_fit_batch_results(result_files, judge_model_id):
    """Parse batch output. Returns {sample_key: {'answer': 'A'|'B'|'', 'swapped': bool}}."""
    by_key = {}
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
                    parts = record_id.split("|", 2)
                    if len(parts) < 3:
                        errors += 1
                        continue
                    _, safe_key, swap_str = parts
                    sample_key = safe_key.replace("%7C", "|")
                    swap = swap_str == "1"

                    text = extract_batch_model_response(
                        rec.get("modelOutput", {}), judge_model_id
                    )
                    ans = _parse_judge_answer(text)
                    by_key[sample_key] = {"answer": ans, "swapped": swap}
                    if ans:
                        parsed += 1
                    else:
                        errors += 1
                except (json.JSONDecodeError, ValueError):
                    errors += 1

    print(f"  Parsed {parsed} answers, {errors} errors")
    return by_key


def run_batch_pref_fit(
    rubric_obj,
    samples,
    judge_model_id,
    judge_name,
    output_dir,
    s3_bucket,
    role_arn,
    region,
):
    """Submit one batch job, wait, download, parse."""
    local_results_dir = os.path.join(output_dir, "batch_output")
    existing = []
    if os.path.isdir(local_results_dir):
        for root, _dirs, files in os.walk(local_results_dir):
            for f in files:
                if f.endswith(".jsonl.out"):
                    existing.append(os.path.join(root, f))
    if existing:
        print(
            f"  Found {len(existing)} existing batch result files, "
            f"skipping upload/submit"
        )
        return parse_pref_fit_batch_results(existing, judge_model_id)

    bedrock = get_bedrock_management_client(region)
    batch_paths = prepare_pref_fit_batch(
        rubric_obj, samples, judge_model_id, output_dir
    )

    job_info = []
    for chunk_idx, batch_path in enumerate(batch_paths):
        file_hash = compute_file_hash(batch_path)[:12]
        safe_judge = re.sub(r"[^a-zA-Z0-9]", "-", judge_name.lower())
        suffix = f"-{chunk_idx}" if len(batch_paths) > 1 else ""
        job_name = f"premise-pref-fit-{safe_judge}-{file_hash}{suffix}"
        job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]

        s3_key = f"premise/pref_fit/{job_name}/input.jsonl"
        s3_output_prefix = f"s3://{s3_bucket}/premise/pref_fit/{job_name}/output/"

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
        job_info.append((chunk_idx, job_name, job_arn, s3_output_prefix))

    all_result_files = []
    for chunk_idx, job_name, job_arn, s3_output_prefix in job_info:
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

    return parse_pref_fit_batch_results(all_result_files, judge_model_id)


# ============================================================
# Sequential Mode (fallback)
# ============================================================


def run_sequential_pref_fit(rubric_obj, samples, judge_model_id, region):
    """Sequential Converse API fallback. Returns {sample_key: {answer, swapped}}."""
    client = get_bedrock_client(region)
    rubric_block = format_rubric_block(rubric_obj)
    by_key = {}
    for s_idx, sample in enumerate(samples):
        key = _sample_key(sample, s_idx)
        swap = _swap_for(key)
        a_text = sample.get("response_b", "") if swap else sample.get("response_a", "")
        b_text = sample.get("response_a", "") if swap else sample.get("response_b", "")
        prompt = PREFERENCE_FIT_JUDGE_PROMPT.format(
            rubric_block=rubric_block,
            user_prompt=(sample.get("prompt") or sample.get("question") or "")[:2000],
            response_a=(a_text or "")[:3000],
            response_b=(b_text or "")[:3000],
        )
        text = invoke_model(
            client, judge_model_id, prompt, temperature=0.0, max_tokens=2048
        )
        ans = _parse_judge_answer(text)
        by_key[key] = {"answer": ans, "swapped": swap}
        if (s_idx + 1) % 25 == 0:
            print(f"  [{s_idx + 1}/{len(samples)}]")
    return by_key


# ============================================================
# Metrics
# ============================================================


def compute_pref_fit_metrics(by_key, samples):
    """Map judge answers (in swapped frame) back to ground truth.

    Returns (summary_dict, per_sample_details)
    """
    n_correct = 0
    n_total = 0
    n_invalid = 0
    details = []

    for s_idx, sample in enumerate(samples):
        key = _sample_key(sample, s_idx)
        truth = _ground_truth(sample)  # 'A' or 'B' in original (unswapped) frame
        rec = by_key.get(key)
        if not rec or not rec.get("answer"):
            n_invalid += 1
            details.append(
                {
                    "sample_id": key,
                    "winner": sample.get("winner", ""),
                    "answer": "",
                    "predicted": "",
                    "correct": False,
                    "swapped": rec.get("swapped") if rec else None,
                }
            )
            continue

        ans = rec["answer"]  # 'A' or 'B' in *swapped* frame
        swapped = rec["swapped"]
        # Map back: if swapped, judge's A is original B and vice versa.
        predicted = ("B" if ans == "A" else "A") if swapped else ans
        correct = predicted == truth
        n_total += 1
        if correct:
            n_correct += 1
        details.append(
            {
                "sample_id": key,
                "winner": sample.get("winner", ""),
                "answer": ans,
                "predicted": predicted,
                "correct": correct,
                "swapped": swapped,
            }
        )

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    summary = {
        "accuracy": round(accuracy, 4),
        "n_correct": n_correct,
        "n_total": n_total,
        "n_invalid": n_invalid,
    }
    return summary, details


# ============================================================
# Main
# ============================================================


def run_pref_fit_experiment(
    traits_file: str,
    samples_file: str,
    output_dir: str,
    max_samples: int = 500,
    judge_model: str | None = None,
    judge_name: str = "judge",
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
    with open(traits_file) as _rf:
        rubric_obj = json.load(_rf)
    n_rubrics = count_rubric_items(rubric_obj)
    samples = load_samples(samples_file, max_samples)
    samples = _filter_labeled(samples)

    print(f"Loaded {n_rubrics} rubrics, {len(samples)} labeled battles")
    print(f"Judge: {judge_name} ({judge_model})")
    print(f"Batch inference: {'disabled' if no_batch else 'enabled'}")

    if no_batch:
        by_key = run_sequential_pref_fit(rubric_obj, samples, judge_model, region)
    else:
        by_key = run_batch_pref_fit(
            rubric_obj,
            samples,
            judge_model,
            judge_name,
            output_dir,
            s3_bucket,
            role_arn,
            region,
        )

    summary, details = compute_pref_fit_metrics(by_key, samples)

    out = {
        **summary,
        "traits_file": traits_file,
        "samples_file": samples_file,
        "judge_model": judge_model,
        "judge_name": judge_name,
        "n_rubrics": n_rubrics,
    }
    out_file = os.path.join(output_dir, "pref_fit.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    raw_file = os.path.join(output_dir, "pref_fit_raw.jsonl")
    with open(raw_file, "w") as rf:
        rf.writelines(json.dumps(d) + "\n" for d in details)

    print(f"\nResults saved to {out_file}")
    print(
        f"  accuracy={summary['accuracy']:.4f} "
        f"({summary['n_correct']}/{summary['n_total']}, "
        f"invalid={summary['n_invalid']})"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="PReMISE preference-fit audit")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--traits", help="Path to rubrics.json (canonical schema).")
    src.add_argument("--rubric", help="Path to rubrics.json (canonical schema).")
    parser.add_argument(
        "--samples",
        required=True,
        help="Normalized battle JSONL (prompt, response_a, response_b, winner).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model (default: config role 'judge').",
    )
    parser.add_argument(
        "--judge-name",
        default="judge",
        help="Short label for the judge (used in batch job names).",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch inference, use sequential Converse API.",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--role-arn", default=DEFAULT_BATCH_ROLE_ARN)
    parser.add_argument(
        "--method-label",
        default=None,
        help="Friendly method label recorded in the results file (e.g. 'premise_v0').",
    )
    parser.add_argument(
        "--dataset-source",
        default=None,
        help="Source preference dataset the rubric was discovered on.",
    )
    parser.add_argument(
        "--bench",
        default=None,
        help="Benchmark identifier when evaluating on a held-out bench "
        "(e.g. 'judgebench'). Mutually informational with --eval-split.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Split label for in-distribution eval (e.g. 'holdout').",
    )
    args = parser.parse_args()

    rubric_arg = args.traits or args.rubric

    run_pref_fit_experiment(
        rubric_arg,
        args.samples,
        args.output,
        max_samples=args.max_samples,
        judge_model=args.judge_model,
        judge_name=args.judge_name,
        no_batch=args.no_batch,
        region=args.region,
        s3_bucket=args.s3_bucket,
        role_arn=args.role_arn,
    )


if __name__ == "__main__":
    main()
