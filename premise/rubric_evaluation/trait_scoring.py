# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Pairwise Trait Scoring.

Scores candidate model responses against a baseline using policy-level rubrics
via LLM judges. Aggregates results into per-model per-trait win rates.

Default mode: Bedrock batch inference (server-side parallel).
Fallback: --no-batch for sequential Converse API calls.

Usage:
    # Score benchmark responses (auto-select median baseline)
    premise audit trait-score \
        --benchmark mbpp \
        --benchmark-dir <benchmarks>/mbpp/ \
        --traits <traits.json> \
        --baseline auto \
        --output results/trait_scoring/mbpp/

    # Score creative writing with an explicit baseline model
    premise audit trait-score \
        --benchmark creative_writing \
        --benchmark-dir <benchmarks>/creative_writing/ \
        --traits <traits.json> \
        --baseline "<baseline-model-id>" \
        --output results/trait_scoring/creative_writing/

    # Score pairwise battles (responses already paired)
    premise audit trait-score \
        --input-preference-data <preferences.jsonl> \
        --traits <traits.json> \
        --output results/trait_scoring/battles/

`--benchmark-dir` expects one subdirectory per model, each holding the response
file named by BENCHMARK_CONFIG for that benchmark (and optionally a
`results.json` carrying the benchmark's own metric, used by `--baseline auto`):

    <benchmark-dir>/
      <model-id>/
        responses.jsonl        # {prompt_id, <prompt_field>, <response_field>}
        results.json           # optional: {"<metric_key>": <float>}

Generating those responses is up to you — any harness will do. Use
`--input-preference-data`
instead if your responses are already paired.

    # Sequential mode (no batch)
    premise audit trait-score \
        --benchmark mbpp --benchmark-dir <benchmarks>/mbpp/ \
        --traits <traits.json> \
        --baseline auto --output results/trait_scoring/mbpp/ --no-batch

    # Resume a partial run (sequential only)
    premise audit trait-score \
        --benchmark mbpp --benchmark-dir <benchmarks>/mbpp/ \
        --traits <traits.json> \
        --baseline auto --output results/trait_scoring/mbpp/ --no-batch --resume
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

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
    load_traits,
    resolve_batch_settings,
    submit_batch_job,
    upload_to_s3,
    wait_for_batch_job,
)
from premise.utils.metrics import parse_score, score_to_outcome
from premise.utils.prompts import PAIRWISE_CATEGORY_JUDGE_PROMPT, PAIRWISE_JUDGE_PROMPT

# Benchmark response file config: (filename, prompt_field, response_field, metric_key)
BENCHMARK_CONFIG = {
    "creative_writing": ("responses.jsonl", "prompt", "response", "composite_score"),
    "mbpp": ("samples_mbpp.jsonl", "prompt", "response", "pass_at_1"),
    "humaneval": ("samples_humaneval.jsonl", "prompt", "response", "pass_at_1"),
    "strongreject": ("responses.jsonl", "forbidden_prompt", "response", "mean_score"),
    "arena_hard": ("responses_converted.jsonl", "prompt", "response", "win_rate"),
}


def benchmark_config(benchmark: str) -> tuple[str, str, str, str]:
    """Look up a benchmark's layout, naming the known set when it isn't one.

    A bare `BENCHMARK_CONFIG[benchmark]` gave a KeyError with no hint that this
    table is the thing to extend, so a typo and an unsupported benchmark looked
    identical.
    """
    try:
        return BENCHMARK_CONFIG[benchmark]
    except KeyError:
        raise ValueError(
            f"Unknown --benchmark '{benchmark}'. Known: "
            f"{', '.join(sorted(BENCHMARK_CONFIG))}.\n"
            "To score another benchmark, add a row to BENCHMARK_CONFIG in "
            f"{__name__}: benchmark -> (response filename, prompt field, "
            "response field, metric key in results.json). Or pass paired "
            "responses with --input-preference-data instead, which needs no benchmark "
            "layout at all."
        ) from None


def load_benchmark_responses(benchmark_dir: str, benchmark: str) -> dict:
    """Load responses from benchmark output directories.

    Returns: {model_id: [{prompt_id, prompt, response}, ...]}
    """
    filename, prompt_field, response_field, _ = benchmark_config(benchmark)
    base = Path(benchmark_dir)
    models = {}

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue

        response_file = model_dir / filename
        if not response_file.exists():
            print(f"  WARN: No {filename} in {model_dir.name}, skipping")
            continue

        responses = []
        with open(response_file) as f:
            if filename.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        responses.append(json.loads(line))
            else:
                responses = json.load(f)

        # Normalize to common schema
        samples = []
        for i, r in enumerate(responses):
            prompt_text = r.get(prompt_field, r.get("prompt", ""))
            response_text = r.get(response_field, "")
            prompt_id = r.get("task_id", f"{benchmark}_{i}")
            samples.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "response": response_text,
                }
            )

        # Derive model_id from results.json or directory name
        results_file = model_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            model_id = results.get("model", model_dir.name.replace("__", "/"))
        else:
            model_id = model_dir.name.replace("__", "/")

        models[model_id] = samples
        print(f"  Loaded {len(samples)} responses from {model_id}")

    return models


def load_target_metrics(benchmark_dir: str, benchmark: str) -> dict:
    """Load target metric for each model from results.json files.

    Returns: {model_id: float}
    """
    _, _, _, metric_key = benchmark_config(benchmark)
    base = Path(benchmark_dir)
    metrics = {}

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        results_file = model_dir / "results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            results = json.load(f)
        model_id = results.get("model", model_dir.name.replace("__", "/"))
        value = results.get(metric_key)
        if value is not None:
            metrics[model_id] = value

    return metrics


def select_baseline(metrics: dict, explicit: str | None = None) -> str:
    """Select baseline model. 'auto' picks the median by target metric."""
    if explicit and explicit != "auto":
        if explicit not in metrics:
            # Try fuzzy match
            for m in metrics:
                if explicit.lower() in m.lower():
                    print(f"  Baseline fuzzy match: {explicit} -> {m}")
                    return m
            print(f"  WARN: Baseline '{explicit}' not found, falling back to auto")
        else:
            return explicit

    # Auto: pick median model
    sorted_models = sorted(metrics.items(), key=lambda x: x[1])
    median_idx = len(sorted_models) // 2
    baseline_id, baseline_score = sorted_models[median_idx]
    print(f"  Auto-selected baseline: {baseline_id} (score={baseline_score})")
    return baseline_id


def load_existing_scores(output_dir: str) -> set:
    """Load already-scored (model, prompt_id, trait_id) tuples for resume."""
    scores_file = os.path.join(output_dir, "trait_scores.jsonl")
    done = set()
    if os.path.exists(scores_file):
        with open(scores_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["model"], r["prompt_id"], r["trait_id"]))
    return done


# ============================================================
# Batch Inference Mode
# ============================================================


def load_samples_file(path: str, max_samples=None):
    """Load a baseline-only samples JSONL file.

    Each line: {benchmark, prompt_id, prompt, baseline_name, baseline_response,
                applicable_traits: [rubric_id, ...]}.

    Returns (baseline_responses, applicable_by_prompt, baseline_name) where:
      - baseline_responses: {pid: {prompt_id, prompt, response}}
      - applicable_by_prompt: {pid: [rubric_id, ...]} (empty list if absent)
      - baseline_name: the model id of the baseline (taken from the first row)

    pid = "<benchmark>__<prompt_id>" composite to keep records unique across
    benchmarks (prompt_ids collide otherwise).
    """
    baseline_responses = {}
    applicable = {}
    baseline_name = None

    n = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = f"{r.get('benchmark', '')}__{r['prompt_id']}"
            prompt_text = r.get("prompt", "")
            base_resp = r.get("baseline_response", "")
            if not prompt_text or not base_resp:
                continue
            if baseline_name is None:
                baseline_name = r.get("baseline_name", "baseline")
            baseline_responses[pid] = {
                "prompt_id": pid,
                "prompt": prompt_text,
                "response": base_resp,
            }
            applicable[pid] = list(r.get("applicable_traits", []) or [])
            n += 1
            if max_samples and n >= max_samples:
                break

    return baseline_responses, applicable, (baseline_name or "baseline")


def load_candidate_file(name: str, path: str, valid_pids: set):
    """Load a per-candidate JSONL produced by build_inputs.

    Each line: {benchmark, prompt_id, response}. Returns
      [{prompt_id: pid, prompt: "", response: ...}, ...]
    pid = "<benchmark>__<prompt_id>" — only rows whose pid appears in
    valid_pids (i.e., present in the samples file) are kept.
    """
    out = []
    seen = set()
    skipped = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = f"{r.get('benchmark', '')}__{r['prompt_id']}"
            if pid not in valid_pids or pid in seen:
                skipped += 1
                continue
            resp = r.get("response", "")
            if not resp:
                skipped += 1
                continue
            seen.add(pid)
            out.append(
                {
                    "prompt_id": pid,
                    "prompt": "",  # filled at record-build time from baseline_responses
                    "response": resp,
                }
            )
    return out, skipped


def build_scoring_records(
    candidate_ids,
    all_responses,
    baseline_responses,
    traits,
    max_samples=None,
    applicable_by_prompt=None,
    group_by_category=False,
):
    """Build list of scoring records for batch or sequential processing.

    When applicable_by_prompt is provided, only the listed traits are scored
    for each prompt. When group_by_category is True, traits are grouped by
    `category` per (candidate, prompt) so a single judge call covers all
    traits in that category.

    Per-trait record (default):
      {candidate_id, prompt_id, prompt, candidate_response, baseline_response,
       trait_id, trait_description}
    Per-category record (group_by_category=True):
      {candidate_id, prompt_id, prompt, candidate_response, baseline_response,
       category, traits: [{id, description}, ...]}
    """
    trait_by_id = {t["id"]: t for t in traits}

    records = []
    for candidate_id in candidate_ids:
        samples = all_responses[candidate_id]
        if max_samples:
            samples = samples[:max_samples]

        for sample in samples:
            prompt_id = sample["prompt_id"]
            baseline_sample = baseline_responses.get(prompt_id)
            if not baseline_sample:
                continue

            # Resolve which traits to score for this prompt.
            if applicable_by_prompt is not None:
                ids = applicable_by_prompt.get(prompt_id, [])
                # Keep only ids that exist in the traits file (in case of drift).
                applicable_traits = [trait_by_id[i] for i in ids if i in trait_by_id]
            else:
                applicable_traits = traits
            if not applicable_traits:
                continue

            common = {
                "candidate_id": candidate_id,
                "prompt_id": prompt_id,
                # Prompt may live on the candidate sample (legacy paths) or
                # only on the baseline sample (baseline-only samples-file path).
                "prompt": sample.get("prompt") or baseline_sample.get("prompt", ""),
                "candidate_response": sample["response"],
                "baseline_response": baseline_sample["response"],
            }

            if group_by_category:
                by_cat = defaultdict(list)
                for t in applicable_traits:
                    by_cat[t.get("category", "uncategorized")].append(t)
                for cat, ts in by_cat.items():
                    records.append(
                        {
                            **common,
                            "category": cat,
                            "traits": [
                                {"id": t["id"], "description": t["description"]}
                                for t in ts
                            ],
                        }
                    )
            else:
                for t in applicable_traits:
                    records.append(
                        {
                            **common,
                            "trait_id": t["id"],
                            "trait_description": t["description"],
                        }
                    )
    return records


def parse_category_response(text: str, expected_ids: set) -> dict:
    """Parse the JSON response of PAIRWISE_CATEGORY_JUDGE_PROMPT.

    Returns {
      "traits": {trait_id: {candidate, baseline, candidate_reasoning, baseline_reasoning}},
      "overall": {candidate, baseline, candidate_reasoning, baseline_reasoning} | None,
    }
    Unknown ids and non-integer scores are silently dropped.
    """
    empty = {"traits": {}, "overall": None}
    if not text:
        return empty
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return empty
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return empty

    def _to_int(x):
        try:
            v = int(x)
        except (TypeError, ValueError):
            return None
        return v if 0 <= v <= 10 else None

    out_traits = {}
    raw = data.get("trait_scores", {})
    if isinstance(raw, dict):
        for rid, entry in raw.items():
            if rid not in expected_ids or not isinstance(entry, dict):
                continue
            cand = _to_int(entry.get("candidate_adherence"))
            base = _to_int(entry.get("baseline_adherence"))
            if cand is None and base is None:
                continue
            out_traits[rid] = {
                "candidate_adherence": cand,
                "baseline_adherence": base,
                "candidate_reasoning": entry.get("candidate_reasoning", ""),
                "baseline_reasoning": entry.get("baseline_reasoning", ""),
            }

    overall_raw = data.get("overall_score")
    overall = None
    if isinstance(overall_raw, dict):
        cand = _to_int(overall_raw.get("candidate_adherence"))
        base = _to_int(overall_raw.get("baseline_adherence"))
        if cand is not None or base is not None:
            overall = {
                "candidate_adherence": cand,
                "baseline_adherence": base,
                "candidate_reasoning": overall_raw.get("candidate_reasoning", ""),
                "baseline_reasoning": overall_raw.get("baseline_reasoning", ""),
            }

    return {"traits": out_traits, "overall": overall}


BATCH_MAX_RECORDS = 10_000  # Bedrock batch inference limit


def _format_traits_block(traits) -> str:
    """Format traits as a numbered list for the category judge prompt."""
    lines = []
    for i, t in enumerate(traits, 1):
        lines.append(f"{i}. **Trait ID:** {t['id']}")
        lines.append(f"   **Category:** {t.get('category', 'uncategorized')}")
        lines.append(f"   **Description:** {t['description']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_trait_score_template(traits) -> str:
    """Format the per-trait JSON skeleton inside the response_format block."""
    parts = []
    for t in traits:
        parts.append(
            f'    "{t["id"]}": {{"candidate_reasoning": "<brief explanation '
            f'for candidate>", "candidate_adherence": <0-10>, '
            f'"baseline_reasoning": "<brief explanation for baseline>", '
            f'"baseline_adherence": <0-10>}}'
        )
    return ",\n".join(parts)


def build_category_judge_prompt(rec: dict) -> str:
    """Format a category-mode scoring record into a judge prompt."""
    return PAIRWISE_CATEGORY_JUDGE_PROMPT.format(
        user_prompt=rec["prompt"],
        candidate_response=rec["candidate_response"],
        baseline_response=rec["baseline_response"],
        category=rec["category"],
        traits_block=_format_traits_block(rec["traits"]),
        trait_score_template=_format_trait_score_template(rec["traits"]),
    )


def prepare_batch_inputs(records, judge_model_id, output_dir):
    """Build JSONL batch input files for Bedrock batch inference.

    Auto-detects per-trait vs per-category records (presence of `traits` key
    selects the category prompt template). Splits into BATCH_MAX_RECORDS chunks.

    Returns: list of paths to batch input files.
    """
    chunks = [
        records[i : i + BATCH_MAX_RECORDS]
        for i in range(0, len(records), BATCH_MAX_RECORDS)
    ]
    print(f"  Splitting {len(records)} records into {len(chunks)} batch files")

    batch_paths = []
    expected_traits_by_record: dict[str, list[str]] = {}
    for chunk_idx, chunk in enumerate(chunks):
        suffix = f"_{chunk_idx}" if len(chunks) > 1 else ""
        batch_path = os.path.join(output_dir, f"batch_input{suffix}.jsonl")

        with open(batch_path, "w") as f:
            for i, rec in enumerate(chunk):
                global_idx = chunk_idx * BATCH_MAX_RECORDS + i

                if "traits" in rec:  # category mode
                    record_id = (
                        f"{global_idx}|{rec['candidate_id']}|{rec['prompt_id']}|"
                        f"cat:{rec['category']}"
                    )
                    expected_traits_by_record[record_id] = [
                        t["id"] for t in rec["traits"]
                    ]
                    prompt = build_category_judge_prompt(rec)
                    # Per-trait reasoning + scores + category overall.
                    max_tokens = 600 + 200 * len(rec["traits"])
                else:  # per-trait mode
                    record_id = (
                        f"{global_idx}|{rec['candidate_id']}|{rec['prompt_id']}|"
                        f"{rec['trait_id']}"
                    )
                    prompt = PAIRWISE_JUDGE_PROMPT.format(
                        trait_id=rec["trait_id"],
                        trait_description=rec["trait_description"],
                        user_prompt=rec["prompt"],
                        response_a=rec["candidate_response"],
                        response_b=rec["baseline_response"],
                    )
                    max_tokens = 500

                batch_record = {
                    "recordId": record_id,
                    "modelInput": get_batch_model_input(
                        judge_model_id, prompt, temperature=0.0, max_tokens=max_tokens
                    ),
                }
                f.write(json.dumps(batch_record) + "\n")

        print(f"  Chunk {chunk_idx}: {len(chunk)} records -> {batch_path}")
        batch_paths.append(batch_path)

    # Persist the expected-traits map so resume from existing batch outputs works.
    if expected_traits_by_record:
        with open(os.path.join(output_dir, "expected_traits_by_record.json"), "w") as f:
            json.dump(expected_traits_by_record, f)

    return batch_paths, expected_traits_by_record


def parse_batch_scoring_results(
    result_files, judge_model_id, expected_traits_by_record=None
):
    """Parse downloaded .jsonl.out files into trait score records.

    Per-trait records: returns {model, prompt_id, trait_id, score, judge_model}
    where `score` is the legacy 0-10 pairwise (Response A relative to B) score.

    Category records (recordId fourth field starts with "cat:"): one input
    record fans out to multiple per-trait results. Each result has both
    `candidate_adherence` and `baseline_adherence` (independent 0-10 scores)
    plus reasoning strings, and the legacy `score` field set to
    `candidate_adherence` for backward-compat with downstream aggregators.

    Args:
      expected_traits_by_record: optional dict mapping recordId -> list of
        trait ids the request asked about. Used to validate completeness;
        any missing trait gets a placeholder error result.
    """
    results = []
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
                    _, candidate_id, prompt_id, last = parts

                    model_output = rec.get("modelOutput", {})
                    text = extract_batch_model_response(model_output, judge_model_id)

                    if last.startswith("cat:"):
                        category = last[len("cat:") :]
                        expected_ids = set(
                            (expected_traits_by_record or {}).get(record_id, [])
                        )
                        parsed_obj = (
                            parse_category_response(text, expected_ids)
                            if text
                            else {"traits": {}, "overall": None}
                        )

                        for tid in expected_ids or parsed_obj["traits"].keys():
                            entry = parsed_obj["traits"].get(tid)
                            if entry is None:
                                results.append(
                                    {
                                        "model": candidate_id,
                                        "prompt_id": prompt_id,
                                        "trait_id": tid,
                                        "category": category,
                                        "score": None,
                                        "candidate_adherence": None,
                                        "baseline_adherence": None,
                                        "judge_model": judge_model_id,
                                        "error": "parse_failed",
                                    }
                                )
                                errors += 1
                            else:
                                results.append(
                                    {
                                        "model": candidate_id,
                                        "prompt_id": prompt_id,
                                        "trait_id": tid,
                                        "category": category,
                                        # Legacy `score` = candidate adherence so existing
                                        # downstream aggregators keep working.
                                        "score": entry["candidate_adherence"],
                                        "candidate_adherence": entry[
                                            "candidate_adherence"
                                        ],
                                        "baseline_adherence": entry[
                                            "baseline_adherence"
                                        ],
                                        "candidate_reasoning": entry[
                                            "candidate_reasoning"
                                        ],
                                        "baseline_reasoning": entry[
                                            "baseline_reasoning"
                                        ],
                                        "judge_model": judge_model_id,
                                    }
                                )
                                parsed += 1

                        # Optional: category overall score, kept as a separate row
                        if parsed_obj["overall"] is not None:
                            ov = parsed_obj["overall"]
                            results.append(
                                {
                                    "model": candidate_id,
                                    "prompt_id": prompt_id,
                                    "trait_id": f"__overall__:{category}",
                                    "category": category,
                                    "score": ov["candidate_adherence"],
                                    "candidate_adherence": ov["candidate_adherence"],
                                    "baseline_adherence": ov["baseline_adherence"],
                                    "candidate_reasoning": ov["candidate_reasoning"],
                                    "baseline_reasoning": ov["baseline_reasoning"],
                                    "judge_model": judge_model_id,
                                }
                            )
                    else:
                        trait_id = last
                        score = parse_score(text) if text else None
                        result = {
                            "model": candidate_id,
                            "prompt_id": prompt_id,
                            "trait_id": trait_id,
                            "score": score,
                            "judge_model": judge_model_id,
                        }
                        if score is None:
                            result["error"] = "parse_failed"
                            errors += 1
                        else:
                            parsed += 1
                        results.append(result)

                except (json.JSONDecodeError, ValueError):
                    errors += 1

    print(f"  Parsed {parsed} scores, {errors} errors")
    return results


def run_batch_scoring(records, judge_model_id, output_dir, s3_bucket, role_arn, region):
    """Run trait scoring via Bedrock batch inference.

    Auto-splits into chunks of BATCH_MAX_RECORDS (10K) per Bedrock limits.
    Submits all chunks, then waits for all to complete.

    Returns: list of score records.
    """
    # Check for already-downloaded batch results (resume after parse failure)
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
        # Recover the recordId -> [trait_ids] map persisted at submit time so
        # category-mode parsing can validate completeness.
        expected_path = os.path.join(output_dir, "expected_traits_by_record.json")
        expected_traits_by_record = {}
        if os.path.exists(expected_path):
            with open(expected_path) as f:
                expected_traits_by_record = json.load(f)
        return parse_batch_scoring_results(
            existing_results, judge_model_id, expected_traits_by_record
        )

    # Prepare batch input files (split into 10K-record chunks)
    batch_paths, expected_traits_by_record = prepare_batch_inputs(
        records, judge_model_id, output_dir
    )

    bedrock = get_bedrock_management_client(region)
    job_arns = []

    # Upload and submit each chunk
    for chunk_idx, batch_path in enumerate(batch_paths):
        file_hash = compute_file_hash(batch_path)[:12]
        suffix = f"-{chunk_idx}" if len(batch_paths) > 1 else ""
        job_name = f"premise-trait-scoring-{file_hash}{suffix}"
        job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]

        s3_key = f"premise/trait-scoring/{job_name}/input.jsonl"
        s3_output_prefix = f"s3://{s3_bucket}/premise/trait-scoring/{job_name}/output/"

        print(f"\n  --- Chunk {chunk_idx}/{len(batch_paths) - 1} ---")
        print(f"  Uploading to s3://{s3_bucket}/{s3_key}")
        s3_input_uri = upload_to_s3(batch_path, s3_bucket, s3_key, region)

        print(f"  Submitting batch job: {job_name}")
        job_arn = submit_batch_job(
            bedrock, s3_input_uri, s3_output_prefix, job_name, judge_model_id, role_arn
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
    return parse_batch_scoring_results(
        all_result_files, judge_model_id, expected_traits_by_record
    )


# ============================================================
# Sequential Mode (fallback)
# ============================================================


def run_sequential_scoring(records, judge_model_id, output_dir, region, resume=False):
    """Run trait scoring via sequential Converse API calls.

    Writes results incrementally to trait_scores.jsonl.
    """
    client = get_bedrock_client(region)

    # Resume support
    done = set()
    if resume:
        done = load_existing_scores(output_dir)
        if done:
            print(f"\nResuming: {len(done)} scores already recorded")

    total_calls = len(records) - len(done)
    print(f"  Total API calls: {total_calls}")

    scores_file = os.path.join(output_dir, "trait_scores.jsonl")
    call_count = 0
    errors = 0
    all_results = []

    with open(scores_file, "a") as scores_fh:
        for rec in records:
            call_count += 1
            if call_count % 50 == 0:
                print(f"  [{call_count}/{total_calls}] errors={errors}")

            if "traits" in rec:
                # Category mode — one judge call returns scores for multiple traits.
                expected_ids = {t["id"] for t in rec["traits"]}
                # Skip the whole call if every (candidate, prompt, trait) is already done.
                if all(
                    (rec["candidate_id"], rec["prompt_id"], tid) in done
                    for tid in expected_ids
                ):
                    continue

                prompt = build_category_judge_prompt(rec)
                max_tokens = 600 + 200 * len(rec["traits"])
                response = invoke_model(
                    client,
                    judge_model_id,
                    prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                parsed_obj = parse_category_response(response or "", expected_ids)

                for tid in expected_ids:
                    if (rec["candidate_id"], rec["prompt_id"], tid) in done:
                        continue
                    entry = parsed_obj["traits"].get(tid)
                    if entry is None:
                        record = {
                            "model": rec["candidate_id"],
                            "prompt_id": rec["prompt_id"],
                            "trait_id": tid,
                            "category": rec["category"],
                            "score": None,
                            "candidate_adherence": None,
                            "baseline_adherence": None,
                            "judge_model": judge_model_id,
                            "error": "parse_failed",
                        }
                        errors += 1
                    else:
                        record = {
                            "model": rec["candidate_id"],
                            "prompt_id": rec["prompt_id"],
                            "trait_id": tid,
                            "category": rec["category"],
                            "score": entry["candidate_adherence"],
                            "candidate_adherence": entry["candidate_adherence"],
                            "baseline_adherence": entry["baseline_adherence"],
                            "candidate_reasoning": entry["candidate_reasoning"],
                            "baseline_reasoning": entry["baseline_reasoning"],
                            "judge_model": judge_model_id,
                        }
                    scores_fh.write(json.dumps(record) + "\n")
                    all_results.append(record)

                if parsed_obj["overall"] is not None:
                    ov = parsed_obj["overall"]
                    rec_overall = {
                        "model": rec["candidate_id"],
                        "prompt_id": rec["prompt_id"],
                        "trait_id": f"__overall__:{rec['category']}",
                        "category": rec["category"],
                        "score": ov["candidate_adherence"],
                        "candidate_adherence": ov["candidate_adherence"],
                        "baseline_adherence": ov["baseline_adherence"],
                        "candidate_reasoning": ov["candidate_reasoning"],
                        "baseline_reasoning": ov["baseline_reasoning"],
                        "judge_model": judge_model_id,
                    }
                    scores_fh.write(json.dumps(rec_overall) + "\n")
                    all_results.append(rec_overall)
                scores_fh.flush()
                continue

            # Per-trait mode (legacy).
            key = (rec["candidate_id"], rec["prompt_id"], rec["trait_id"])
            if key in done:
                continue

            prompt = PAIRWISE_JUDGE_PROMPT.format(
                trait_id=rec["trait_id"],
                trait_description=rec["trait_description"],
                user_prompt=rec["prompt"],
                response_a=rec["candidate_response"],
                response_b=rec["baseline_response"],
            )
            response = invoke_model(client, judge_model_id, prompt, temperature=0.0)
            score = parse_score(response)
            record = {
                "model": rec["candidate_id"],
                "prompt_id": rec["prompt_id"],
                "trait_id": rec["trait_id"],
                "score": score,
                "judge_model": judge_model_id,
            }
            if score is None:
                record["error"] = "parse_failed"
                errors += 1
            scores_fh.write(json.dumps(record) + "\n")
            scores_fh.flush()
            all_results.append(record)

    print(f"\nSequential scoring complete: {call_count} calls, {errors} errors")
    return all_results


# ============================================================
# Main orchestration
# ============================================================


def run_trait_scoring(
    traits_file: str,
    output_dir: str,
    benchmark: str | None = None,
    benchmark_dir: str | None = None,
    arena_data: str | None = None,
    samples_file: str | None = None,
    candidates: list[str] | None = None,
    baseline: str = "auto",
    judge_model: str | None = None,
    max_samples: int | None = None,
    no_batch: bool = False,
    resume: bool = False,
    group_by_category: bool = False,
    region: str = DEFAULT_REGION,
    s3_bucket: str = DEFAULT_S3_BUCKET,
    role_arn: str = DEFAULT_BATCH_ROLE_ARN,
):
    os.makedirs(output_dir, exist_ok=True)
    from premise.providers import model_for_role

    judge_model_id = judge_model or model_for_role("judge")
    if not no_batch:
        s3_bucket, role_arn = resolve_batch_settings(s3_bucket, role_arn)
    traits = load_traits(traits_file)

    print(f"Loaded {len(traits)} traits")
    print(f"Judge model: {judge_model_id}")
    print(f"Batch inference: {'disabled' if no_batch else 'enabled'}")

    # Load responses
    if benchmark and benchmark_dir:
        try:
            benchmark_config(benchmark)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        print(f"\nLoading {benchmark} responses from {benchmark_dir}")
        all_responses = load_benchmark_responses(benchmark_dir, benchmark)
        target_metrics = load_target_metrics(benchmark_dir, benchmark)

        # Select baseline
        print(f"\nSelecting baseline from {len(target_metrics)} models with scores")
        baseline_id = select_baseline(target_metrics, baseline)

        if baseline_id not in all_responses:
            print(f"ERROR: Baseline model '{baseline_id}' has no responses")
            sys.exit(1)

        baseline_responses = {s["prompt_id"]: s for s in all_responses[baseline_id]}
        candidate_ids = [m for m in all_responses if m != baseline_id]
        applicable_by_prompt = None

    elif arena_data:
        # Arena mode: responses are already paired
        print(f"\nLoading arena battles from {arena_data}")
        from premise.utils import load_samples

        battles = load_samples(arena_data, max_samples)
        print(f"  Loaded {len(battles)} battles")

        all_responses = {}
        baseline_responses = {}
        baseline_id = "__arena_baseline__"

        for i, battle in enumerate(battles):
            prompt_id = battle.get("question_id", f"arena_{i}")
            model_a = battle.get("model_a", "model_a")
            prompt_text = ""
            if "conversation_a" in battle:
                conv = battle["conversation_a"]
                if isinstance(conv, list) and conv:
                    prompt_text = conv[0].get("content", "")[:2000]

            resp_a = ""
            if "conversation_a" in battle:
                conv = battle["conversation_a"]
                if isinstance(conv, list) and len(conv) > 1:
                    resp_a = conv[-1].get("content", "")

            resp_b = ""
            if "conversation_b" in battle:
                conv = battle["conversation_b"]
                if isinstance(conv, list) and len(conv) > 1:
                    resp_b = conv[-1].get("content", "")

            if model_a not in all_responses:
                all_responses[model_a] = []
            all_responses[model_a].append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "response": resp_a,
                }
            )
            baseline_responses[prompt_id] = {
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "response": resp_b,
            }

        candidate_ids = list(all_responses.keys())
        target_metrics = {}
        applicable_by_prompt = None

    elif samples_file:
        # Baseline-only samples file + per-candidate JSONLs.
        print(f"\nLoading samples from {samples_file}")
        baseline_responses, applicable_by_prompt, baseline_id = load_samples_file(
            samples_file, max_samples
        )
        target_metrics = {}
        print(f"  baseline: {baseline_id}")
        print(f"  baseline_responses: {len(baseline_responses)}")
        n_with_app = sum(1 for v in applicable_by_prompt.values() if v)
        print(
            f"  prompts with applicable_traits: {n_with_app}/{len(applicable_by_prompt)}"
        )

        if not candidates:
            print("ERROR: --samples-file requires at least one --candidate name=path")
            sys.exit(1)

        all_responses = {}
        valid_pids = set(baseline_responses)
        for spec in candidates:
            if "=" not in spec:
                print(f"ERROR: --candidate must be name=path, got {spec!r}")
                sys.exit(1)
            name, path = spec.split("=", 1)
            samples, skipped = load_candidate_file(name, path, valid_pids)
            if not samples:
                print(
                    f"  WARN: candidate {name!r} produced 0 rows after join, skipping"
                )
                continue
            all_responses[name] = samples
            print(f"  candidate {name:<48} {len(samples):>5} rows  ({skipped} skipped)")

        if not all_responses:
            print("ERROR: no candidates produced any rows after the join")
            sys.exit(1)
        candidate_ids = list(all_responses.keys())

    else:
        print(
            "ERROR: Must provide --benchmark/--benchmark-dir, "
            "--input-preference-data, or --samples-file"
        )
        sys.exit(1)

    # Build scoring records
    records = build_scoring_records(
        candidate_ids,
        all_responses,
        baseline_responses,
        traits,
        max_samples,
        applicable_by_prompt=applicable_by_prompt,
        group_by_category=group_by_category,
    )
    print(f"\nScoring {len(candidate_ids)} candidates x {len(traits)} traits")
    print(f"Baseline: {baseline_id}")
    print(f"Total records: {len(records)}")

    # Run scoring
    if no_batch:
        score_results = run_sequential_scoring(
            records, judge_model_id, output_dir, region, resume=resume
        )
    else:
        score_results = run_batch_scoring(
            records, judge_model_id, output_dir, s3_bucket, role_arn, region
        )
        # Write results to trait_scores.jsonl (batch mode writes after completion)
        scores_file = os.path.join(output_dir, "trait_scores.jsonl")
        with open(scores_file, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in score_results)
        print(f"Raw scores: {scores_file}")

    # Aggregate win rates
    aggregate_win_rates(output_dir, baseline_id, target_metrics, traits)


def aggregate_win_rates(
    output_dir: str, baseline_id: str, target_metrics: dict, traits: list
):
    """Read trait_scores.jsonl and compute per-model per-trait win rates."""
    scores_file = os.path.join(output_dir, "trait_scores.jsonl")

    # model -> trait -> {wins, losses, ties, total}
    model_stats = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "total": 0})
    )

    with open(scores_file) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            score = r["score"]
            if score is None:
                continue

            model = r["model"]
            trait_id = r["trait_id"]
            outcome = score_to_outcome(score)

            model_stats[model][trait_id]["total"] += 1
            if outcome == 2:
                model_stats[model][trait_id]["wins"] += 1
            elif outcome == 0:
                model_stats[model][trait_id]["losses"] += 1
            else:
                model_stats[model][trait_id]["ties"] += 1

    # Compute win rates
    win_rates = {}
    for model_id, trait_stats in sorted(model_stats.items()):
        win_rates[model_id] = {}
        for trait_id, s in sorted(trait_stats.items()):
            total = s["total"]
            if total > 0:
                wr = (s["wins"] + 0.5 * s["ties"]) / total
            else:
                wr = 0.5
            win_rates[model_id][trait_id] = {
                "win_rate": round(wr, 4),
                "wins": s["wins"],
                "losses": s["losses"],
                "ties": s["ties"],
                "total": total,
            }

    output = {
        "baseline": baseline_id,
        "target_metrics": target_metrics,
        "trait_ids": [t["id"] for t in traits],
        "models": win_rates,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    win_rates_file = os.path.join(output_dir, "win_rates.json")
    with open(win_rates_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Win rates: {win_rates_file}")

    # Summary
    summary = {
        "baseline": baseline_id,
        "n_candidates": len(win_rates),
        "n_traits": len(traits),
        "target_metrics": target_metrics,
        "per_model_mean_win_rate": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }
    for model_id, trait_wr in win_rates.items():
        rates = [v["win_rate"] for v in trait_wr.values()]
        summary["per_model_mean_win_rate"][model_id] = (
            round(sum(rates) / len(rates), 4) if rates else 0.5
        )

    summary_file = os.path.join(output_dir, "scoring_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_file}")

    # Print table
    print(f"\n{'Model':<45} {'Mean WR':>8}  {'Target':>8}")
    print("-" * 65)
    for model_id in sorted(win_rates.keys()):
        mean_wr = summary["per_model_mean_win_rate"][model_id]
        target = target_metrics.get(model_id, "N/A")
        target_str = f"{target:.3f}" if isinstance(target, float) else str(target)
        print(f"  {model_id:<43} {mean_wr:>8.3f}  {target_str:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PReMISE pairwise trait scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--traits", required=True, help="Traits JSON file (rubric corpus)"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--benchmark",
        help="Benchmark whose response layout to read: "
        + ", ".join(sorted(BENCHMARK_CONFIG))
        + " (see BENCHMARK_CONFIG to add one)",
    )
    parser.add_argument(
        "--benchmark-dir", help="Benchmark directory with per-model responses"
    )
    parser.add_argument(
        "--input-preference-data",
        dest="arena_data",
        help="Pairwise preference JSONL (alternative to benchmark mode)",
    )
    parser.add_argument("--arena-data", dest="arena_data", help=argparse.SUPPRESS)
    parser.add_argument(
        "--samples-file",
        help="Baseline samples JSONL: per-prompt {benchmark, prompt_id, "
        "prompt, baseline_name, baseline_response, applicable_traits}. "
        "Requires --candidate.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Candidate JSONL (repeatable). Each PATH has rows "
        "{benchmark, prompt_id, response}. Joined with --samples-file "
        "on (benchmark, prompt_id).",
    )
    parser.add_argument(
        "--group-by-category",
        action="store_true",
        help="Group applicable traits by category and judge each "
        "(candidate, prompt, category) bundle in a single LLM call. "
        "Reduces request count by ~6-8x.",
    )
    parser.add_argument(
        "--baseline",
        default="auto",
        help="Baseline model name, or 'auto' for median (default: auto)",
    )
    parser.add_argument(
        "--judge-model",
        help="Judge model ID (default: config role 'judge')",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Max prompts to score per model pair"
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch inference, use sequential Converse API",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-scored tuples (sequential mode only)",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--s3-bucket",
        default=DEFAULT_S3_BUCKET,
        help="S3 bucket for batch inference (default: config aws.s3_bucket)",
    )
    parser.add_argument(
        "--role-arn",
        default=DEFAULT_BATCH_ROLE_ARN,
        help="IAM role ARN for Bedrock batch jobs",
    )
    args = parser.parse_args()

    run_trait_scoring(
        traits_file=args.traits,
        output_dir=args.output,
        benchmark=args.benchmark,
        benchmark_dir=args.benchmark_dir,
        arena_data=args.arena_data,
        samples_file=args.samples_file,
        candidates=args.candidate,
        baseline=args.baseline,
        judge_model=args.judge_model,
        max_samples=args.max_samples,
        no_batch=args.no_batch,
        resume=args.resume,
        group_by_category=args.group_by_category,
        region=args.region,
        s3_bucket=args.s3_bucket,
        role_arn=args.role_arn,
    )
