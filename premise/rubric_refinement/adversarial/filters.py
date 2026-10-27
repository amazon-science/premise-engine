# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Adversarial repair, Stages 3-4: drop variants, then measure the survivors.

Nothing here writes rubric text. `repair.py` (Stages 1-2) produced every variant
and its predicted metrics; these two stages decide which are worth paying for and
then find out what they actually score:

    3. prune_topn   — hard filters on the predicted metrics + top-N per trait
    4. batch_eval   — iter-0 batch eval: real consistency + adversarial v0

Stage 3 is cheap and predicted; Stage 4 is the expensive real thing, which is why
the filter comes first. Stage 4's results are what `pareto.py` selects on.

`resume_main` is crash recovery for Stage 4, not a stage of its own: a batch
consistency run submits ~21 chunks over hours, and a crash after submission
leaves completed Bedrock jobs whose results were never aggregated. It re-reads the
ARNs out of `consistency.log` and finishes the aggregation.

The predicted-metric filters consult vfr/alpha/flip/meta only. The paper's
*constrained* variant additionally bounds preference-fit loss against v0; that is
not implemented — see the package `__init__` docstring.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from premise.rubric_refinement.tree import (
    ROOT,
    _run_tool,
    atomic_save,
    load_tree,
    now_iso,
)

# ---------------------------------------------------------------------------
# Stage 3: hard filters + top-N per trait
# ---------------------------------------------------------------------------


def prune_topn_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.15,
        help="max predicted α drop below v0 tolerated (default 0.15)",
    )
    ap.add_argument("--flip-threshold", type=float, default=20.0)
    ap.add_argument("--meta-drop", type=int, default=1)
    ap.add_argument("--top-n-per-trait", type=int, default=20)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    tree_path = os.path.join(args.root, "tree.json")
    tree = load_tree(tree_path)

    # v0 reference metrics per trait (real if available, else the predictor's own v0 prediction if we had one)
    v0s = {n["trait_id"]: n for n in tree["nodes"] if n.get("depth", 0) == 0}

    if args.reset:
        for n in tree["nodes"]:
            if n.get("status") in ("prune_survivor", "pruned_by_predictor"):
                n["status"] = "predicted"
                n.pop("rejection_reason", None)
                n["updated_at"] = now_iso()

    # Stage 1: hard filters
    per_trait_nodes = defaultdict(list)
    hard_prune = 0
    hard_keep = 0

    for n in tree["nodes"]:
        if n.get("status") not in ("predicted", "prune_survivor"):
            continue
        pm = n.get("predicted_metrics", {})
        conf = n.get("prediction_confidence", {})
        v0 = v0s.get(n["trait_id"])
        if not v0:
            continue
        v0m = v0.get("metrics", {})
        # If v0 has no real metrics, we can still use predicted VFR improvement as the first filter
        v0_vfr = v0m.get("vfr_mean") or v0m.get("vfr")
        v0_meta = v0m.get("meta")
        v0_alpha = v0m.get("alpha")
        v0_flip = v0m.get("flip")

        reasons = []
        pv = pm.get("vfr")
        if pv is None:
            reasons.append("no predicted vfr")
        elif v0_vfr is not None and pv >= v0_vfr:
            reasons.append(f"pred_vfr={pv} not < v0={v0_vfr}")

        if (
            v0_alpha is not None
            and pm.get("alpha") is not None
            and pm["alpha"] < v0_alpha - args.alpha_threshold
        ):
            reasons.append(f"pred_α < v0 - {args.alpha_threshold}")

        if (
            v0_meta is not None
            and pm.get("meta") is not None
            and pm["meta"] < v0_meta - args.meta_drop
        ):
            reasons.append(f"pred_meta drop > {args.meta_drop}")

        if (
            v0_flip is not None
            and pm.get("flip") is not None
            and pm["flip"] > v0_flip + args.flip_threshold
        ):
            reasons.append(f"pred_flip > v0 + {args.flip_threshold}")

        verdict = n.get("predictor_gate_verdict", "")
        metric_map = {
            "likely_fail_alpha": "alpha",
            "likely_fail_flip": "flip",
            "likely_fail_vfr": "vfr",
            "likely_fail_meta": "meta",
        }
        m = metric_map.get(verdict)
        if m and conf.get(m) == "high":
            reasons.append(f"verdict={verdict} high conf")

        if reasons:
            n["status"] = "pruned_by_predictor"
            n["rejection_reason"] = " & ".join(reasons)
            n["decision_source"] = "predictor"
            n["updated_at"] = now_iso()
            hard_prune += 1
        else:
            per_trait_nodes[n["trait_id"]].append(n)
            hard_keep += 1

    # Stage 2: top-N per trait (pooled across depths)
    topn_drop = 0
    final_keep = 0
    for nodes in per_trait_nodes.values():
        nodes.sort(
            key=lambda n: (
                n.get("predicted_metrics", {}).get("vfr", 1.0),
                -(n.get("predicted_metrics", {}).get("meta") or 0),
                -(n.get("predicted_metrics", {}).get("alpha") or 0),
                n.get("predicted_metrics", {}).get("flip", 100),
            )
        )
        survivors = nodes[: args.top_n_per_trait]
        dropped = nodes[args.top_n_per_trait :]
        for n in survivors:
            n["status"] = "prune_survivor"
            n["decision_source"] = "predictor"
            n.pop("rejection_reason", None)
            n["updated_at"] = now_iso()
            final_keep += 1
        for n in dropped:
            n["status"] = "pruned_by_predictor"
            n["rejection_reason"] = f"topN_drop (rank > {args.top_n_per_trait})"
            n["decision_source"] = "predictor"
            n["updated_at"] = now_iso()
            topn_drop += 1

    atomic_save(tree, tree_path)

    print(f"Hard filters: {hard_keep} kept, {hard_prune} pruned")
    print(
        f"Top-{args.top_n_per_trait} per trait: {final_keep} survivors, {topn_drop} additionally dropped"
    )

    # Per-trait summary
    by = Counter(
        (n["trait_id"], n["status"])
        for n in tree["nodes"]
        if n["status"] in ("prune_survivor", "pruned_by_predictor")
    )
    traits_with_survivors = sorted(
        {t for (t, s), _ in by.items() if s == "prune_survivor"}
    )
    print(f"\n{len(traits_with_survivors)} traits have survivors")
    for t in traits_with_survivors[:10]:
        s = by.get((t, "prune_survivor"), 0)
        p = by.get((t, "pruned_by_predictor"), 0)
        print(f"  {t}: survivors={s}  pruned={p}")


# ---------------------------------------------------------------------------
# Stage 4: held-out IRA, consistency, and adversarial evaluation
# ---------------------------------------------------------------------------


def build_merged_traits(tree, target_statuses, out_path):
    """Write a single traits JSON with all nodes-of-interest as rubrics (unique ids)."""
    rubrics = []
    id_map = {}  # rubric_id → node_id (different for v0 split)
    for n in tree["nodes"]:
        if n.get("status") not in target_statuses:
            continue
        rp = n.get("rubric_path")
        if not rp or not os.path.exists(rp):
            continue
        try:
            d = json.loads(Path(rp).read_text())
        except Exception:
            continue
        orig_rubric = d["rubrics"][0]
        # Use node_id as the rubric id in the merged file so results map back
        rubrics.append(
            {
                "id": n[
                    "id"
                ],  # e.g. "exception_handling__grounded+parsimony" or "exception_handling__v0"
                "corpus": orig_rubric.get("corpus", "discovered"),
                "category": orig_rubric.get("category", "refined"),
                "description": orig_rubric["description"],
            }
        )
        id_map[n["id"]] = n["id"]
    out = {"schema_version": "1.0.0", "rubrics": rubrics}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return len(rubrics)


def resolve_eval_samples(root, dataset=None, samples=None, adv_samples=None):
    """Locate the two iter-1 eval slices, preferring explicit paths over the manifest.

    `refine adversarial seed` recorded what it wrote under
    `datasets.<dataset>.samples_eval_{pairwise,adv}`, so read them back rather than
    reconstructing filenames. A root seeded with several datasets has several
    candidates and `--dataset` picks one; with exactly one there is nothing to pick.
    """
    if samples and adv_samples:
        return samples, adv_samples

    manifest_path = os.path.join(root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found — run `refine adversarial seed` first, or "
            "pass --samples and --adv-samples explicitly."
        )
    datasets = json.loads(Path(manifest_path).read_text()).get("datasets") or {}
    if not datasets:
        raise ValueError(f"{manifest_path} lists no datasets.")

    if dataset is None:
        if len(datasets) > 1:
            raise ValueError(
                f"{manifest_path} has {len(datasets)} datasets "
                f"({', '.join(sorted(datasets))}); pass --dataset to say which "
                "one's eval slices to use."
            )
        dataset = next(iter(datasets))
    elif dataset not in datasets:
        raise ValueError(
            f"Dataset '{dataset}' is not in {manifest_path} "
            f"(has: {', '.join(sorted(datasets))})"
        )

    entry = datasets[dataset]
    resolved = samples or entry.get("samples_eval_pairwise")
    resolved_adv = adv_samples or entry.get("samples_eval_adv")
    if not resolved or not resolved_adv:
        raise ValueError(
            f"{manifest_path}: dataset '{dataset}' has no samples_eval_pairwise/"
            "samples_eval_adv; it was seeded by an older version — pass --samples "
            "and --adv-samples explicitly."
        )
    return resolved, resolved_adv


def batch_eval_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--n-attempts", type=int, default=50)
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument(
        "--dataset",
        default=None,
        help="Which seeded dataset's eval slices to score on. Only needed when the "
        "root holds more than one.",
    )
    ap.add_argument(
        "--samples",
        default=None,
        help="Pairwise battles JSONL for the consistency run (default: from manifest)",
    )
    ap.add_argument(
        "--adv-samples",
        default=None,
        help="Prompts JSONL for the adversarial run (default: from manifest)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build merged traits + compute request counts, do NOT submit",
    )
    args = ap.parse_args()

    tree_path = os.path.join(args.root, "tree.json")
    tree = load_tree(tree_path)

    work_dir = os.path.join(args.root, f"batch_eval_iter{args.iter}")
    os.makedirs(work_dir, exist_ok=True)
    merged_traits_path = os.path.join(work_dir, "merged_traits.json")
    n = build_merged_traits(tree, {"prune_survivor", "pending"}, merged_traits_path)
    print(f"Merged {n} nodes into {merged_traits_path}")
    # Dry-run stats
    total_consistency = n * args.n_samples * 5
    total_ira = n * args.n_samples * 3
    total_attacker = n * args.n_attempts
    print("\nRequest volume estimates:")
    print(f"  IRA (3-judge panel):         {total_ira:,} calls")
    print(f"  Consistency (DeepSeek V3): {total_consistency:,} calls")
    print(f"  Adv-attacker (DeepSeek R1): {total_attacker:,} calls")
    print(f"  Adv-judge (DeepSeek V3):    {total_attacker:,} calls")
    print(
        f"  Adv-verifier (Mistral):     {total_attacker:,} calls (only high-scoring survive)"
    )
    print(
        f"  TOTAL:                      "
        f"~{(total_ira + total_consistency + 3 * total_attacker):,} calls"
    )
    if args.dry_run:
        return

    samples_path, adv_samples_path = resolve_eval_samples(
        args.root,
        dataset=args.dataset,
        samples=args.samples,
        adv_samples=args.adv_samples,
    )
    print(
        f"\nEval slices:\n  pairwise: {samples_path}\n  adversarial: {adv_samples_path}"
    )
    consistency_out = os.path.join(work_dir, "consistency")
    ira_out = os.path.join(work_dir, "ira")
    adv_out = os.path.join(work_dir, "adversarial")
    os.makedirs(consistency_out, exist_ok=True)
    os.makedirs(ira_out, exist_ok=True)
    os.makedirs(adv_out, exist_ok=True)

    # --- Launch reliability and adversarial jobs concurrently ---
    ira_cmd = [
        sys.executable,
        "-m",
        "premise.rubric_evaluation.ira",
        "--traits",
        merged_traits_path,
        "--samples",
        samples_path,
        "--output",
        ira_out,
        "--max-samples",
        str(args.n_samples),
    ]
    consistency_cmd = [
        sys.executable,
        "-m",
        "premise.rubric_evaluation.consistency",
        "--traits",
        merged_traits_path,
        "--samples",
        samples_path,
        "--output",
        consistency_out,
        "--max-samples",
        str(args.n_samples),
        "--n-runs",
        "5",
        "--temperature",
        "0.7",
    ]
    adv_cmd = [
        sys.executable,
        "-m",
        "premise.rubric_evaluation.adversarial",
        "--traits",
        merged_traits_path,
        "--samples",
        adv_samples_path,
        "--output",
        adv_out,
        "--n-attempts",
        str(args.n_attempts),
        "--run-id",
        f"refine-corpus-iter{args.iter}",
        "--verifier-version",
        "v0",
    ]

    print("\nLaunching three batch jobs concurrently:")
    print(f"  IRA:         {' '.join(ira_cmd[1:])}")
    print(f"  CONSISTENCY: {' '.join(consistency_cmd[1:])}")
    print(f"  ADVERSARIAL: {' '.join(adv_cmd[1:])}")
    print(
        f"\nLogs: {work_dir}/ira.log  {work_dir}/consistency.log  "
        f"{work_dir}/adversarial.log"
    )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {
            ex.submit(_run_tool, ira_cmd, os.path.join(work_dir, "ira.log")): "ira",
            ex.submit(
                _run_tool, consistency_cmd, os.path.join(work_dir, "consistency.log")
            ): "consistency",
            ex.submit(
                _run_tool, adv_cmd, os.path.join(work_dir, "adversarial.log")
            ): "adversarial",
        }
        for fut in as_completed(futs):
            name = futs[fut]
            rc = fut.result()
            print(f"{name} completed rc={rc} elapsed={(time.time() - t0) / 60:.1f} min")

    print(
        f"\nAll batch jobs done in {(time.time() - t0) / 60:.1f} min. Now aggregating into tree.json."
    )

    # --- Aggregate ---
    aggregate_ira_to_tree(tree, ira_out)
    aggregate_consistency_to_tree(tree, consistency_out, args.iter)
    aggregate_adv_to_tree(tree, adv_out, args.iter)
    atomic_save(tree, tree_path)

    print("Tree updated. Done.")


def aggregate_ira_to_tree(tree, ira_dir):
    scores_path = os.path.join(ira_dir, "ira_scores.json")
    if not os.path.exists(scores_path):
        print(f"WARN: {scores_path} missing")
        return
    data = json.loads(Path(scores_path).read_text())
    n_updated = 0
    for node_id, result in data.items():
        node = next((item for item in tree["nodes"] if item["id"] == node_id), None)
        if node is None:
            continue
        alpha = result.get("krippendorff_alpha")
        if alpha is not None:
            node.setdefault("metrics", {})["alpha"] = alpha
            node["updated_at"] = now_iso()
            n_updated += 1
    print(f"  ira: updated {n_updated} nodes")


def aggregate_consistency_to_tree(tree, consistency_dir, iter_n):
    scores_path = os.path.join(consistency_dir, "consistency_scores.json")
    if not os.path.exists(scores_path):
        print(f"WARN: {scores_path} missing")
        return
    data = json.loads(Path(scores_path).read_text())
    per_node = data.get("traits") or data
    n_updated = 0
    for nid, r in per_node.items():
        node = next((x for x in tree["nodes"] if x["id"] == nid), None)
        if not node:
            continue
        flip = (
            r.get("flip_pct")
            or r.get("flip_rate_percent")
            or r.get("mean_flip_rate")
            or r.get("win_flip_rate")
        )
        node.setdefault("metrics", {})
        if flip is not None:
            node["metrics"]["flip"] = flip
        node["batch_eval_iter"] = iter_n
        node["updated_at"] = now_iso()
        n_updated += 1
    print(f"  consistency: updated {n_updated} nodes")


def aggregate_adv_to_tree(tree, adv_dir, iter_n):
    res_path = os.path.join(adv_dir, "adversarial_gameability_results.json")
    if not os.path.exists(res_path):
        print(f"WARN: {res_path} missing")
        return
    data = json.loads(Path(res_path).read_text())
    n_updated = 0
    # Each key is a node_id (we set rubric ids = node ids in merged_traits)
    for nid, r in data.items():
        if not isinstance(r, dict):
            continue
        node = next((x for x in tree["nodes"] if x["id"] == nid), None)
        if not node:
            continue
        vfr = r.get("verified_fool_rate") or r.get("vfr_percent")
        if vfr is not None:
            node.setdefault("metrics", {})
            node["metrics"]["vfr"] = vfr
            node["metrics"]["vfr_mean"] = vfr
        n_updated += 1
    # Now classify each node with real consistency+VFR metrics
    v0s = {n["trait_id"]: n for n in tree["nodes"] if n.get("depth", 0) == 0}
    for nid, r in data.items():
        node = next((x for x in tree["nodes"] if x["id"] == nid), None)
        if not node:
            continue
        if node.get("depth", 0) == 0:
            node["status"] = "stable_accept"
            node["decision_source"] = "measured"
            continue
        v0 = v0s.get(node["trait_id"])
        if not v0:
            continue
        v0m = v0.get("metrics", {})
        m = node.get("metrics", {})
        # Classify: strict if VFR below v0 AND flip <= v0+5pp
        v0_vfr = v0m.get("vfr_mean") or v0m.get("vfr")
        v0_flip = v0m.get("flip")
        vfr = m.get("vfr")
        flip = m.get("flip")
        reasons = []
        if v0_vfr is not None and vfr is not None and vfr >= v0_vfr + 0.05:
            reasons.append(f"VFR {vfr:.3f} >= v0 {v0_vfr:.3f} + 0.05")
        if v0_flip is not None and flip is not None and flip > v0_flip + 10:
            reasons.append(f"flip {flip}% > v0 {v0_flip}% + 10")
        if reasons:
            node["status"] = "rejected_quality"
            node["rejection_reason"] = " & ".join(reasons)
        else:
            if (
                v0_vfr is not None
                and vfr is not None
                and vfr <= v0_vfr
                and v0_flip is not None
                and flip is not None
                and flip <= v0_flip + 5
            ):
                node["status"] = "stable_accept"
            else:
                node["status"] = "tentative_accept"
        node["decision_source"] = "measured"
        node["updated_at"] = now_iso()
    print(f"  adversarial: updated {n_updated} nodes")


# ---------------------------------------------------------------------------
# Resume consistency aggregation after subprocess crash
# ---------------------------------------------------------------------------

from premise.utils import (
    DEFAULT_BATCH_ROLE_ARN,
    DEFAULT_S3_BUCKET,
    compute_file_hash,
    download_batch_results,
    extract_batch_model_response,
    get_bedrock_management_client,
    resolve_batch_settings,
    submit_batch_job,
    upload_to_s3,
)
from premise.utils.metrics import parse_score, score_to_outcome


# Parse arns from the consistency log
def parse_submitted_arns(log_path):
    """Extract (chunk_idx, arn) pairs from consistency.log."""
    out = {}
    with open(log_path) as f:
        for line in f:
            m = re.search(r"\[chunk (\d+)\] submitted → (arn:[^\s]+)", line)
            if m:
                out[int(m.group(1))] = m.group(2)
    return out


def parse_s3_prefix_from_arn(bedrock, arn):
    """Query bedrock for the job's outputDataConfig."""
    resp = bedrock.get_model_invocation_job(jobIdentifier=arn)
    s3_uri = resp["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
    # s3://bucket/prefix/  -> extract prefix
    m = re.match(r"s3://([^/]+)/(.+)", s3_uri)
    return m.group(1), m.group(2).rstrip("/")


def parse_consistency_outcomes(result_files, model_id):
    """Returns {node_id: {sample_idx: {run_idx: outcome}}}."""
    data = defaultdict(lambda: defaultdict(dict))
    parsed = 0
    errors = 0
    for fpath in result_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    record_id = rec.get("recordId", "")
                    # `premise audit consistency` schema:
                    # "{global_idx}|{trait_id}|{sample_idx}|{run_idx}"
                    parts = record_id.split("|", 3)
                    if len(parts) != 4:
                        errors += 1
                        continue
                    _, nid, s_idx, r_idx = parts
                    s_idx = int(s_idx)
                    r_idx = int(r_idx)
                    mo = rec.get("modelOutput", {})
                    text = extract_batch_model_response(mo, model_id)
                    score = parse_score(text) if text else None
                    outcome = score_to_outcome(score)
                    data[nid][s_idx][r_idx] = outcome
                    parsed += 1
                except Exception:
                    errors += 1
    print(f"  Parsed {parsed} consistency outcomes, {errors} errors")
    return data


def compute_flip_per_node(outcomes):
    """Mean flip% per node: fraction of samples where the 5 runs disagree."""
    flips = {}
    for nid, samples in outcomes.items():
        n_flipped = 0
        n_total = 0
        for runs in samples.values():
            outs = [v for v in runs.values() if v is not None]
            if len(outs) < 2:
                continue
            n_total += 1
            if len(set(outs)) > 1:
                n_flipped += 1
        flips[nid] = round(100 * n_flipped / max(1, n_total), 1) if n_total else None
    return flips


def resume_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument(
        "--log",
        default=None,
        help="Stage 4 consistency log to re-read "
        "(default: <root>/batch_eval_iter0/consistency.log)",
    )
    ap.add_argument(
        "--local-dir",
        default=None,
        help="where recovered batch outputs land (default: "
        "<root>/batch_eval_iter0/consistency/batch_output_recovered)",
    )
    ap.add_argument(
        "--resubmit-missing",
        action="store_true",
        help="resubmit chunk(s) that crashed during upload",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="skip S3 download (use already-downloaded files)",
    )
    args = ap.parse_args()

    # Derived from --root unless overridden, so pointing the resume at another
    # corpus root does not silently keep reading the default root's log.
    if args.log is None:
        args.log = os.path.join(args.root, "batch_eval_iter0", "consistency.log")
    if args.local_dir is None:
        args.local_dir = os.path.join(
            args.root, "batch_eval_iter0", "consistency", "batch_output_recovered"
        )

    os.makedirs(args.local_dir, exist_ok=True)
    bedrock = get_bedrock_management_client()

    # Identify completed chunks. The run's chunk count is whatever Stage 4
    # wrote as batch_input_<i>.jsonl, so read it from disk rather than
    # assuming a fixed number.
    submitted = parse_submitted_arns(args.log)
    print(f"Found {len(submitted)} submitted chunks in log: {sorted(submitted.keys())}")
    chunk_dir = os.path.join(args.root, "batch_eval_iter0", "consistency")
    all_chunks = sorted(
        int(m.group(1))
        for fn in os.listdir(chunk_dir)
        if (m := re.fullmatch(r"batch_input_(\d+)\.jsonl", fn))
    )
    missing = [i for i in all_chunks if i not in submitted]
    print(f"Missing chunks: {missing} (of {len(all_chunks)} on disk)")

    # Download results for each completed ARN
    all_result_files = []
    if not args.skip_download:
        for chunk_idx, arn in sorted(submitted.items()):
            try:
                bucket, prefix = parse_s3_prefix_from_arn(bedrock, arn)
            except Exception as e:
                print(f"  chunk {chunk_idx}: get_job_info failed: {e}")
                continue
            chunk_dir = os.path.join(args.local_dir, f"chunk_{chunk_idx}")
            files = download_batch_results(bucket, prefix, chunk_dir)
            print(
                f"  chunk {chunk_idx}: downloaded {len(files)} files from s3://{bucket}/{prefix}"
            )
            all_result_files.extend(files)
    else:
        for root, dirs, files in os.walk(args.local_dir):
            for fn in files:
                if fn.endswith(".jsonl.out"):
                    all_result_files.append(os.path.join(root, fn))

    print(f"\nTotal result files: {len(all_result_files)}")

    # Parse
    from premise.providers import model_for_role

    judge_model = model_for_role("judge")
    outcomes = parse_consistency_outcomes(all_result_files, judge_model)
    print(f"Nodes with consistency data: {len(outcomes)}")

    # Aggregate flip%
    flips = compute_flip_per_node(outcomes)

    # Merge into tree
    tree_path = os.path.join(args.root, "tree.json")
    tree = load_tree(tree_path)
    updated = 0
    for n in tree["nodes"]:
        nid = n["id"]
        if nid in flips and flips[nid] is not None:
            n.setdefault("metrics", {})
            n["metrics"]["flip"] = flips[nid]
            n["updated_at"] = now_iso()
            updated += 1
    atomic_save(tree, tree_path)
    print(f"Wrote flip% for {updated} nodes")

    # Optionally resubmit missing chunks
    if args.resubmit_missing and missing:
        s3_bucket, role_arn = resolve_batch_settings(
            DEFAULT_S3_BUCKET, DEFAULT_BATCH_ROLE_ARN
        )
        print(f"\nResubmitting missing chunks: {missing}")
        for chunk_idx in missing:
            batch_path = os.path.join(
                args.root,
                "batch_eval_iter0",
                "consistency",
                f"batch_input_{chunk_idx}.jsonl",
            )
            if not os.path.exists(batch_path):
                print(f"  chunk {chunk_idx}: input file {batch_path} missing, skip")
                continue
            file_hash = compute_file_hash(batch_path)[:12]
            job_name = f"premise-consistency-resume-{file_hash}-{chunk_idx}"
            s3_key = f"premise/consistency/{job_name}/input.jsonl"
            s3_output = f"s3://{s3_bucket}/premise/consistency/{job_name}/output/"
            print(f"  chunk {chunk_idx}: uploading...")
            s3_input = upload_to_s3(batch_path, s3_bucket, s3_key)
            print(f"  chunk {chunk_idx}: submitting...")
            arn = submit_batch_job(
                bedrock,
                s3_input,
                s3_output,
                job_name,
                judge_model,
                role_arn,
            )
            print(f"  chunk {chunk_idx}: ARN={arn}")
