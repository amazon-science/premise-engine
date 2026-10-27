# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Adversarial repair — Stages 1-2: rewrite each trait under the anti-gaming operators.

This is where the adversarial strategy's repair actually happens, and it is the
only place in the package that writes new rubric text.

There is no separate rewriting pass. Stage 1 (`predict_prep_main`) enumerates
operator subsets per trait (`enumerate_depths`), then builds one prompt per trait
bundle carrying the operator library (`format_ops`) and asks the model, for each
candidate, to do two things in a single call — predict the post-repair metrics
AND write the repaired rubric text ("your output IS the refined rubric"). Stage 2
(`merge_bundles_main`) harvests that text from the `refined_rubric_description`
field, writes each variant to `rubrics/combo/<trait>__<ops>.json`, and upserts a
node into `tree.json`.

Everything downstream of here only filters and measures — `filters.py` (Stages
3-4: prune, batch eval) and `pareto.py` (Stage 5) read metrics and copy this text
through unchanged. The operator definitions themselves live in `operators.py`,
the tree structure in `..tree`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from premise.rubric_refinement.adversarial.operators import OPERATORS
from premise.rubric_refinement.tree import (
    ROOT,
    atomic_save,
    canonical_node_id,
    load_tree,
    now_iso,
    upsert,
    write_rubric_file,
)
from premise.utils import DEFAULT_REGION

# ---------------------------------------------------------------------------
# Stage 1: build per-bundle predictor prompts
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_HEADER = """You are a reliability-constrained rubric refinement METRIC PREDICTOR and RUBRIC WRITER.

You will process {n_traits} traits in this bundle. For EACH trait, for EVERY candidate op-subset listed,
you must:
  1. Predict final metrics after applying that op-chain to v0.
  2. Write the refined rubric description TEXT (what v0 would read like after edits).
  3. Estimate per-metric confidence and a gate verdict.

You do NOT call a separate generator — your output IS the refined rubric. Reason about operator
interactions: compounding anti-gaming clauses saturate VFR but increase parse complexity (hurts α,
flip); conflicting rules erode α; longer rubrics increase T=0.7 flip noise regardless of content.

## Operator library (same for all traits)

{operators_section}

## Rubric-writing constraints (SB-PLR structural budget — STRICT)

- Total rubric ≤ 150 words
- Main definition (first paragraph) ≤ 60 words, no nested if/then
- At most 2 anti-gaming / do-not-reward clauses, each ≤ 40 words
- At most 1 applicability gate ("Score 0 if <flat predicate>.")
- At most 3 flat numbered criteria (if used), NO nesting
- Prompt-agnostic: no topic-specific examples, no fabricated names

Rubrics MUST preserve v0's core intent — you're editing, not replacing.

## Output format

ONE JSON object per candidate, NEWLINE-DELIMITED, NO markdown fences, NO commentary.
Expected total lines = sum of candidates across all traits in this bundle ({expected_lines} lines).

Schema:
{{"trait_id": "<id>", "node_id": "<trait_id>__<ops>", "ops": [<sorted ops>], "refined_rubric_description": "<rubric text>", "predicted_metrics": {{"meta": <0-5>, "alpha": <-0.3 to +0.5>, "flip": <0-100>, "vfr": <0-1>}}, "confidence": {{"meta": "high|medium|low", "alpha": "...", "flip": "...", "vfr": "..."}}, "gate_verdict": "likely_pass|likely_fail_alpha|likely_fail_flip|likely_fail_vfr|likely_fail_meta|uncertain", "rationale": "<one sentence>"}}

"""


TRAIT_BLOCK = """## ============================================================
## Trait {idx}/{n_traits}: {trait_id}  (dataset: {dataset}, attack data: {has_attack})
## ============================================================

### v0 definition
{v0_desc}

### v0 measured metrics
{v0_metrics}

### Attack patterns on v0 (for context)
{attack_patterns}

### Candidates to predict + write rubric text for ({n_candidates} total)

{candidates_section}

"""


def enumerate_depths(existing_ids, trait_id, max_depth, operators=None):
    selected_operators = operators or OPERATORS
    out = []
    for d in range(1, max_depth + 1):
        for subset in combinations(selected_operators, d):
            ops = sorted(subset)
            nid = canonical_node_id(trait_id, ops)
            if nid not in existing_ids:
                out.append(
                    {"trait_id": trait_id, "ops": ops, "depth": d, "node_id": nid}
                )
    return out


def format_v0(node):
    rp = node.get("rubric_path")
    if rp and os.path.exists(rp):
        d = json.loads(Path(rp).read_text())
        return d["rubrics"][0]["description"]
    return "(not found)"


def format_v0_metrics(node):
    m = node.get("metrics", {})
    if not m or all(
        m.get(k) is None for k in ("meta", "alpha", "flip", "vfr", "vfr_mean")
    ):
        return "(not yet measured — v0 will be batch-evaluated alongside candidates)"
    vfr = m.get("vfr_mean") or m.get("vfr")
    meta = m.get("meta")
    alpha = m.get("alpha")
    flip = m.get("flip")
    parts = []
    parts.append(f"meta={meta}/5" if meta is not None else "meta=?")
    parts.append(f"α={alpha:+.3f}" if isinstance(alpha, (int, float)) else "α=?")
    parts.append(f"flip={flip}%" if flip is not None else "flip=?")
    parts.append(f"VFR={vfr}" if vfr is not None else "VFR=?")
    return ", ".join(parts)


def format_attack_patterns(root, trait_id):
    p = os.path.join(root, "attack_patterns", f"{trait_id}.json")
    if not os.path.exists(p):
        return "(no attack-pattern data for this trait; predict conservatively)"
    d = json.loads(Path(p).read_text())
    lines = []
    for pat in d.get("patterns", [])[:5]:
        lines.append(f"- {pat.get('pattern')} (n={pat.get('count')})")
    return "\n".join(lines) or "(no top patterns)"


def format_ops(op_meta, operators=None):
    selected_operators = operators or OPERATORS
    return "\n".join(
        f"- **{op}**: {op_meta['operators'].get(op, {}).get('intent', '')}"
        for op in selected_operators
    )


def format_candidates(cands):
    by_depth = defaultdict(list)
    for c in cands:
        by_depth[c["depth"]].append(c)
    out = []
    for d in sorted(by_depth):
        out.append(f"\n**Depth {d}** ({len(by_depth[d])} candidates):")
        for c in by_depth[d]:
            out.append(f"  - `{c['node_id']}`  ops={c['ops']}")
    return "\n".join(out)


def parse_prediction_jsonl(text: str) -> list[dict]:
    """Parse JSON objects from a model response that should be NDJSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()

    parsed = []
    for line in cleaned.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    if parsed:
        return parsed

    # Fallback for providers that concatenate JSON objects instead of
    # respecting the requested newline-delimited format.
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(cleaned):
        start = cleaned.find("{", offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict):
            parsed.append(value)
        offset = start + end
    return parsed


def predict_prep_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument(
        "--operator",
        dest="operators",
        action="append",
        choices=OPERATORS,
        help="Limit candidates to this canonical operator; repeat to select a set.",
    )
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument(
        "--traits-per-bundle",
        type=int,
        default=10,
        help="number of traits to pack into one sub-agent bundle",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Call the configured provider and write prediction JSONL files.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Predictor/writer model. Defaults to config role 'judge'.",
    )
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument(
        "--allow-large-bundles",
        action="store_true",
        help="Allow --execute with more than 100 candidate output lines per request.",
    )
    args = ap.parse_args()
    selected_operators = args.operators or OPERATORS
    if args.max_depth > len(selected_operators):
        ap.error(
            f"--max-depth {args.max_depth} exceeds the "
            f"{len(selected_operators)} selected operators"
        )

    tree_path = os.path.join(args.root, "tree.json")
    tree = load_tree(tree_path)
    op_meta = json.loads(
        Path(os.path.join(args.root, "operator_clauses.json")).read_text()
    )
    existing_ids = {n["id"] for n in tree["nodes"]}

    out_dir = os.path.join(args.root, "predictor_logs", f"iter_{args.iter}")
    os.makedirs(out_dir, exist_ok=True)

    # Build per-trait candidate lists
    v0_nodes = [n for n in tree["nodes"] if n.get("depth", 0) == 0]
    per_trait = {}
    for v0 in v0_nodes:
        cands = enumerate_depths(
            existing_ids,
            v0["trait_id"],
            args.max_depth,
            selected_operators,
        )
        per_trait[v0["trait_id"]] = {"v0": v0, "cands": cands}

    # Group traits into bundles
    trait_ids = list(per_trait.keys())
    bundles = [
        trait_ids[i : i + args.traits_per_bundle]
        for i in range(0, len(trait_ids), args.traits_per_bundle)
    ]

    bundle_summaries = []
    for bidx, tids in enumerate(bundles):
        trait_blocks = []
        total_cands = 0
        for idx, tid in enumerate(tids):
            v0 = per_trait[tid]["v0"]
            cands = per_trait[tid]["cands"]
            total_cands += len(cands)
            trait_blocks.append(
                TRAIT_BLOCK.format(
                    idx=idx + 1,
                    n_traits=len(tids),
                    trait_id=tid,
                    dataset=v0.get("dataset", "?"),
                    has_attack="yes"
                    if v0.get("has_attack_data")
                    else "NO (predict conservatively)",
                    v0_desc=format_v0(v0),
                    v0_metrics=format_v0_metrics(v0),
                    attack_patterns=format_attack_patterns(args.root, tid),
                    n_candidates=len(cands),
                    candidates_section=format_candidates(cands),
                )
            )
        prompt = PROMPT_TEMPLATE_HEADER.format(
            n_traits=len(tids),
            operators_section=format_ops(op_meta, selected_operators),
            expected_lines=total_cands,
        ) + "\n".join(trait_blocks)

        prompt_path = os.path.join(out_dir, f"bundle_{bidx:02d}__prompt.md")
        with open(prompt_path, "w") as f:
            f.write(prompt)

        payload = {
            "bundle_idx": bidx,
            "traits": tids,
            "n_candidates_total": total_cands,
            "expected_output_lines": total_cands,
        }
        with open(os.path.join(out_dir, f"bundle_{bidx:02d}__payload.json"), "w") as f:
            json.dump(payload, f, indent=2)

        bundle_summaries.append(
            {
                "bundle": bidx,
                "traits": tids,
                "n_cands": total_cands,
                "prompt_path": prompt_path,
                "expected_output_path": os.path.join(
                    out_dir, f"bundle_{bidx:02d}__predictions.jsonl"
                ),
            }
        )

        print(
            f"bundle {bidx}: {len(tids)} traits, {total_cands} candidates → {prompt_path}"
        )

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(bundle_summaries, f, indent=2)
    total = sum(b["n_cands"] for b in bundle_summaries)
    print(f"\nTotal: {len(bundles)} bundles, {total} candidates to predict")

    if not args.execute:
        print("Prepared prompts only. Re-run with --execute to call a provider.")
        return

    oversized = [b for b in bundle_summaries if b["n_cands"] > 100]
    if oversized and not args.allow_large_bundles:
        largest = max(b["n_cands"] for b in oversized)
        raise SystemExit(
            "--execute would request more than 100 JSON lines in one response "
            f"(largest bundle: {largest}). Lower --max-depth and/or "
            "--traits-per-bundle, or pass --allow-large-bundles."
        )

    from premise.providers import model_for_role
    from premise.utils import get_bedrock_client, invoke_model

    model = args.model or model_for_role("judge")
    client = get_bedrock_client(args.region)
    print(f"Executing {len(bundle_summaries)} bundles with {model}...")
    for index, bundle in enumerate(bundle_summaries, 1):
        prompt = Path(bundle["prompt_path"]).read_text()
        response = invoke_model(
            client,
            model,
            prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        raw_path = bundle["expected_output_path"].replace(
            "__predictions.jsonl", "__raw.txt"
        )
        Path(raw_path).write_text(response or "")
        predictions = parse_prediction_jsonl(response or "")
        with open(bundle["expected_output_path"], "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in predictions)
        print(
            f"  [{index}/{len(bundle_summaries)}] bundle {bundle['bundle']}: "
            f"{len(predictions)}/{bundle['n_cands']} parsed"
        )
        if not predictions:
            raise RuntimeError(
                f"No prediction JSON parsed from {raw_path}; inspect the raw response."
            )


# ---------------------------------------------------------------------------
# Stage 2: merge predictor JSONLs into tree.json
# ---------------------------------------------------------------------------


def merge_bundles_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--iter", type=int, required=True)
    args = ap.parse_args()

    tree_path = os.path.join(args.root, "tree.json")
    tree = load_tree(tree_path)
    log_dir = os.path.join(args.root, "predictor_logs", f"iter_{args.iter}")

    # Index v0 nodes by trait_id for corpus/category lookup
    v0_by_trait = {}
    for n in tree["nodes"]:
        if (
            n.get("depth", 0) == 0
            and n.get("rubric_path")
            and os.path.exists(n["rubric_path"])
        ):
            try:
                v0_by_trait[n["trait_id"]] = json.loads(
                    Path(n["rubric_path"]).read_text()
                )
            except Exception:
                pass

    merged = 0
    skipped_overwrite = 0
    skipped_missing_text = 0
    parse_err = 0
    total_lines = 0

    for pred_path in sorted(glob.glob(f"{log_dir}/bundle_*__predictions.jsonl")):
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    p = json.loads(line)
                except Exception:
                    parse_err += 1
                    continue
                trait = p.get("trait_id")
                if not trait:
                    parse_err += 1
                    continue
                ops = sorted(p.get("ops", []))
                nid = canonical_node_id(trait, ops)
                desc = p.get("refined_rubric_description") or p.get("description")
                if not desc:
                    skipped_missing_text += 1
                    continue
                pm = p.get("predicted_metrics", {})
                parent_id = (
                    canonical_node_id(trait, ops[:-1])
                    if len(ops) >= 2
                    else canonical_node_id(trait, [])
                    if len(ops) == 1
                    else None
                )

                existing = next((n for n in tree["nodes"] if n["id"] == nid), None)
                if existing and existing.get("status") not in (
                    "predicted",
                    "pending",
                    "pruned_by_predictor",
                    "prune_survivor",
                ):
                    # Real-eval'd → preserve real metrics, overlay predictions for accuracy tracking
                    existing["predicted_metrics"] = pm
                    existing["prediction_confidence"] = p.get("confidence", {})
                    existing["predictor_rationale"] = p.get("rationale", "")
                    existing["predictor_gate_verdict"] = p.get(
                        "gate_verdict", "uncertain"
                    )
                    existing["predictor_iteration"] = args.iter
                    existing["updated_at"] = now_iso()
                    skipped_overwrite += 1
                    continue

                rubric_path = write_rubric_file(
                    args.root, trait, ops, desc, v0_data=v0_by_trait.get(trait)
                )

                node = {
                    "id": nid,
                    "trait_id": trait,
                    "parent_id": parent_id,
                    "depth": len(ops),
                    "ops": ops,
                    "applied_path": ops,
                    "rubric_path": rubric_path,
                    "status": "predicted",
                    "decision_source": None,
                    "metrics": {},
                    "predicted_metrics": pm,
                    "prediction_confidence": p.get("confidence", {}),
                    "predictor_rationale": p.get("rationale", ""),
                    "predictor_gate_verdict": p.get("gate_verdict", "uncertain"),
                    "predictor_iteration": args.iter,
                }
                upsert(tree, node)
                merged += 1

    atomic_save(tree, tree_path)
    print(f"Lines processed: {total_lines}")
    print(f"Merged: {merged} new predicted nodes")
    print(f"Skipped (already measured): {skipped_overwrite}")
    print(f"Skipped (no text): {skipped_missing_text}")
    print(f"Parse errors: {parse_err}")

    by_trait = Counter(
        n["trait_id"] for n in tree["nodes"] if n.get("status") == "predicted"
    )
    print(f"\n{len(by_trait)} traits have predicted nodes")
    for t, c in by_trait.most_common(5):
        print(f"  {t}: {c}")
