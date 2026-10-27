# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Seed scaffolder for the predictor-beam refinement pipeline.

Builds the inputs that `premise refine adversarial predict` consumes.

Given a dataset path, policy rubrics, preference data, and optional
adversarial-VFR results, writes:

    <root>/
    ├── manifest.json
    ├── tree.json                                # one v0 node per trait
    ├── operator_clauses.json                    # 7-operator schema, no per-trait clauses yet
    ├── rubrics/v0_split/<dataset>/<trait>__v0.json
    └── samples/
        ├── <dataset>/{samples_20a,samples_20b,samples_full}.jsonl
        ├── pairwise_<dataset>_50.jsonl          # for refine adversarial eval (--iter 1)
        ├── adv_<dataset>_50.jsonl               # for refine adversarial eval (--iter 1)
        └── paper/{pairwise_<dataset>_200,adv_<dataset>_200}.jsonl

The `50` and `200` are `--n-eval` and `--n-full`. Rather than have downstream
stages reconstruct these names, `manifest.json` records the paths it wrote under
`datasets.<dataset>.samples_eval_*`, and `refine adversarial eval` reads them
from there.

If an adversarial-VFR results file is supplied, each v0 file gets a
`v0_vfr_v0_verifier` field and `high_vfr=True/False` based on `--vfr-threshold`.
With `--high-vfr-only`, only high-VFR traits are seeded.
"""

from __future__ import annotations

import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

# Operator definitions live in operators.py — the single source of truth shared
# with repair.py. They get written to operator_clauses.json here so
# `refine adversarial predict` can read the intents without a separate per-trait
# clause-generation pass.
from premise.rubric_refinement.adversarial.operators import OPERATOR_TABLE
from premise.rubric_refinement.tree import ROOT


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_traits(traits_file: str) -> list[dict]:
    d = json.loads(Path(traits_file).read_text())
    return d.get("rubrics") or d.get("traits") or []


def _load_vfr(vfr_file: str | None) -> dict[str, float]:
    if not vfr_file or not os.path.exists(vfr_file):
        return {}
    d = json.loads(Path(vfr_file).read_text())
    out = {}
    for tid, info in d.items():
        if isinstance(info, dict) and "verified_fool_rate" in info:
            out[tid] = float(info["verified_fool_rate"])
    return out


def _filter_traits(traits, vfr_by_id, threshold, high_vfr_only):
    if not high_vfr_only:
        return traits
    if not vfr_by_id:
        raise ValueError("--high-vfr-only requires --vfr-results")
    return [t for t in traits if vfr_by_id.get(t["id"], 0.0) >= threshold]


def _write_v0_files(traits, vfr_by_id, threshold, dataset, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for r in traits:
        vfr = vfr_by_id.get(r["id"])
        payload = {
            "schema_version": "1.0.0",
            "trait_id": r["id"],
            "operator": "baseline",
            "operator_slug": "v0",
            "rubrics": [
                {
                    "id": r["id"],
                    "corpus": r.get("corpus", f"discovered_{dataset}"),
                    "category": "refined",
                    "description": r["description"],
                }
            ],
            "dataset": dataset,
            "high_vfr": (vfr is not None and vfr >= threshold),
            "v0_vfr_v0_verifier": vfr,
        }
        path = out_dir / f"{r['id']}__v0.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(
            {
                "id": r["id"],
                "path": str(path),
                "vfr": vfr,
                "high_vfr": payload["high_vfr"],
            }
        )
    return written


def _slice_battles(battles_path, out_a, out_b, out_full, n=20, full_n=200, seed=42):
    """Disjoint slices: 20a, 20b, full_n, in shuffled order with given seed."""
    with open(battles_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    a = lines[:n]
    b = lines[n : 2 * n]
    full = lines[2 * n : 2 * n + full_n]
    out_a.write_text("\n".join(a) + ("\n" if a else ""))
    out_b.write_text("\n".join(b) + ("\n" if b else ""))
    out_full.write_text("\n".join(full) + ("\n" if full else ""))
    return len(a), len(b), len(full)


def _slice_battles_n(battles_path, out_path, n, seed):
    with open(battles_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    sel = lines[:n]
    out_path.write_text("\n".join(sel) + ("\n" if sel else ""))
    return len(sel)


def _write_operator_clauses(out_path, trait_ids):
    op_meta = {
        "schema_version": "1.0.0",
        "operators": {
            name: {"intent": intent, "pattern": pattern}
            for name, intent, pattern in OPERATOR_TABLE
        },
        "clauses_by_trait": {tid: {} for tid in trait_ids},
        "generated_at": _now_iso(),
        "note": "Per-trait clauses are empty; predictor uses global operator intents.",
    }
    out_path.write_text(json.dumps(op_meta, indent=2))


def _build_tree(written_v0_files, dataset, has_attack_data: bool):
    nodes = []
    for entry in written_v0_files:
        tid = entry["id"]
        node = {
            "id": f"{tid}__v0",
            "trait_id": tid,
            "parent_id": None,
            "depth": 0,
            "ops": [],
            "applied_path": [],
            "is_root": True,
            "rubric_path": entry["path"],
            "dataset": dataset,
            "has_attack_data": has_attack_data,
            "status": "pending",
            "metrics": {
                "vfr": entry.get("vfr"),
                "vfr_a": None,
                "vfr_b": None,
                "vfr_mean": entry.get("vfr"),
                "meta": None,
                "alpha": None,
                "flip": None,
            },
            "rejection_reason": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        nodes.append(node)
    return {"nodes": nodes, "created_at": _now_iso()}


def seed(
    *,
    root: str,
    dataset: str,
    traits_file: str,
    battles_file: str,
    dataset_path: str | None = None,
    vfr_file: str | None = None,
    high_vfr_only: bool = False,
    vfr_threshold: float = 0.5,
    seed_value: int = 42,
    n_screen: int = 20,
    n_full: int = 200,
    n_eval: int = 50,
) -> dict:
    """Seed the predictor-beam corpus root for a single dataset.

    Returns a small summary dict with paths and counts.
    """
    # High-level workflows provide the exact dataset directory. The legacy
    # root/dataset layout remains available to low-level callers.
    root_path = Path(dataset_path) if dataset_path else Path(root) / dataset
    root_path.mkdir(parents=True, exist_ok=True)

    traits = _load_traits(traits_file)
    vfr_by_id = _load_vfr(vfr_file)
    traits = _filter_traits(traits, vfr_by_id, vfr_threshold, high_vfr_only)
    if not traits:
        raise ValueError(
            f"No traits to seed (after high-VFR filter); "
            f"vfr_file={vfr_file}, threshold={vfr_threshold}"
        )

    # 1. v0 rubric files
    v0_dir = root_path / "rubrics" / "v0_split" / dataset
    written = _write_v0_files(traits, vfr_by_id, vfr_threshold, dataset, v0_dir)

    # 2. samples: per-dataset screening slices, plus the eval slices below
    screen_dir = root_path / "samples" / dataset
    screen_dir.mkdir(parents=True, exist_ok=True)
    samples_a = screen_dir / "samples_20a.jsonl"
    samples_b = screen_dir / "samples_20b.jsonl"
    samples_full = screen_dir / "samples_full.jsonl"
    na, nb, nfull = _slice_battles(
        battles_file,
        samples_a,
        samples_b,
        samples_full,
        n=n_screen,
        full_n=n_full,
        seed=seed_value,
    )

    # refine adversarial eval (iter 1) slices. Named after the dataset so seeding a
    # second dataset into the same root does not overwrite the first one's; the
    # paths land in manifest.json for `refine adversarial eval` to pick up.
    samples_dir = root_path / "samples"
    pairwise_eval = samples_dir / f"pairwise_{dataset}_{n_eval}.jsonl"
    adv_eval = samples_dir / f"adv_{dataset}_{n_eval}.jsonl"
    n_pairwise = _slice_battles_n(battles_file, pairwise_eval, n_eval, seed_value + 1)
    n_adv = _slice_battles_n(battles_file, adv_eval, n_eval, seed_value + 2)

    # pareto-eval (iter 2) layout — per dataset
    paper_dir = samples_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_pairwise = paper_dir / f"pairwise_{dataset}_{n_full}.jsonl"
    paper_adv = paper_dir / f"adv_{dataset}_{n_full}.jsonl"
    n_paper_pairwise = _slice_battles_n(
        battles_file, paper_pairwise, n_full, seed_value + 3
    )
    n_paper_adv = _slice_battles_n(battles_file, paper_adv, n_full, seed_value + 4)

    # 3. operator_clauses.json
    op_path = root_path / "operator_clauses.json"
    _write_operator_clauses(op_path, [w["id"] for w in written])

    # 4. tree.json
    tree = _build_tree(written, dataset, has_attack_data=False)
    tree_path = root_path / "tree.json"
    if tree_path.exists():
        existing = json.loads(Path(tree_path).read_text())
        # Merge: keep existing nodes, add new ones (skip duplicates by id)
        existing_ids = {n["id"] for n in existing.get("nodes", [])}
        for node in tree["nodes"]:
            if node["id"] not in existing_ids:
                existing["nodes"].append(node)
        existing["updated_at"] = _now_iso()
        tree_path.write_text(json.dumps(existing, indent=2))
        merged_n = len(existing["nodes"])
    else:
        tree_path.write_text(json.dumps(tree, indent=2))
        merged_n = len(tree["nodes"])

    # 5. attack_patterns/ — empty dir for now (predict will see "(none)")
    (root_path / "attack_patterns").mkdir(exist_ok=True)

    # 6. manifest
    manifest_path = root_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(Path(manifest_path).read_text())
        manifest.setdefault("datasets", {})
    else:
        manifest = {"schema_version": "1.0.0", "datasets": {}}
    manifest["datasets"][dataset] = {
        "input_preference_data": battles_file,
        "policy_rubrics_file": traits_file,
        "vfr_results": vfr_file,
        "battles": battles_file,
        "traits_file": traits_file,
        "vfr_file": vfr_file,
        "v0_split_dir": str(v0_dir),
        "samples_a": str(samples_a),
        "samples_b": str(samples_b),
        "samples_full": str(samples_full),
        "samples_eval_pairwise": str(pairwise_eval),
        "samples_eval_adv": str(adv_eval),
        "samples_paper_pairwise": str(paper_pairwise),
        "samples_paper_adv": str(paper_adv),
        "n_screen_a": na,
        "n_screen_b": nb,
        "n_full": nfull,
        "n_eval_pairwise": n_pairwise,
        "n_eval_adv": n_adv,
        "n_paper_pairwise": n_paper_pairwise,
        "n_paper_adv": n_paper_adv,
        "high_vfr_only": high_vfr_only,
        "vfr_threshold": vfr_threshold,
        "seed": seed_value,
        "traits": written,
        "high_vfr_count": sum(1 for w in written if w["high_vfr"]),
        "n_traits": len(written),
        "seeded_at": _now_iso(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "root": str(root_path),
        "dataset": dataset,
        "n_traits_seeded": len(written),
        "n_high_vfr": sum(1 for w in written if w["high_vfr"]),
        "tree_nodes_total": merged_n,
        "manifest_path": str(manifest_path),
        "tree_path": str(tree_path),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Seed the predictor-beam corpus root.")
    ap.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--dataset", help=argparse.SUPPRESS)
    ap.add_argument(
        "--dataset-path",
        help="Directory that will contain this dataset's repair state.",
    )
    ap.add_argument(
        "--policy-rubrics-file",
        dest="traits_file",
        help="Rubric JSON produced by discovery.",
    )
    ap.add_argument("--traits-file", dest="traits_file", help=argparse.SUPPRESS)
    ap.add_argument(
        "--input-preference-data",
        dest="battles_file",
        help="Pairwise preference JSONL used to prepare repair samples.",
    )
    ap.add_argument("--battles-file", dest="battles_file", help=argparse.SUPPRESS)
    ap.add_argument(
        "--vfr-results",
        dest="vfr_file",
        default=None,
        help="adversarial_gameability_results.json with per-trait verified_fool_rate",
    )
    ap.add_argument("--vfr-file", dest="vfr_file", help=argparse.SUPPRESS)
    ap.add_argument(
        "--high-vfr-only",
        action="store_true",
        help="Only seed traits with VFR >= --vfr-threshold (requires --vfr-results)",
    )
    ap.add_argument("--vfr-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n-screen",
        type=int,
        default=20,
        help="Size of each disjoint screening set (samples_20a/b)",
    )
    ap.add_argument(
        "--n-full", type=int, default=200, help="Size of paper-grade validation slice"
    )
    ap.add_argument(
        "--n-eval",
        type=int,
        default=50,
        help="Size of refine adversarial eval (iter-1) slice",
    )
    args = ap.parse_args()
    if not args.traits_file:
        ap.error("--policy-rubrics-file is required")
    if not args.battles_file:
        ap.error("--input-preference-data is required")
    if args.dataset_path:
        dataset = Path(args.battles_file).stem
    elif args.dataset:
        dataset = args.dataset
    else:
        ap.error("--dataset-path is required")

    summary = seed(
        root=args.root,
        dataset=dataset,
        dataset_path=args.dataset_path,
        traits_file=args.traits_file,
        battles_file=args.battles_file,
        vfr_file=args.vfr_file,
        high_vfr_only=args.high_vfr_only,
        vfr_threshold=args.vfr_threshold,
        seed_value=args.seed,
        n_screen=args.n_screen,
        n_full=args.n_full,
        n_eval=args.n_eval,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
