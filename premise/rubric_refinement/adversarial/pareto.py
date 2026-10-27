# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Adversarial repair, Stage 5: the selection rule, plus the eval that feeds it.

This is where the adversarial strategy ends. `compute` takes the measured nodes
and returns the non-dominated ones; `pareto_eval_main` is the paper-settings
(iter-2) eval run on those winners.

A Pareto winner is an iter-N measured node that:
  - is non-dominated on (vfr ↓, alpha ↑) by any other iter-N measured node
    of the same trait, AND
  - strictly improves on v0 by --min-vfr-drop on VFR (default 0).
  - in constrained mode, retains alpha within --max-alpha-drop of v0.

For each surviving node, we look up its `rubric_path` (refined rubric text)
and emit one entry into <root>/<dataset>/rubrics/pareto.json. The audit
runners can then point at that file as a normal traits JSON.

Note this module emits rubric text but never authors it: the description copied
into pareto.json is whatever `repair.py` wrote, passed through unchanged. The
domination test reads `metrics.vfr` and `metrics.alpha`.

Outputs:
  <root>/<dataset>/pareto_winners.json  (just {"ids": [...], "criteria": ...})
  <root>/<dataset>/rubrics/pareto.json  (a traits-format file the audit consumes)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from premise.rubric_refinement.tree import ROOT, _run_tool, atomic_save, load_tree


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_dominated(node, others) -> bool:
    """Node is dominated when another is no-worse on (VFR, alpha) and
    strictly-better on at least one."""
    a_vfr = node["metrics"].get("vfr")
    a_alpha = node["metrics"].get("alpha")
    if a_vfr is None or a_alpha is None:
        return True  # incomplete metrics — exclude
    for o in others:
        if o["id"] == node["id"]:
            continue
        b_vfr = o["metrics"].get("vfr")
        b_alpha = o["metrics"].get("alpha")
        if b_vfr is None or b_alpha is None:
            continue
        no_worse = (b_vfr <= a_vfr) and (b_alpha >= a_alpha)
        strict = (b_vfr < a_vfr) or (b_alpha > a_alpha)
        if no_worse and strict:
            return True
    return False


def compute(
    root: str,
    dataset: str | None,
    *,
    iter_n: int = 1,
    min_vfr_drop: float = 0.0,
    max_alpha_drop: float | None = 0.0,
) -> dict:
    """Compute Pareto winners for one (root/dataset) corpus."""
    root_path = Path(root)
    if (root_path / "tree.json").exists():
        ds_root = root_path
        if dataset is None:
            manifest = json.loads((root_path / "manifest.json").read_text())
            datasets = list((manifest.get("datasets") or {}).keys())
            if len(datasets) != 1:
                raise ValueError(
                    "--dataset is required when the repair root contains "
                    "zero or multiple datasets"
                )
            dataset = datasets[0]
    else:
        if dataset is None:
            raise ValueError("--dataset is required for a multi-dataset root")
        ds_root = root_path / dataset
    tree_path = ds_root / "tree.json"
    if not tree_path.exists():
        raise FileNotFoundError(tree_path)
    tree = json.loads(tree_path.read_text())

    v0_by_trait = {n["trait_id"]: n for n in tree["nodes"] if n.get("depth", 0) == 0}
    measured = [
        n
        for n in tree["nodes"]
        if n.get("batch_eval_iter") == iter_n
        and n.get("depth", 0) > 0
        and n["metrics"].get("vfr") is not None
        and n["metrics"].get("alpha") is not None
    ]

    winners = []
    by_trait: dict[str, list[dict]] = {}
    for n in measured:
        by_trait.setdefault(n["trait_id"], []).append(n)

    for tid, cands in by_trait.items():
        v0 = v0_by_trait.get(tid)
        if not v0:
            continue
        v0_vfr = v0["metrics"].get("vfr")
        v0_alpha = v0["metrics"].get("alpha")
        if v0_vfr is None or (max_alpha_drop is not None and v0_alpha is None):
            continue
        # Filter on min VFR drop
        improved = [
            candidate
            for candidate in cands
            if candidate["metrics"]["vfr"] < v0_vfr - min_vfr_drop
        ]
        if max_alpha_drop is not None:
            improved = [
                candidate
                for candidate in improved
                if candidate["metrics"]["alpha"] >= v0_alpha - max_alpha_drop
            ]
        # Pareto-front on (vfr, alpha)
        winners.extend(c for c in improved if not is_dominated(c, improved))

    # Build outputs
    ids_path = ds_root / "pareto_winners.json"
    ids_payload = {
        "ids": [w["id"] for w in winners],
        "criteria": {
            "iter": iter_n,
            "min_vfr_drop": min_vfr_drop,
            "max_alpha_drop": max_alpha_drop,
            "metric_axes": ["vfr", "alpha"],
        },
        "generated_at": _now_iso(),
        "n_winners": len(winners),
        "n_traits_with_winners": len({w["trait_id"] for w in winners}),
    }
    ids_path.write_text(json.dumps(ids_payload, indent=2))

    # Build a traits-format rubric file for audit
    rubrics_dir = ds_root / "rubrics"
    rubrics_dir.mkdir(parents=True, exist_ok=True)
    pareto_path = rubrics_dir / "pareto.json"

    rubric_entries = []
    for w in winners:
        rp = w.get("rubric_path")
        if not rp or not os.path.exists(rp):
            continue
        try:
            d = json.loads(Path(rp).read_text())
            r = d["rubrics"][0]
            rubric_entries.append(
                {
                    "id": w["id"],  # node_id, e.g. trait__op1+op2
                    "trait_id": w["trait_id"],
                    "ops": w.get("ops", []),
                    "depth": w.get("depth", 0),
                    "category": r.get("category", "refined"),
                    "corpus": r.get("corpus", f"premise_v0_refined_{dataset}"),
                    "description": r["description"],
                    "measured_metrics": {
                        "vfr": w["metrics"].get("vfr"),
                        "alpha": w["metrics"].get("alpha"),
                    },
                    "v0_vfr": v0_by_trait[w["trait_id"]]["metrics"].get("vfr"),
                    "v0_alpha": v0_by_trait[w["trait_id"]]["metrics"].get("alpha"),
                }
            )
        except Exception:
            continue

    pareto_payload = {
        "schema_version": "1.0.0",
        "dataset": dataset,
        "method": "premise_v0_refined",
        "iter": iter_n,
        "rubrics": rubric_entries,
    }
    pareto_path.write_text(json.dumps(pareto_payload, indent=2))

    summary = {
        "dataset": dataset,
        "n_traits_total": len(by_trait),
        "n_traits_with_winners": ids_payload["n_traits_with_winners"],
        "n_winners": len(winners),
        "n_rubrics_written": len(rubric_entries),
        "pareto_winners_json": str(ids_path),
        "pareto_rubrics_json": str(pareto_path),
    }
    return summary


# ---------------------------------------------------------------------------
# Stage 5: iter-2 paper-settings eval on Pareto winners
# ---------------------------------------------------------------------------


def build_per_dataset_traits(tree, pareto_ids, work_dir):
    """Per-dataset traits JSON. Returns {dataset: traits_path, count}."""
    v0s = {n["trait_id"]: n for n in tree["nodes"] if n.get("depth", 0) == 0}
    by_ds = defaultdict(list)
    for nid in pareto_ids:
        node = next((x for x in tree["nodes"] if x["id"] == nid), None)
        if not node:
            continue
        rp = node.get("rubric_path")
        if not rp or not os.path.exists(rp):
            continue
        try:
            d = json.loads(Path(rp).read_text())
        except Exception:
            continue
        orig = d["rubrics"][0]
        # No default: the dataset decides which battles this trait is scored on, so
        # guessing one would silently score it against the wrong corpus.
        ds = v0s[node["trait_id"]].get("dataset")
        if not ds:
            raise ValueError(
                f"v0 node for trait '{node['trait_id']}' has no 'dataset' field; "
                "re-seed the corpus with `refine adversarial seed`."
            )
        by_ds[ds].append(
            {
                "id": nid,
                "corpus": orig.get("corpus", "discovered"),
                "category": orig.get("category", "refined"),
                "description": orig["description"],
            }
        )
    result = {}
    for ds, rubrics in by_ds.items():
        p = os.path.join(work_dir, f"{ds}_traits.json")
        with open(p, "w") as f:
            json.dump({"schema_version": "1.0.0", "rubrics": rubrics}, f, indent=2)
        result[ds] = {"path": p, "count": len(rubrics)}
    return result


def aggregate(tree, work_dir, ds):
    """Pull results back into tree for this dataset."""
    ds_dir = os.path.join(work_dir, ds)
    # Meta
    fp = os.path.join(ds_dir, "meta", "meta_trait_results_merged.json")
    if os.path.exists(fp):
        d = json.loads(Path(fp).read_text()).get("traits", {})
        for nid, r in d.items():
            node = next((x for x in tree["nodes"] if x["id"] == nid), None)
            if node:
                node["metrics"]["meta"] = r.get("majority_score")
                node["metrics"]["meta_traits"] = r.get("meta_traits", {})
    # IRA
    fp = os.path.join(ds_dir, "ira", "ira_scores.json")
    if os.path.exists(fp):
        d = json.loads(Path(fp).read_text()).get("traits", {})
        for nid, r in d.items():
            node = next((x for x in tree["nodes"] if x["id"] == nid), None)
            if node:
                node["metrics"]["alpha"] = r.get("krippendorff_alpha")
    # Consistency
    fp = os.path.join(ds_dir, "consistency", "consistency_scores.json")
    if os.path.exists(fp):
        d = json.loads(Path(fp).read_text()).get("traits", {})
        for nid, r in d.items():
            node = next((x for x in tree["nodes"] if x["id"] == nid), None)
            if node:
                node["metrics"]["flip_paper"] = r.get("win_flip_rate")
    # Paraphrase
    fp = os.path.join(ds_dir, "paraphrase", "paraphrase_scores.json")
    if os.path.exists(fp):
        d = json.loads(Path(fp).read_text()).get("traits", {})
        for nid, r in d.items():
            node = next((x for x in tree["nodes"] if x["id"] == nid), None)
            if node:
                node["metrics"]["paraphrase_rho"] = r.get("winrate_rho") or r.get(
                    "raw_rho"
                )
    # Adversarial
    fp = os.path.join(ds_dir, "adversarial", "adversarial_gameability_results.json")
    if os.path.exists(fp):
        d = json.loads(Path(fp).read_text())
        for nid, r in d.items():
            if not isinstance(r, dict):
                continue
            node = next((x for x in tree["nodes"] if x["id"] == nid), None)
            if node:
                node["metrics"]["vfr_paper"] = r.get("verified_fool_rate")


def pareto_eval_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--n-attempts", type=int, default=200)
    ap.add_argument("--iter", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-meta", action="store_true")
    ap.add_argument("--skip-ira", action="store_true")
    ap.add_argument("--skip-consistency", action="store_true")
    ap.add_argument("--skip-paraphrase", action="store_true")
    ap.add_argument("--skip-adversarial", action="store_true")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="subset of datasets to run; default all",
    )
    args = ap.parse_args()

    tree_path = os.path.join(args.root, "tree.json")
    tree = load_tree(tree_path)
    pareto_ids = json.loads(
        Path(os.path.join(args.root, "pareto_winners.json")).read_text()
    )["ids"]

    work_dir = os.path.join(args.root, f"pareto_paper_eval_iter{args.iter}")
    os.makedirs(work_dir, exist_ok=True)

    per_ds = build_per_dataset_traits(tree, pareto_ids, work_dir)
    if args.datasets:
        per_ds = {k: v for k, v in per_ds.items() if k in args.datasets}
    total_traits = sum(v["count"] for v in per_ds.values())
    print(
        f"Pareto refinements to evaluate: {total_traits} across {len(per_ds)} datasets"
    )
    for ds, info in per_ds.items():
        print(f"  {ds}: {info['count']}")

    # Request volume
    req = {
        "meta": 15,
        "ira": args.n_samples * 3,
        "consistency": args.n_samples * 5,
        "paraphrase": 5 * args.n_samples,
        "adversarial": args.n_attempts * 3,
    }
    skipped = set()
    for name in ["meta", "ira", "consistency", "paraphrase", "adversarial"]:
        if getattr(args, f"skip_{name}"):
            skipped.add(name)
    vol = sum(req[k] for k in req if k not in skipped) * total_traits
    print(f"\nRequest volume (ex-skipped): ~{vol:,}")
    if args.dry_run:
        return

    paper_samples = os.path.join(args.root, "samples", "paper")

    # Build command list per dataset
    all_cmds = []
    for ds, info in per_ds.items():
        ds_work = os.path.join(work_dir, ds)
        os.makedirs(ds_work, exist_ok=True)
        pairwise = os.path.join(paper_samples, f"pairwise_{ds}_200.jsonl")
        adv = os.path.join(paper_samples, f"adv_{ds}_200.jsonl")

        if not args.skip_meta:
            out = os.path.join(ds_work, "meta")
            os.makedirs(out, exist_ok=True)
            all_cmds.append(
                (
                    f"{ds}-meta",
                    [
                        sys.executable,
                        "-m",
                        "premise.rubric_evaluation.meta_trait",
                        "--traits",
                        info["path"],
                        "--output",
                        out,
                    ],
                )
            )
        if not args.skip_ira:
            out = os.path.join(ds_work, "ira")
            os.makedirs(out, exist_ok=True)
            all_cmds.append(
                (
                    f"{ds}-ira",
                    [
                        sys.executable,
                        "-m",
                        "premise.rubric_evaluation.ira",
                        "--traits",
                        info["path"],
                        "--samples",
                        pairwise,
                        "--output",
                        out,
                        "--max-samples",
                        str(args.n_samples),
                    ],
                )
            )
        if not args.skip_consistency:
            out = os.path.join(ds_work, "consistency")
            os.makedirs(out, exist_ok=True)
            all_cmds.append(
                (
                    f"{ds}-cons",
                    [
                        sys.executable,
                        "-m",
                        "premise.rubric_evaluation.consistency",
                        "--traits",
                        info["path"],
                        "--samples",
                        pairwise,
                        "--output",
                        out,
                        "--max-samples",
                        str(args.n_samples),
                        "--n-runs",
                        "5",
                        "--temperature",
                        "0.7",
                    ],
                )
            )
        if not args.skip_paraphrase:
            out = os.path.join(ds_work, "paraphrase")
            os.makedirs(out, exist_ok=True)
            all_cmds.append(
                (
                    f"{ds}-para",
                    [
                        sys.executable,
                        "-m",
                        "premise.rubric_evaluation.paraphrase",
                        "--traits",
                        info["path"],
                        "--samples",
                        pairwise,
                        "--output",
                        out,
                        "--max-samples",
                        str(args.n_samples),
                        "--k",
                        "5",
                    ],
                )
            )
        if not args.skip_adversarial:
            out = os.path.join(ds_work, "adversarial")
            os.makedirs(out, exist_ok=True)
            all_cmds.append(
                (
                    f"{ds}-adv",
                    [
                        sys.executable,
                        "-m",
                        "premise.rubric_evaluation.adversarial",
                        "--traits",
                        info["path"],
                        "--samples",
                        adv,
                        "--output",
                        out,
                        "--n-attempts",
                        str(args.n_attempts),
                        "--run-id",
                        f"refine-pareto-iter{args.iter}-{ds}",
                        "--verifier-version",
                        "v13",
                    ],
                )
            )

    print(f"\nLaunching {len(all_cmds)} jobs concurrently. Logs under {work_dir}/.")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(all_cmds)) as ex:
        futs = {
            ex.submit(_run_tool, cmd, os.path.join(work_dir, f"{name}.log")): name
            for name, cmd in all_cmds
        }
        for fut in as_completed(futs):
            name = futs[fut]
            rc = fut.result()
            print(
                f"  {name} done rc={rc} ({(time.time() - t0) / 60:.1f} min)", flush=True
            )

    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Aggregating.")
    for ds in per_ds:
        aggregate(tree, work_dir, ds)
    atomic_save(tree, tree_path)
    print("Tree updated.")


def main():
    ap = argparse.ArgumentParser(description="Compute Pareto winners per dataset.")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument(
        "--dataset",
        default=None,
        help="Dataset name. Optional for a guided single-dataset repair root.",
    )
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument(
        "--min-vfr-drop",
        type=float,
        default=0.0,
        help="Required strict VFR improvement over v0 (default: 0).",
    )
    ap.add_argument(
        "--max-alpha-drop",
        type=float,
        default=0.0,
        help="Maximum alpha loss from v0 (default: 0, paper constrained point).",
    )
    ap.add_argument(
        "--unconstrained",
        action="store_true",
        help="Do not constrain alpha; select the VFR/alpha Pareto frontier.",
    )
    args = ap.parse_args()
    summary = compute(
        args.root,
        args.dataset,
        iter_n=args.iter,
        min_vfr_drop=args.min_vfr_drop,
        max_alpha_drop=None if args.unconstrained else args.max_alpha_drop,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
