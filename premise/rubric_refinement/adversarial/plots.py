# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Pareto frontier plots: alpha/flip against verified fool rate.

Reporting only — nothing here feeds back into the tree or the selection. Two
modes: `paper` plots the paper-settings (iter-2) numbers on the Pareto winners
plus their v0 baselines, `iter0` plots the cheaper screening round.

Kept out of `pareto.py` so that importing the selection rule does not pull in
matplotlib; `_pyplot()` defers the import further still, to call time, and pins
the non-interactive Agg backend without imposing it on the process globally.

Both modes read `tree.json` and `pareto_winners.json` from `--root`. The paper
mode additionally wants each trait's v0 VFR under the v13 verifier, which
lives on the discovery-side rubric files rather than in the tree; pass those with
`--v0-files`. Without them the paper mode has no v0 baselines to plot, so it says
so and plots the refinements alone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from premise.rubric_refinement.tree import ROOT


def load_tree_and_data(root: str = ROOT, v0_files: list[str] | None = None):
    """Read the tree, the Pareto winner ids, and any v0 VFRs to plot against.

    `v0_files` are rubric JSONs whose entries may carry `v0_vfr_v13_verifier`;
    traits without it are simply absent from the returned map, which the callers
    treat as "no v0 baseline for this trait".
    """
    t = json.loads(Path(f"{root}/tree.json").read_text())
    pareto_ids = set(json.loads(Path(f"{root}/pareto_winners.json").read_text())["ids"])

    v0_vfr_v13 = {}
    for f in v0_files or []:
        if not os.path.exists(f):
            print(f"  WARN: --v0-files entry not found, skipping: {f}")
            continue
        for r in json.loads(Path(f).read_text()).get("rubrics", []):
            if r.get("v0_vfr_v13_verifier") is not None:
                v0_vfr_v13[r["id"]] = r["v0_vfr_v13_verifier"]

    return t, pareto_ids, v0_vfr_v13


def pareto_front(points):
    """Given list of (x, y) where x=reliability (higher better), y=VFR (lower better),
    return the indices of Pareto-dominant points."""
    n = len(points)
    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i iff j's x >= i's x AND j's y <= i's y with at least one strict
            if (
                points[j][0] >= points[i][0]
                and points[j][1] <= points[i][1]
                and (points[j][0] > points[i][0] or points[j][1] < points[i][1])
            ):
                is_pareto[i] = False
                break
    return [i for i, p in enumerate(is_pareto) if p]


def dataset_colors(records) -> dict[str, str]:
    """Assign a stable colour per dataset actually present in `records`.

    Built from the data rather than a fixed table so any number of datasets
    plots, named anything. Sorted so the same corpus always gets the same colour
    across the two modes and across reruns; cycles if there are more datasets
    than colours.
    """
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    names = sorted({r["dataset"] for r in records if r.get("dataset")})
    return {ds: palette[i % len(palette)] for i, ds in enumerate(names)}


def _pyplot():
    """Return pyplot with the non-interactive Agg backend selected.

    Called from the plotting functions rather than at module scope so that
    importing this module has no global side effect.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_paper_mode(outdir, root=ROOT, v0_files=None):
    """Paper-settings Pareto: alpha vs VFR under the v13 verifier.

    Plots every Pareto-winning refinement that has both metrics, plus the v0
    baseline of each trait whose VFR was supplied via `v0_files`.
    """
    plt = _pyplot()
    t, pareto_ids, v0_vfr_v13 = load_tree_and_data(root, v0_files)
    v0s = {n["trait_id"]: n for n in t["nodes"] if n.get("depth", 0) == 0}

    # Collect (trait_id, dataset, variant_label, alpha, vfr)
    records = []
    for n in t["nodes"]:
        if n["id"] not in pareto_ids:
            continue
        m = n.get("metrics", {})
        if m.get("alpha") is None or m.get("vfr_paper") is None:
            continue
        tid = n["trait_id"]
        ds = v0s[tid].get("dataset")
        records.append(
            {
                "trait": tid,
                "dataset": ds,
                "type": "refined",
                "node_id": n["id"],
                "alpha": m["alpha"],
                "vfr": m["vfr_paper"],
                "ops": "+".join(n["ops"]),
            }
        )

    # v0s with paper metrics + v13 VFR
    for tid, v0 in v0s.items():
        v0pm = v0.get("v0_paper_metrics", {})
        if v0pm.get("alpha") is None or v0_vfr_v13.get(tid) is None:
            continue
        records.append(
            {
                "trait": tid,
                "dataset": v0.get("dataset"),
                "type": "v0",
                "node_id": f"{tid}__v0",
                "alpha": v0pm["alpha"],
                "vfr": v0_vfr_v13[tid],
                "ops": "v0",
            }
        )

    if not records:
        raise ValueError(
            f"Nothing to plot: no Pareto node in {root}/tree.json has both "
            "metrics.alpha and metrics.vfr_paper. Run `refine adversarial "
            "pareto-eval` first."
        )
    if not any(r["type"] == "v0" for r in records):
        print(
            "  Note: no v0 baselines to plot — pass --v0-files with rubric JSONs "
            "carrying v0_vfr_v13_verifier to show where each trait started."
        )

    dataset_color = dataset_colors(records)

    # ---- Corpus-level aggregate plot ----
    fig, ax = plt.subplots(figsize=(9, 7))
    for ds in sorted(dataset_color):
        ds_records = [r for r in records if r["dataset"] == ds]
        v0_pts = [(r["alpha"], r["vfr"]) for r in ds_records if r["type"] == "v0"]
        ref_pts = [(r["alpha"], r["vfr"]) for r in ds_records if r["type"] == "refined"]
        if v0_pts:
            xs, ys = zip(*v0_pts)
            ax.scatter(
                xs,
                ys,
                marker="s",
                s=60,
                facecolors="none",
                edgecolors=dataset_color[ds],
                linewidths=1.5,
                label=f"{ds} v0 (n={len(v0_pts)})",
            )
        if ref_pts:
            xs, ys = zip(*ref_pts)
            ax.scatter(
                xs,
                ys,
                marker="o",
                s=30,
                color=dataset_color[ds],
                alpha=0.6,
                label=f"{ds} refined (n={len(ref_pts)})",
            )

    ax.set_xlabel(r"IRA Krippendorff $\alpha$  (↑ better)", fontsize=12)
    ax.set_ylabel(r"VFR v13  (↓ better)", fontsize=12)
    ax.set_title(
        "Reliability–Robustness Pareto: v0 vs Pareto-winning refinements", fontsize=13
    )
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Global Pareto front
    pts = [(r["alpha"], r["vfr"]) for r in records]
    pareto_idx = pareto_front(pts)
    px = sorted([(pts[i][0], pts[i][1]) for i in pareto_idx])
    if px:
        px_sorted = sorted(px, key=lambda p: -p[0])  # sort by alpha desc
        xs = [p[0] for p in px_sorted]
        ys = [p[1] for p in px_sorted]
        ax.plot(
            xs,
            ys,
            "k--",
            alpha=0.4,
            linewidth=1,
            label=f"Global Pareto front (n={len(pareto_idx)})",
        )

    plt.tight_layout()
    out_global = os.path.join(outdir, "pareto_paper_corpus.png")
    plt.savefig(out_global, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Corpus plot → {out_global}")

    # ---- Per-trait grid (sorted by v0 VFR desc so high-vulnerability traits are top) ----
    trait_v0 = {}
    for r in records:
        if r["type"] == "v0":
            trait_v0[r["trait"]] = r
    # Each panel is one trait's v0 → refinements arrows, so with no v0 baselines
    # there is nothing to anchor a panel on; skip the grid rather than emit an
    # empty figure.
    if trait_v0:
        _plot_per_trait_grid(plt, records, trait_v0, dataset_color, outdir)
    else:
        print("  Skipping per-trait grid — it plots v0 → refinement arrows.")

    # ---- CSV export ----
    csv_path = os.path.join(outdir, "pareto_paper_points.csv")
    with open(csv_path, "w") as f:
        f.write("trait,dataset,type,node_id,ops,alpha,vfr_v13\n")
        for r in records:
            f.write(
                f"{r['trait']},{r['dataset']},{r['type']},{r['node_id']},"
                f"{r['ops']},{r['alpha']:.3f},{r['vfr']:.3f}\n"
            )
    print(f"CSV → {csv_path}  ({len(records)} rows)")


def _plot_per_trait_grid(plt, records, trait_v0, dataset_color, outdir):
    """One small panel per trait: its v0 square, arrows out to each refinement."""
    # Traits with v0 sorted by v0 VFR desc
    trait_order = sorted(trait_v0.keys(), key=lambda t: -trait_v0[t]["vfr"])
    n_traits = len(trait_order)
    ncols = 6
    nrows = (n_traits + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.4, nrows * 1.9), sharex=True, sharey=True
    )
    axes = axes.flatten() if n_traits > 1 or ncols > 1 else [axes]

    for i, tid in enumerate(trait_order):
        ax = axes[i]
        trait_records = [r for r in records if r["trait"] == tid]
        ds = trait_records[0]["dataset"] if trait_records else "?"
        color = dataset_color.get(ds, "black")
        v0 = trait_v0[tid]
        refined = [r for r in trait_records if r["type"] == "refined"]
        # Draw arrows from v0 to each refinement
        if v0 and refined:
            for r in refined:
                ax.annotate(
                    "",
                    xy=(r["alpha"], r["vfr"]),
                    xytext=(v0["alpha"], v0["vfr"]),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "alpha": 0.35,
                        "lw": 0.6,
                    },
                )
        if refined:
            ax.scatter(
                [r["alpha"] for r in refined],
                [r["vfr"] for r in refined],
                marker="o",
                s=25,
                color=color,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
            )
        if v0:
            ax.scatter(
                [v0["alpha"]],
                [v0["vfr"]],
                marker="s",
                s=70,
                facecolors="none",
                edgecolors=color,
                linewidths=2.0,
            )
        ax.set_title(f"{tid[:30]}\n(v0 VFR={v0['vfr']:.2f})", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.03, 0.85)
        ax.grid(alpha=0.2)

    for j in range(n_traits, len(axes)):
        axes[j].set_visible(False)

    # Shared axis labels
    fig.supxlabel(r"IRA Krippendorff $\alpha$  (↑ better)", fontsize=11)
    fig.supylabel(r"VFR v13  (↓ better)", fontsize=11)
    fig.suptitle(
        r"Per-trait reliability–robustness: ■ v0 baseline → ● Pareto refinements  "
        r"(sorted by v0 VFR desc, colored by dataset)",
        fontsize=11,
    )
    plt.tight_layout()
    out_grid = os.path.join(outdir, "pareto_paper_per_trait.png")
    plt.savefig(out_grid, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-trait grid → {out_grid}")


def plot_iter0_mode(outdir, root=ROOT, v0_files=None):
    """Iter-0 screening scatter: flip vs VFR, both lower-is-better.

    Both axes improve downward, so the Pareto front sits bottom-left.
    """
    plt = _pyplot()
    t, pareto_ids, v0_vfr_v13 = load_tree_and_data(root, v0_files)
    v0s = {n["trait_id"]: n for n in t["nodes"] if n.get("depth", 0) == 0}

    records = []
    for n in t["nodes"]:
        if n.get("depth", 0) == 0:
            continue
        m = n.get("metrics", {})
        if m.get("vfr") is None or m.get("flip") is None:
            continue
        tid = n["trait_id"]
        records.append(
            {
                "trait": tid,
                "dataset": v0s[tid].get("dataset"),
                "type": "pareto_winner" if n["id"] in pareto_ids else "refined",
                "node_id": n["id"],
                "vfr": m["vfr"],
                "flip": m["flip"],
                "ops": "+".join(n["ops"]),
            }
        )
    # v0 nodes intentionally omitted — iter_0 vfr unmeasured; regime differs from v13 in paper.
    if not records:
        raise ValueError(
            f"Nothing to plot: no measured node in {root}/tree.json has both "
            "metrics.vfr and metrics.flip. Run `refine adversarial eval` first."
        )

    dataset_color = dataset_colors(records)

    fig, ax = plt.subplots(figsize=(9, 7))
    # Plot non-Pareto refinements first
    for ds in sorted(dataset_color):
        normal = [
            (r["flip"], r["vfr"])
            for r in records
            if r["dataset"] == ds and r["type"] == "refined"
        ]
        if normal:
            xs, ys = zip(*normal)
            ax.scatter(
                xs,
                ys,
                marker="o",
                s=15,
                color=dataset_color[ds],
                alpha=0.25,
                label=f"{ds} refined (n={len(normal)})",
            )
    # Pareto winners in brighter color
    for ds in sorted(dataset_color):
        pwin = [
            (r["flip"], r["vfr"])
            for r in records
            if r["dataset"] == ds and r["type"] == "pareto_winner"
        ]
        if pwin:
            xs, ys = zip(*pwin)
            ax.scatter(
                xs,
                ys,
                marker="*",
                s=80,
                color=dataset_color[ds],
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
                label=f"{ds} Pareto (n={len(pwin)})",
            )

    ax.set_xlabel(r"flip%  (↓ better)", fontsize=12)
    ax.set_ylabel(r"VFR v0  (↓ better)", fontsize=12)
    ax.set_title(
        f"Iter-0 scatter: flip vs VFR on {len(records)} refinements", fontsize=13
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    out = os.path.join(outdir, "pareto_iter0_corpus.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Iter-0 plot → {out}")


def plot_main():
    ap = argparse.ArgumentParser(
        description="Pareto frontier plots for an adversarial-repair corpus."
    )
    ap.add_argument("--mode", choices=["paper", "iter0", "both"], default="both")
    ap.add_argument(
        "--root",
        default=ROOT,
        help=f"Corpus root holding tree.json and pareto_winners.json (default: {ROOT})",
    )
    ap.add_argument(
        "--v0-files",
        nargs="*",
        default=None,
        help="Rubric JSONs carrying v0_vfr_v13_verifier, for the paper mode's v0 "
        "baselines. Omit to plot refinements only.",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Where to write the plots (default: <root>/plots)",
    )
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(args.root, "plots")
    os.makedirs(outdir, exist_ok=True)

    if args.mode in ("paper", "both"):
        plot_paper_mode(outdir, root=args.root, v0_files=args.v0_files)
    if args.mode in ("iter0", "both"):
        plot_iter0_mode(outdir, root=args.root, v0_files=args.v0_files)
