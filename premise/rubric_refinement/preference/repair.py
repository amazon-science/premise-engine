# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Preference repair: rank a trait pool by preference fit, keep the top K.

One of the paper's two repair strategies (its Table-1 row is `+pref repair`).
"Repair" here operates on the rubric *set*, not on rubric text: every trait
already carries a per-trait preference-fit score — how often that trait's own
verdict agrees with the human's chosen response — and the repair is to drop the
traits that fit preferences poorly. Sort the pool by that score, cut at K. No
rewriting, no operators, no search; the trait text is left exactly as
discovered.

The score comes from `premise.rubric_evaluation.pref_fit_analysis`, either as a
metrics JSON it already wrote (`--metrics`) or computed here from raw scores
(`--bt-scores` or `--ira-dir`, plus `--battles`). Nothing is re-judged, so this
costs no model calls.

Contrast with the other repair (the `..adversarial` subpackage): that one repairs
at the *text* level, rewriting each trait under anti-gaming operators, then
selects on (verified fool rate, flip rate). Both strategies end
in a selection rule; only the adversarial one rewrites first. Here preference
fit is the objective rather than a filter, and it is the only thing consulted.

The two compose — the paper's `PReMISE-pref+refined` variant runs this first and
adversarial repair on the survivors — but nothing in this module assumes that.

Usage:
    premise refine pref-repair \\
        --traits traits_all.json \\
        --metrics pref_fit_metrics.json \\
        --top-k 20 \\
        --output prefrank_K20.json

    premise refine pref-repair \\
        --traits traits_all.json \\
        --bt-scores bt_scores.jsonl --battles battles.jsonl \\
        --top-k 20 --output prefrank_K20.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from premise.rubric_evaluation.pref_fit_analysis import (
    build_score_matrix_from_bt,
    build_score_matrix_from_ira,
    compute_per_trait_accuracy,
    load_bt_scores,
    load_ira_raw,
    load_winners,
)


def load_trait_pool(traits_file: str) -> list[dict]:
    """Read the candidate pool from a discovery traits file.

    Accepts either key used across the pipeline: `rubrics` (discovery output,
    seed input) or `traits`.
    """
    d = json.loads(Path(traits_file).read_text())
    pool = d.get("rubrics") or d.get("traits") or []
    if not pool:
        raise ValueError(
            f"No traits found in {traits_file} (looked for 'rubrics'/'traits')"
        )
    return pool


def load_pref_fit(
    *,
    metrics_file: str | None = None,
    bt_scores: str | None = None,
    ira_dir: str | None = None,
    battles: str | None = None,
) -> dict[str, float]:
    """Get {trait_id: preference-fit accuracy in [0, 1]} from one of three sources.

    `metrics_file` reads the `per_trait_accuracy` block a previous
    `quality pref-fit-analysis` run wrote. The other two recompute it from raw
    per-(sample, trait) scores, which needs `battles` for the winner labels.
    """
    if metrics_file:
        d = json.loads(Path(metrics_file).read_text())
        acc = d.get("per_trait_accuracy")
        if not isinstance(acc, dict):
            raise ValueError(
                f"{metrics_file} has no 'per_trait_accuracy' object; is it a "
                "pref-fit-analysis output?"
            )
        return {tid: float(v) for tid, v in acc.items()}

    if not battles:
        raise ValueError("--battles is required with --bt-scores / --ira-dir")
    winners = load_winners(battles)

    if bt_scores:
        trait_data = load_bt_scores(bt_scores)
        _, trait_ids, matrix, actual = build_score_matrix_from_bt(trait_data, winners)
    elif ira_dir:
        trait_data = load_ira_raw(ira_dir)
        _, trait_ids, matrix, actual = build_score_matrix_from_ira(trait_data, winners)
    else:
        raise ValueError("Provide one of --metrics, --bt-scores, --ira-dir")

    return compute_per_trait_accuracy(matrix, actual, trait_ids)


def rank_by_preference_fit(
    pool: list[dict],
    pref_fit: dict[str, float],
    *,
    min_accuracy: float | None = None,
) -> tuple[list[dict], list[str], int]:
    """Sort the pool by preference fit, descending.

    Returns (ranked, unscored, n_scored):
      ranked    — the kept traits in rank order, each annotated with
                  `pref_fit_accuracy` and `pref_fit_rank` (rank is over the full
                  scored set, so it survives the `min_accuracy` cut unchanged).
      unscored  — trait_ids absent from `pref_fit`. Never silently kept: a trait
                  with no measurement cannot be ranked against one that has it.
      n_scored  — how many traits had a score, before `min_accuracy` filtering.

    Ties break on source_count then trait_id, so the ordering is deterministic
    across runs.
    """
    scored, unscored = [], []
    for t in pool:
        tid = t.get("id") or t.get("trait_id")
        if tid in pref_fit:
            scored.append((tid, t))
        else:
            unscored.append(tid)

    scored.sort(key=lambda p: (-pref_fit[p[0]], -p[1].get("source_count", 0), p[0]))

    ranked = []
    for rank, (tid, t) in enumerate(scored, start=1):
        acc = pref_fit[tid]
        if min_accuracy is not None and acc < min_accuracy:
            continue
        entry = dict(t)
        entry["pref_fit_accuracy"] = round(acc, 4)
        entry["pref_fit_rank"] = rank
        ranked.append(entry)
    return ranked, sorted(unscored), len(scored)


def run(
    *,
    traits_file: str,
    output: str,
    top_k: int = 20,
    metrics_file: str | None = None,
    bt_scores: str | None = None,
    ira_dir: str | None = None,
    battles: str | None = None,
    min_accuracy: float | None = None,
) -> dict[str, Any]:
    """Rank a trait pool by preference fit and write the top-K rubric file.

    The output matches the schema every downstream consumer expects (audits,
    `refine adversarial seed`): a `rubrics` list of trait dicts. Each kept trait
    also carries its `pref_fit_accuracy` and `pref_fit_rank`, and a `selection`
    block records how the cut was made.
    """
    pool = load_trait_pool(traits_file)
    pref_fit = load_pref_fit(
        metrics_file=metrics_file,
        bt_scores=bt_scores,
        ira_dir=ira_dir,
        battles=battles,
    )
    ranked, unscored, n_scored = rank_by_preference_fit(
        pool, pref_fit, min_accuracy=min_accuracy
    )
    selected = ranked[:top_k]
    if not selected:
        raise ValueError(
            f"Nothing selected: {len(pool)} pool traits, {len(pref_fit)} scored, "
            f"min_accuracy={min_accuracy}"
        )

    accs = [t["pref_fit_accuracy"] for t in selected]
    payload = {
        "schema_version": "1.0.0",
        "rubrics": selected,
        "n_rubrics": len(selected),
        "selection": {
            # Kept verbatim: downstream consumers key off this string.
            "strategy": "preference_fit_topk",
            "paper_row": "+pref repair",
            "metric": "per_trait_accuracy",
            "top_k": top_k,
            "n_pool": len(pool),
            "n_scored": n_scored,
            "n_above_min_accuracy": len(ranked),
            "n_unscored": len(unscored),
            "unscored_trait_ids": unscored,
            "min_accuracy": min_accuracy,
            "source": metrics_file or bt_scores or ira_dir,
            "battles": battles,
            "cutoff_accuracy": min(accs),
            "mean_accuracy": round(sum(accs) / len(accs), 4),
        },
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Pool: {len(pool)} traits, {n_scored} with a preference-fit score")
    if unscored:
        print(f"  Unscored (excluded): {len(unscored)} — {', '.join(unscored[:5])}")
    if min_accuracy is not None:
        print(f"  Above --min-accuracy {min_accuracy}: {len(ranked)}/{n_scored}")
    print(f"Selected top {len(selected)} by per-trait accuracy:")
    for t in selected:
        tid = t.get("id") or t.get("trait_id")
        print(f"  {t['pref_fit_rank']:>3}. {t['pref_fit_accuracy']:.3f}  {tid}")
    print(
        f"  cutoff={payload['selection']['cutoff_accuracy']:.3f} "
        f"mean={payload['selection']['mean_accuracy']:.3f}"
    )
    print(f"Wrote {out_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rank traits by preference fit and keep the top K."
    )
    ap.add_argument(
        "--traits", required=True, help="Candidate pool JSON (discovery output)"
    )
    ap.add_argument("--output", "-o", required=True, help="Output rubric JSON path")
    ap.add_argument("--top-k", "-k", type=int, default=20, help="How many to keep")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--metrics",
        default=None,
        help="pref-fit-analysis output JSON (reads per_trait_accuracy)",
    )
    src.add_argument(
        "--bt-scores", default=None, help="bt_scores.jsonl (needs --battles)"
    )
    src.add_argument(
        "--ira-dir",
        default=None,
        help="IRA results dir with raw_*.jsonl (needs --battles)",
    )
    ap.add_argument(
        "--battles", default=None, help="Battles JSONL, for actual winner labels"
    )
    ap.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Drop traits below this accuracy before cutting at K",
    )
    args = ap.parse_args()

    run(
        traits_file=args.traits,
        output=args.output,
        top_k=args.top_k,
        metrics_file=args.metrics,
        bt_scores=args.bt_scores,
        ira_dir=args.ira_dir,
        battles=args.battles,
        min_accuracy=args.min_accuracy,
    )


if __name__ == "__main__":
    main()
