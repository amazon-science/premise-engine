# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Target Metric Correlation.

Computes Spearman correlation between per-model trait win rates and target
metrics (benchmark scores or ELO ratings). Lightweight stats-only analysis
-- no LLM calls.

`--win-rates` files come from `premise audit trait-score`.

Usage:
    # Correlate with benchmark scores
    premise audit target-correlation \
        --win-rates <trait_scoring_dir>/win_rates.json \
        --output results/correlation/benchmark/

    # Correlate with ELO ratings
    premise audit target-correlation \
        --win-rates <trait_scoring_dir>/win_rates.json \
        --elo-json <elo_ratings.json> \
        --output results/correlation/elo/

    # Correlate with category-level ELO
    premise audit target-correlation \
        --win-rates <trait_scoring_dir>/win_rates.json \
        --elo-json <elo_ratings.json> \
        --elo-category coding \
        --output results/correlation/elo_coding/

    # Compare multiple rubric corpora on the same target
    premise audit target-correlation \
        --win-rates <baseline_dir>/win_rates.json \
                    <discovered_dir>/win_rates.json \
        --output results/correlation/comparison/
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime

import numpy as np
from scipy import stats


def load_win_rates(win_rates_file: str) -> dict:
    """Load win rates JSON produced by `premise audit trait-score`."""
    with open(win_rates_file) as f:
        data = json.load(f)
    return data


def load_elo_ratings(elo_json: str, category: str = "overall") -> dict:
    """Load ELO ratings from JSON file.

    Supports two formats:
    - Flat: {"model_a": 1234, ...}
    - Nested: {"overall": {"model_a": 1234}, "coding": {...}, ...}
    """
    with open(elo_json) as f:
        data = json.load(f)

    # Nested format
    if category in data and isinstance(data[category], dict):
        return data[category]

    # Flat format (assume overall)
    if category == "overall" and all(
        isinstance(v, (int, float)) for v in data.values()
    ):
        return data

    print(
        f"WARN: Category '{category}' not found in ELO JSON. "
        f"Available: {[k for k in data if isinstance(data[k], dict)]}"
    )
    return {}


def match_models(win_rates_models: dict, target_metrics: dict) -> list:
    """Match model IDs between win rates and target metrics.

    Tries exact match first, then normalized matching (case-insensitive,
    strip org prefix).
    """
    matched = []

    # Build lookup for target metrics with normalized keys
    target_lookup = {}
    for model_id, value in target_metrics.items():
        target_lookup[model_id] = (model_id, value)
        # Also index by normalized forms
        normalized = model_id.lower().replace("/", "__").replace("-", "_")
        target_lookup[normalized] = (model_id, value)
        # Strip org prefix (e.g., "Qwen/Qwen2.5-7B" -> "Qwen2.5-7B")
        if "/" in model_id:
            short = model_id.split("/", 1)[1]
            target_lookup[short.lower()] = (model_id, value)

    for model_id in win_rates_models:
        # Try exact match
        if model_id in target_lookup:
            orig_id, value = target_lookup[model_id]
            matched.append((model_id, value))
            continue

        # Try normalized
        normalized = model_id.lower().replace("/", "__").replace("-", "_")
        if normalized in target_lookup:
            orig_id, value = target_lookup[normalized]
            matched.append((model_id, value))
            continue

        # Try short name
        if "/" in model_id:
            short = model_id.split("/", 1)[1].lower()
            if short in target_lookup:
                orig_id, value = target_lookup[short]
                matched.append((model_id, value))
                continue

        print(f"  WARN: No target metric for model '{model_id}', skipping")

    return matched


def compute_correlations(
    win_rates_data: dict,
    target_metrics: dict,
    min_models: int = 5,
    significance: float = 0.05,
) -> dict:
    """Compute per-trait Spearman correlation with target metric.

    Returns: {trait_id: {rho, p_value, significant, n_models}}
    """
    models_data = win_rates_data["models"]
    trait_ids = win_rates_data["trait_ids"]

    # Match models
    matched = match_models(models_data, target_metrics)
    if len(matched) < min_models:
        print(f"ERROR: Only {len(matched)} matched models (need >= {min_models})")
        return {}

    print(f"  Matched {len(matched)} models for correlation")
    model_ids = [m[0] for m in matched]
    target_values = [m[1] for m in matched]

    correlations = {}
    for trait_id in trait_ids:
        # Build win rate vector (same order as target_values)
        win_rate_vector = []
        for model_id in model_ids:
            trait_data = models_data[model_id].get(trait_id, {})
            wr = trait_data.get("win_rate", 0.5)
            win_rate_vector.append(wr)

        # Check for constant values (rho undefined)
        if len(set(win_rate_vector)) <= 1:
            correlations[trait_id] = {
                "rho": float("nan"),
                "p_value": float("nan"),
                "significant": False,
                "n_models": len(matched),
            }
            continue

        rho, p_value = stats.spearmanr(win_rate_vector, target_values)

        correlations[trait_id] = {
            "rho": round(float(rho), 4),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < significance),
            "n_models": len(matched),
        }

    return correlations


def compute_regression(win_rates_data: dict, target_metrics: dict) -> dict:
    """Fit Ridge regression on trait win rates to predict target metric.

    Returns R^2 (leave-one-out CV), coefficients, predicted vs actual.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import LeaveOneOut
    except ImportError:
        print("  WARN: sklearn not available, skipping regression")
        return {}

    models_data = win_rates_data["models"]
    trait_ids = win_rates_data["trait_ids"]
    matched = match_models(models_data, target_metrics)

    if len(matched) < 5:
        return {}

    model_ids = [m[0] for m in matched]
    target_values = np.array([m[1] for m in matched])

    # Build feature matrix: models x traits
    X = np.zeros((len(model_ids), len(trait_ids)))
    for i, model_id in enumerate(model_ids):
        for j, trait_id in enumerate(trait_ids):
            trait_data = models_data[model_id].get(trait_id, {})
            X[i, j] = trait_data.get("win_rate", 0.5)

    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    predictions = np.zeros(len(model_ids))

    for train_idx, test_idx in loo.split(X):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], target_values[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])

    # R^2 from LOO predictions
    ss_res = np.sum((target_values - predictions) ** 2)
    ss_tot = np.sum((target_values - np.mean(target_values)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Fit final model for coefficients
    final_model = Ridge(alpha=1.0)
    final_model.fit(X, target_values)

    coefficients = {
        trait_ids[j]: round(float(final_model.coef_[j]), 4)
        for j in range(len(trait_ids))
    }

    predicted_vs_actual = {
        model_ids[i]: {
            "actual": round(float(target_values[i]), 4),
            "predicted": round(float(predictions[i]), 4),
        }
        for i in range(len(model_ids))
    }

    return {
        "r_squared": round(float(r_squared), 4),
        "r_squared_note": "Leave-one-out cross-validated",
        "coefficients": coefficients,
        "predicted_vs_actual": predicted_vs_actual,
    }


def run_target_correlation(
    win_rates_files: list,
    output_dir: str,
    elo_json: str | None = None,
    elo_category: str = "overall",
    min_models: int = 5,
    significance: float = 0.05,
    regression: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for wr_file in win_rates_files:
        print(f"\n=== Processing {wr_file} ===")
        wr_data = load_win_rates(wr_file)

        # Determine target metrics
        if elo_json:
            print(f"Loading ELO ratings from {elo_json} (category: {elo_category})")
            target_metrics = load_elo_ratings(elo_json, elo_category)
            target_name = f"elo_{elo_category}"
        elif wr_data.get("target_metrics"):
            target_metrics = wr_data["target_metrics"]
            target_name = "benchmark"
        else:
            print(
                "ERROR: No target metrics found. Provide --elo-json or "
                "ensure win_rates.json has target_metrics."
            )
            sys.exit(1)

        print(f"  Target metrics for {len(target_metrics)} models")

        # Compute per-trait correlation
        correlations = compute_correlations(
            wr_data, target_metrics, min_models=min_models, significance=significance
        )

        if not correlations:
            print("  No correlations computed, skipping")
            continue

        # Regression
        regression_result = {}
        if regression:
            print("  Computing regression...")
            regression_result = compute_regression(wr_data, target_metrics)

        # Summarize
        rhos = [c["rho"] for c in correlations.values() if not np.isnan(c["rho"])]
        significant = [tid for tid, c in correlations.items() if c["significant"]]
        sorted_by_rho = sorted(
            [
                (tid, c["rho"])
                for tid, c in correlations.items()
                if not np.isnan(c["rho"])
            ],
            key=lambda x: -abs(x[1]),
        )

        result = {
            "source": wr_file,
            "target_metric": target_name,
            "baseline": wr_data.get("baseline"),
            "n_traits": len(correlations),
            "n_significant": len(significant),
            "mean_rho": round(float(np.mean(rhos)), 4) if rhos else None,
            "mean_abs_rho": round(float(np.mean([abs(r) for r in rhos])), 4)
            if rhos
            else None,
            "r_squared": regression_result.get("r_squared"),
            "correlations": correlations,
            "regression": regression_result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        all_results.append(result)

        # Print correlation table
        print(f"\n  {'Trait':<40} {'rho':>8} {'p-value':>10} {'Sig':>5}")
        print("  " + "-" * 65)
        for trait_id, rho in sorted_by_rho:
            c = correlations[trait_id]
            sig = "*" if c["significant"] else ""
            print(f"  {trait_id:<40} {rho:>8.3f} {c['p_value']:>10.4f} {sig:>5}")

        print(f"\n  Mean rho: {result['mean_rho']}")
        print(f"  Mean |rho|: {result['mean_abs_rho']}")
        print(f"  Significant traits: {len(significant)}/{len(correlations)}")
        if regression_result:
            print(f"  Regression R^2 (LOO): {regression_result['r_squared']}")

    # Save results
    if len(all_results) == 1:
        # Single corpus
        output = all_results[0]
    else:
        # Multi-corpus comparison
        output = {
            "comparison": True,
            "results": all_results,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    correlation_file = os.path.join(output_dir, "correlation.json")
    with open(correlation_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {correlation_file}")

    # Write human-readable table
    table_file = os.path.join(output_dir, "correlation_table.txt")
    with open(table_file, "w") as f:
        for result in all_results:
            f.write(f"Source: {result['source']}\n")
            f.write(f"Target: {result['target_metric']}\n")
            f.write(f"Baseline: {result['baseline']}\n")
            f.write(
                f"Models: {result['correlations'][next(iter(result['correlations'].keys()))]['n_models']}\n"
            )
            f.write(f"\n{'Trait':<40} {'rho':>8} {'p-value':>10} {'Sig':>5}\n")
            f.write("-" * 65 + "\n")

            sorted_traits = sorted(
                result["correlations"].items(),
                key=lambda x: -abs(x[1]["rho"]) if not np.isnan(x[1]["rho"]) else 0,
            )
            for trait_id, c in sorted_traits:
                sig = "*" if c["significant"] else ""
                rho = c["rho"]
                if np.isnan(rho):
                    f.write(f"{trait_id:<40} {'NaN':>8} {'NaN':>10}\n")
                else:
                    f.write(
                        f"{trait_id:<40} {rho:>8.3f} {c['p_value']:>10.4f} {sig:>5}\n"
                    )

            f.write(f"\nMean rho: {result['mean_rho']}\n")
            f.write(f"Mean |rho|: {result['mean_abs_rho']}\n")
            f.write(f"Significant: {result['n_significant']}/{result['n_traits']}\n")
            if result.get("r_squared") is not None:
                f.write(f"Regression R^2 (LOO): {result['r_squared']}\n")

            if result.get("regression", {}).get("predicted_vs_actual"):
                f.write(f"\n{'Model':<45} {'Actual':>8} {'Predicted':>10}\n")
                f.write("-" * 65 + "\n")
                f.writelines(
                    f"{model_id:<45} {pva['actual']:>8.3f} {pva['predicted']:>10.3f}\n"
                    for model_id, pva in sorted(
                        result["regression"]["predicted_vs_actual"].items()
                    )
                )

            f.write("\n" + "=" * 65 + "\n\n")

    print(f"Table saved to {table_file}")

    # Summary JSON
    summary = {
        "results": [
            {
                "source": r["source"],
                "target_metric": r["target_metric"],
                "mean_rho": r["mean_rho"],
                "mean_abs_rho": r["mean_abs_rho"],
                "n_significant": r["n_significant"],
                "n_traits": r["n_traits"],
                "r_squared": r.get("r_squared"),
                "top_3_traits": [
                    {"trait": tid, "rho": rho}
                    for tid, rho in sorted(
                        [
                            (t, c["rho"])
                            for t, c in r["correlations"].items()
                            if not np.isnan(c["rho"])
                        ],
                        key=lambda x: -abs(x[1]),
                    )[:3]
                ],
            }
            for r in all_results
        ],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PReMISE target metric correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--win-rates",
        nargs="+",
        required=True,
        help="Path(s) to win_rates.json from trait scoring runs",
    )
    parser.add_argument("--elo-json", help="ELO ratings JSON file")
    parser.add_argument(
        "--elo-category",
        default="overall",
        help="ELO category (overall, coding, math, etc.)",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--min-models",
        type=int,
        default=5,
        help="Minimum models for correlation (default: 5)",
    )
    parser.add_argument(
        "--significance",
        type=float,
        default=0.05,
        help="P-value threshold (default: 0.05)",
    )
    parser.add_argument(
        "--no-regression", action="store_true", help="Skip regression analysis"
    )
    args = parser.parse_args()

    run_target_correlation(
        win_rates_files=args.win_rates,
        output_dir=args.output,
        elo_json=args.elo_json,
        elo_category=args.elo_category,
        min_models=args.min_models,
        significance=args.significance,
        regression=not args.no_regression,
    )
