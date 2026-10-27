# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Preference-fit metrics from per-rubric 0--10 pairwise scores.

Computed post-hoc from existing IRA raw output (raw_*.jsonl per trait) or
bt_scores.jsonl. No new LLM calls.

The primary paper metrics are nested 5-fold logistic-regression accuracy,
normalized coefficient entropy, and entropy-effective dimensionality. Simpler
per-trait and sum-of-deltas diagnostics are retained for repair and debugging.

Usage:
    python -m premise.rubric_evaluation.pref_fit_analysis \
        --ira-dir <path-to-ira-results-dir> \
        --battles <battles.jsonl> \
        --output <output.json>

Or from bt_scores:
    python -m premise.rubric_evaluation.pref_fit_analysis \
        --bt-scores <bt_scores.jsonl> \
        --battles <battles.jsonl> \
        --output <output.json>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_winners(battles_path: str) -> dict:
    """Load {sample_id: winner} from battles JSONL."""
    winners = {}
    with open(battles_path) as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            winners[b.get("id", "")] = b.get("winner", "")
    return winners


def load_ira_raw(ira_dir: str) -> dict:
    """Load per-(sample, trait, judge) 0--10 scores from IRA raw files.

    Legacy files containing only ``outcome`` are mapped back to 0/5/10.
    """
    trait_data: dict = defaultdict(lambda: defaultdict(list))
    ira_path = Path(ira_dir)
    for f in sorted(ira_path.glob("raw_*.jsonl")):
        trait_id = f.stem.replace("raw_", "")
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                sample_id = rec.get("sample_id", "")
                score = rec.get("score")
                if score is None and rec.get("outcome") is not None:
                    score = {0: 0, 1: 5, 2: 10}.get(int(rec["outcome"]))
                if sample_id and score is not None:
                    trait_data[trait_id][sample_id].append(float(score))
    return dict(trait_data)


def load_bt_scores(bt_scores_path: str) -> dict:
    """Load per-(sample, trait) score deltas from bt_scores.jsonl.

    Returns {trait_id: {sample_id: [pairwise scores]}}. Scores are centered at
    five when the feature matrix is built.
    """
    trait_data: dict = defaultdict(dict)
    with open(bt_scores_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            trait_id = rec.get("trait_id", "")
            sample_id = rec.get("sample_id", "")
            score = rec.get("score")
            if not trait_id or not sample_id or score is None:
                continue
            trait_data[trait_id].setdefault(sample_id, []).append(float(score))
    return dict(trait_data)


def build_score_matrix_from_ira(
    trait_data: dict,
    winners: dict,
) -> tuple:
    """Build the score matrix from IRA raw data.

    Returns:
        sample_ids: list of sample IDs (row order)
        trait_ids: list of trait IDs (column order)
        matrix: np.ndarray [n_samples x n_traits] of mean score minus five.
        actual_winners: np.ndarray [n_samples] with values in {+1, -1, 0}
                +1 = A is actual winner, -1 = B is actual winner, 0 = other
    """
    trait_ids = sorted(trait_data.keys())
    # Find common sample_ids across all traits
    sample_sets = [set(trait_data[t].keys()) for t in trait_ids]
    if sample_sets:
        common_samples = sorted(set.intersection(*sample_sets))
    else:
        common_samples = []

    # Filter to samples with known winners
    sample_ids = [s for s in common_samples if winners.get(s) in ("model_a", "model_b")]

    n_samples = len(sample_ids)
    n_traits = len(trait_ids)
    matrix = np.zeros((n_samples, n_traits), dtype=np.float64)
    actual = np.zeros(n_samples, dtype=np.float64)

    for i, sid in enumerate(sample_ids):
        w = winners[sid]
        actual[i] = 1.0 if w == "model_a" else -1.0

        for j, tid in enumerate(trait_ids):
            scores = trait_data[tid].get(sid, [])
            if scores:
                matrix[i, j] = float(np.mean(scores)) - 5.0

    return sample_ids, trait_ids, matrix, actual


def build_score_matrix_from_bt(
    trait_data: dict,
    winners: dict,
) -> tuple:
    """Build the score matrix from bt_scores data.

    Returns same shape as build_score_matrix_from_ira.
    """
    trait_ids = sorted(trait_data.keys())
    sample_sets = [set(trait_data[t].keys()) for t in trait_ids]
    if sample_sets:
        common_samples = sorted(set.intersection(*sample_sets))
    else:
        common_samples = []

    sample_ids = [s for s in common_samples if winners.get(s) in ("model_a", "model_b")]

    n_samples = len(sample_ids)
    n_traits = len(trait_ids)
    matrix = np.zeros((n_samples, n_traits), dtype=np.float64)
    actual = np.zeros(n_samples, dtype=np.float64)

    for i, sid in enumerate(sample_ids):
        w = winners[sid]
        actual[i] = 1.0 if w == "model_a" else -1.0

        for j, tid in enumerate(trait_ids):
            scores = trait_data[tid].get(sid, [])
            if scores:
                matrix[i, j] = float(np.mean(scores)) - 5.0

    return sample_ids, trait_ids, matrix, actual


def compute_per_trait_accuracy(
    matrix: np.ndarray,
    actual: np.ndarray,
    trait_ids: list,
) -> dict:
    """Per-trait predictive accuracy: how often does the trait's direction
    match the actual winner?

    Returns {trait_id: accuracy} where accuracy is in [0, 1].
    Excludes ties (matrix == 0) from the denominator.
    """
    results = {}
    for j, tid in enumerate(trait_ids):
        col = matrix[:, j]
        # Only count non-tie entries
        mask = col != 0
        if mask.sum() == 0:
            results[tid] = 0.0
            continue
        correct = ((col[mask] > 0) & (actual[mask] > 0)) | (
            (col[mask] < 0) & (actual[mask] < 0)
        )
        results[tid] = float(correct.sum()) / float(mask.sum())
    return results


def compute_effective_dimensionality(matrix: np.ndarray) -> dict:
    """Entropy-effective number of independent score directions.

    Returns:
        effective_dimensionality: exp(entropy(normalized covariance eigenvalues))
        n_components_90: number of PCs for 90% explained variance, retained as
                         a secondary diagnostic.
        explained_variance_ratios: list — per-PC variance ratios.
    """
    # Center the matrix
    centered = matrix - matrix.mean(axis=0, keepdims=True)

    # Handle degenerate cases
    if centered.shape[0] < 2 or centered.shape[1] < 2:
        return {
            "effective_dimensionality": float(min(centered.shape)),
            "n_components_90": min(centered.shape),
            "explained_variance_ratios": [],
        }

    # SVD-based PCA
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    variances = s**2
    total_variance = variances.sum()
    if total_variance <= 0:
        return {
            "effective_dimensionality": 0.0,
            "n_components_90": 0,
            "explained_variance_ratios": [],
        }
    explained = variances / total_variance

    cumulative = np.cumsum(explained)
    n_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    positive = explained[explained > 0]
    effective = float(np.exp(-np.sum(positive * np.log(positive))))

    return {
        "effective_dimensionality": effective,
        "n_components_90": n_90,
        "explained_variance_ratios": explained.tolist(),
    }


def compute_pairwise_correlation(matrix: np.ndarray, trait_ids: list) -> dict:
    """Pairwise Pearson correlation between trait columns.

    Returns:
        mean_abs_correlation: float
        n_high_correlation_pairs: int (|r| > 0.7)
        correlation_matrix: list of lists
    """
    n_traits = matrix.shape[1]
    if n_traits < 2:
        return {
            "mean_abs_correlation": 0.0,
            "n_high_correlation_pairs": 0,
            "correlation_matrix": [],
        }

    # Compute correlation matrix (handle constant columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(matrix.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Extract upper triangle (exclude diagonal)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper = corr[mask]

    mean_abs = float(np.abs(upper).mean()) if len(upper) > 0 else 0.0
    n_high = int((np.abs(upper) > 0.7).sum())

    return {
        "mean_abs_correlation": round(mean_abs, 4),
        "n_high_correlation_pairs": n_high,
        "correlation_matrix": corr.tolist(),
    }


def compute_aggregate_pref_fit(matrix: np.ndarray, actual: np.ndarray) -> dict:
    """Aggregate preference-fit: sum-of-deltas across all traits predicts winner.

    Returns:
        accuracy: float
        n_correct: int
        n_total: int
    """
    total_delta = matrix.sum(axis=1)  # sum across traits per battle
    predicted = np.sign(total_delta)  # +1 = A, -1 = B, 0 = tie

    # Exclude ties from accuracy computation
    non_tie = predicted != 0
    if non_tie.sum() == 0:
        return {"accuracy": 0.0, "n_correct": 0, "n_total": 0}

    correct = (predicted[non_tie] == actual[non_tie]).sum()
    total = int(non_tie.sum())
    return {
        "accuracy": round(float(correct) / total, 4),
        "n_correct": int(correct),
        "n_total": total,
    }


def compute_logistic_preference_fit(
    matrix: np.ndarray,
    actual: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Run the paper's nested-CV L2 logistic preference-fit protocol.

    The outer folds estimate held-out accuracy. Each training fold selects the
    regularization strength with an inner stratified CV. Coefficient entropy is
    computed from the mean absolute standardized coefficient across outer
    folds and normalized by log(K).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    mask = np.isfinite(matrix).all(axis=1) & (actual != 0)
    X = matrix[mask]
    y = (actual[mask] > 0).astype(int)
    class_counts = np.bincount(y, minlength=2)
    outer_folds = min(n_splits, int(class_counts.min()))
    if X.shape[0] < 4 or outer_folds < 2 or X.shape[1] == 0:
        return {
            "cv_accuracy": None,
            "cv_accuracy_std": None,
            "fold_accuracies": [],
            "selected_c": [],
            "bt_weight_entropy": None,
            "mean_abs_weights": [],
            "n_samples": int(X.shape[0]),
            "n_folds": 0,
            "reason": "Need at least two examples from each class.",
        }

    outer = StratifiedKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=random_state,
    )
    fold_accuracies = []
    selected_c = []
    absolute_weights = []
    c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    for fold_index, (train_index, test_index) in enumerate(outer.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        inner_counts = np.bincount(y_train, minlength=2)
        inner_folds = min(4, int(inner_counts.min()))

        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=2000,
                    ),
                ),
            ]
        )
        if inner_folds >= 2:
            inner = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=random_state + fold_index + 1,
            )
            estimator = GridSearchCV(
                pipeline,
                {"model__C": c_grid},
                cv=inner,
                scoring="accuracy",
                n_jobs=1,
            )
        else:
            pipeline.set_params(model__C=1.0)
            estimator = pipeline

        estimator.fit(X_train, y_train)
        fold_accuracies.append(float(estimator.score(X_test, y_test)))
        best = (
            estimator.best_estimator_
            if hasattr(estimator, "best_estimator_")
            else estimator
        )
        selected_c.append(float(best.named_steps["model"].C))
        absolute_weights.append(np.abs(best.named_steps["model"].coef_[0]))

    mean_abs_weights = np.mean(np.vstack(absolute_weights), axis=0)
    weight_total = float(mean_abs_weights.sum())
    if matrix.shape[1] <= 1 or weight_total <= 0:
        weight_entropy = 0.0
    else:
        probabilities = mean_abs_weights / weight_total
        positive = probabilities[probabilities > 0]
        weight_entropy = float(
            -np.sum(positive * np.log(positive)) / np.log(matrix.shape[1])
        )

    return {
        "cv_accuracy": round(float(np.mean(fold_accuracies)), 4),
        "cv_accuracy_std": round(float(np.std(fold_accuracies)), 4),
        "fold_accuracies": [round(value, 4) for value in fold_accuracies],
        "selected_c": selected_c,
        "bt_weight_entropy": round(weight_entropy, 4),
        "mean_abs_weights": mean_abs_weights.tolist(),
        "n_samples": int(X.shape[0]),
        "n_folds": outer_folds,
    }


def run_analysis(
    *,
    ira_dir: str | None = None,
    bt_scores_path: str | None = None,
    battles_path: str,
    output_path: str,
) -> dict:
    """Run all extended pref-fit metrics.

    Reads from either IRA raw output or bt_scores. Computes:
    - nested-CV logistic accuracy and normalized coefficient entropy
    - per_trait_accuracy
    - entropy-effective dimensionality
    - effective_predictive_traits (count > 55%)
    - pairwise_correlation
    - aggregate_pref_fit
    """
    winners = load_winners(battles_path)
    print(f"Loaded {len(winners)} battle winners from {battles_path}")

    if ira_dir:
        trait_data = load_ira_raw(ira_dir)
        print(f"Loaded IRA raw data: {len(trait_data)} traits")
        sample_ids, trait_ids, matrix, actual = build_score_matrix_from_ira(
            trait_data,
            winners,
        )
    elif bt_scores_path:
        trait_data = load_bt_scores(bt_scores_path)
        print(f"Loaded bt_scores: {len(trait_data)} traits")
        sample_ids, trait_ids, matrix, actual = build_score_matrix_from_bt(
            trait_data,
            winners,
        )
    else:
        raise ValueError("Must provide either --ira-dir or --bt-scores")

    print(f"Score matrix: {matrix.shape[0]} battles × {matrix.shape[1]} traits")

    # Compute metrics
    per_trait_acc = compute_per_trait_accuracy(matrix, actual, trait_ids)
    mean_acc = (
        round(sum(per_trait_acc.values()) / len(per_trait_acc), 4)
        if per_trait_acc
        else 0.0
    )
    effective_predictive = sum(1 for v in per_trait_acc.values() if v > 0.55)

    dim_info = compute_effective_dimensionality(matrix)
    corr_info = compute_pairwise_correlation(matrix, trait_ids)
    agg_fit = compute_aggregate_pref_fit(matrix, actual)
    logistic_fit = compute_logistic_preference_fit(matrix, actual)

    results = {
        "n_battles": matrix.shape[0],
        "n_traits": matrix.shape[1],
        "trait_ids": trait_ids,
        "per_trait_accuracy": per_trait_acc,
        "mean_per_trait_accuracy": mean_acc,
        "effective_predictive_traits": effective_predictive,
        "preference_fit": logistic_fit,
        "effective_dimensionality": round(dim_info["effective_dimensionality"], 4),
        "effective_dimensionality_ratio": round(
            dim_info["effective_dimensionality"] / matrix.shape[1], 4
        )
        if matrix.shape[1]
        else 0.0,
        "n_components_90": dim_info["n_components_90"],
        "explained_variance_ratios": dim_info["explained_variance_ratios"][:20],
        "mean_abs_correlation": corr_info["mean_abs_correlation"],
        "n_high_correlation_pairs": corr_info["n_high_correlation_pairs"],
        "aggregate_pref_fit": agg_fit,
    }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")
    print(f"  cv_accuracy: {logistic_fit['cv_accuracy']}")
    print(f"  bt_weight_entropy: {logistic_fit['bt_weight_entropy']}")
    print(f"  effective_dimensionality: {results['effective_dimensionality']}")
    print(f"  mean_per_trait_accuracy: {results['mean_per_trait_accuracy']}")
    print(
        f"  effective_predictive_traits: {results['effective_predictive_traits']}/{results['n_traits']}"
    )
    print(f"  mean_abs_correlation: {results['mean_abs_correlation']}")
    print(f"  aggregate_pref_fit: {results['aggregate_pref_fit']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute extended pref-fit metrics from IRA raw output or bt_scores"
    )
    parser.add_argument(
        "--ira-dir",
        default=None,
        help="Directory containing raw_*.jsonl from IRA audit",
    )
    parser.add_argument(
        "--bt-scores",
        default=None,
        help="Path to bt_scores.jsonl from bt_scoring audit",
    )
    parser.add_argument(
        "--battles", required=True, help="Battles JSONL file (for actual winner labels)"
    )
    parser.add_argument("--output", required=True, help="Output path for metrics JSON")

    args = parser.parse_args()

    if not args.ira_dir and not args.bt_scores:
        parser.error("Must provide either --ira-dir or --bt-scores")

    run_analysis(
        ira_dir=args.ira_dir,
        bt_scores_path=args.bt_scores,
        battles_path=args.battles,
        output_path=args.output,
    )
