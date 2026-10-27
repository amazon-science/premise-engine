# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Bradley-Terry Composite Reward Regression.

Learns rubric weights from human preference data using a BT logistic model:

    P(A > B | p) = sigma( sum_k  w_k * delta_s_k )

where delta_s_k = pairwise_score_k - 5 (recentered 0-10 judge score).

Compares corpora on the same dataset using paired statistical tests
(McNemar, paired bootstrap, likelihood ratio) with shared CV folds.

Supports two scoring sources:
  1. IRA batch outputs (--mode ira, 200 samples, 3 judges averaged)
  2. BT scoring outputs (--mode bt, 2000+ samples, 1 judge)

`--config` names the datasets and, per dataset, the two-or-more rubric corpora to
compare (two at minimum — the point is the paired comparison):

    {
      "my_dataset": {
        "battles": "path/to/battles.jsonl",
        "corpora": [
          {"name": "discovered",
           "scores": "results/bt_scoring/discovered/bt_scores.jsonl",
           "traits": "results/discover/run/traits_top_20.json"},
          {"name": "baseline",
           "scores": "results/bt_scoring/baseline/bt_scores.jsonl",
           "traits": "baseline_rubrics.json"}
        ]
      }
    }

The `scores` path must match the `--mode` you run — a `bt_scores.jsonl` file for
`--mode bt`, an IRA output directory for `--mode ira` — so keep one config per
mode. `--datasets` runs a subset of the config's keys.

Usage:
    # bt_scores.jsonl from `premise audit bt-score` (recommended)
    premise audit bt-regress --mode bt --config datasets_bt.json \
        --output results/bt_regression/

    # IRA output dirs from `premise audit ira` (legacy, 200 samples)
    premise audit bt-regress --mode ira --config datasets_ira.json \
        --output results/bt_regression/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from premise.config import get_config
from premise.utils import load_samples, load_traits
from premise.utils.bedrock import extract_batch_model_response
from premise.utils.metrics import parse_score

# ---------------------------------------------------------------------------
# 1. Parse IRA batch outputs for raw 0-10 scores
# ---------------------------------------------------------------------------


def parse_ira_batch_outputs(ira_dir: str) -> dict:
    """Parse all batch output files in an IRA directory.

    Returns: {(sample_idx, trait_id, judge_name): score}
    """
    batch_dir = os.path.join(ira_dir, "batch_output")
    if not os.path.isdir(batch_dir):
        print(f"  WARN: No batch_output/ in {ira_dir}")
        return {}

    scores = {}
    parsed = 0
    errors = 0

    for judge_dir in sorted(Path(batch_dir).iterdir()):
        if not judge_dir.is_dir():
            continue

        for out_file in judge_dir.glob("*.jsonl.out"):
            with open(out_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        record_id = rec.get("recordId", "")
                        parts = record_id.split("|")
                        if len(parts) < 4:
                            errors += 1
                            continue

                        judge_name = parts[1]
                        trait_id = parts[2]
                        sample_idx = int(parts[3])

                        model_output = rec.get("modelOutput", {})
                        model_id = _judge_name_to_model_id(judge_name)
                        text = extract_batch_model_response(model_output, model_id)
                        score = parse_score(text) if text else None

                        if score is not None:
                            scores[(sample_idx, trait_id, judge_name)] = score
                            parsed += 1
                        else:
                            errors += 1

                    except (json.JSONDecodeError, ValueError):
                        errors += 1

    print(f"  Parsed {parsed} scores ({errors} errors)")
    return scores


def _judge_name_to_model_id(judge_name: str) -> str:
    name_lower = judge_name.lower()
    if "deepseek" in name_lower:
        return "deepseek"
    if "mistral" in name_lower:
        return "mistral"
    if "qwen" in name_lower:
        return "qwen"
    return "deepseek"


# ---------------------------------------------------------------------------
# 1b. Parse BT scoring outputs (from `premise audit bt-score`)
# ---------------------------------------------------------------------------


def parse_bt_scores(scores_path: str) -> dict:
    """Parse bt_scores.jsonl into same format as IRA outputs.

    Returns: {(sample_idx, trait_id, judge_name): score}
    """
    scores = {}
    parsed = 0
    errors = 0

    with open(scores_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                sample_idx = rec["sample_idx"]
                trait_id = rec["trait_id"]
                judge = rec.get("judge", "deepseek_v3")
                score = rec["score"]
                scores[(sample_idx, trait_id, judge)] = score
                parsed += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                errors += 1

    print(f"  Parsed {parsed} scores ({errors} errors) from {scores_path}")
    return scores


# ---------------------------------------------------------------------------
# 2. Build feature matrix
# ---------------------------------------------------------------------------

WINNER_MAP = {
    "model_a": 1.0,
    "model_b": 0.0,
    "tie": 0.5,
    "both_bad": 0.5,
}


def build_dataset(scores: dict, samples: list, trait_ids: list) -> tuple:
    """Build feature matrix X and label vector y.

    Returns: (X, y, sample_ids, valid_indices)
        X: (n, K) feature matrix, delta_s recentered at 0
        y: (n,) label vector
        sample_ids: battle IDs
        valid_indices: original sample indices (for aligning across corpora)
    """
    sample_indices = sorted({idx for (idx, _, _) in scores})
    if not sample_indices:
        return np.array([]), np.array([]), [], []

    K = len(trait_ids)
    X_all = np.full((len(sample_indices), K), np.nan)
    y_all = np.full(len(sample_indices), np.nan)
    ids_all = []
    idx_all = []

    for row, s_idx in enumerate(sample_indices):
        if s_idx >= len(samples):
            continue

        battle = samples[s_idx]
        winner = battle.get("winner", "")
        if winner not in WINNER_MAP:
            continue

        y_all[row] = WINNER_MAP[winner]
        ids_all.append(battle.get("id", f"sample_{s_idx}"))
        idx_all.append(s_idx)

        for col, trait_id in enumerate(trait_ids):
            judge_scores = [
                v for (si, ti, _), v in scores.items() if si == s_idx and ti == trait_id
            ]
            if judge_scores:
                X_all[row, col] = np.mean(judge_scores) - 5.0

    # Drop incomplete rows
    valid = ~np.isnan(y_all) & ~np.any(np.isnan(X_all), axis=1)
    return (
        X_all[valid],
        y_all[valid],
        [ids_all[i] for i in range(len(ids_all)) if valid[i]],
        [idx_all[i] for i in range(len(idx_all)) if valid[i]],
    )


# ---------------------------------------------------------------------------
# 3. Paired CV with out-of-fold predictions
# ---------------------------------------------------------------------------


def paired_cv(X, y, n_folds=5, alpha=1.0, random_state=42):
    """Run CV, return per-sample out-of-fold predictions and probabilities.

    Returns dict with:
        oof_preds: (n,) hard predictions
        oof_probs: (n,) P(A wins) probabilities
        fold_accs: list of per-fold accuracies
        fold_lls: list of per-fold log-losses
        weights_per_fold: list of weight vectors
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    # Remove ties for classification
    nontie = y != 0.5
    X_nt, y_nt = X[nontie], y[nontie]
    n = len(y_nt)

    if n < n_folds * 2:
        return None

    oof_preds = np.full(n, np.nan)
    oof_probs = np.full(n, np.nan)
    fold_accs = []
    fold_lls = []
    weights_per_fold = []

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X_nt, y_nt):
        X_tr, X_te = X_nt[train_idx], X_nt[test_idx]
        y_tr, y_te = y_nt[train_idx], y_nt[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        mdl = LogisticRegression(
            C=1.0 / alpha,
            penalty="l2",
            fit_intercept=False,
            solver="lbfgs",
            max_iter=1000,
        )
        mdl.fit(X_tr_s, y_tr)

        preds = mdl.predict(X_te_s)
        probs = mdl.predict_proba(X_te_s)
        # P(class=1)
        if mdl.classes_[1] == 1.0:
            prob_a = probs[:, 1]
        else:
            prob_a = probs[:, 0]

        oof_preds[test_idx] = preds
        oof_probs[test_idx] = prob_a

        fold_accs.append(accuracy_score(y_te, preds))
        fold_lls.append(log_loss(y_te, probs, labels=[0.0, 1.0]))
        weights_per_fold.append(mdl.coef_[0] / scaler.scale_)

    return {
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "y_true": y_nt,
        "fold_accs": fold_accs,
        "fold_lls": fold_lls,
        "weights_per_fold": weights_per_fold,
        "n_nontie": n,
    }


# ---------------------------------------------------------------------------
# 4. Statistical significance tests
# ---------------------------------------------------------------------------


def mcnemar_test(y_true, preds_a, preds_b):
    """McNemar's test comparing two classifiers on the same samples.

    Tests whether the disagreement pattern is symmetric.
    Returns: (chi2, p_value, contingency_table)
    """
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true

    # b: A right, B wrong; c: A wrong, B right
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    # Use exact binomial if b+c < 25
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0, {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c}

    if n_discordant < 25:
        # Exact binomial: P(X >= max(b,c)) under H0: p=0.5
        p_value = scipy_stats.binomtest(max(b, c), n_discordant, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
        p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)

    return (
        float((b - c) ** 2 / (b + c)) if (b + c) > 0 else 0.0,
        float(p_value),
        {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c},
    )


def paired_bootstrap(metric_a, metric_b, n_bootstrap=10000, seed=42):
    """Paired bootstrap test for difference in per-sample metrics.

    metric_a, metric_b: arrays of per-sample scores (e.g., 1=correct, 0=wrong)
    Tests H0: mean(metric_a) = mean(metric_b)

    Returns: (observed_diff, ci_lower, ci_upper, p_value)
        where observed_diff = mean(metric_a) - mean(metric_b)
        p_value is two-sided.
    """
    rng = np.random.RandomState(seed)
    n = len(metric_a)
    diffs = metric_a - metric_b
    observed = np.mean(diffs)

    boot_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[idx])

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Two-sided p-value: fraction of bootstrap diffs on the other side of 0
    if observed >= 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    else:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return float(observed), ci_lower, ci_upper, p_value


def likelihood_ratio_test(ll_a, ll_b, df_diff):
    """Likelihood ratio test comparing two nested models.

    ll_a, ll_b: log-likelihoods (larger = better).
    df_diff: difference in degrees of freedom.
    """
    lr_stat = -2 * (ll_b - ll_a)  # positive if A is better
    p_value = 1 - scipy_stats.chi2.cdf(abs(lr_stat), df=df_diff)
    return float(lr_stat), float(p_value)


def compute_log_likelihood(y_true, probs):
    """Compute total log-likelihood."""
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return float(np.sum(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)))


def compare_two_corpora(name_a, cv_a, name_b, cv_b):
    """Run all pairwise statistical tests between two corpora.

    Returns: dict with test results.
    """
    y = cv_a["y_true"]
    assert np.array_equal(y, cv_b["y_true"]), "Labels must match"

    preds_a = cv_a["oof_preds"]
    preds_b = cv_b["oof_preds"]
    probs_a = cv_a["oof_probs"]
    probs_b = cv_b["oof_probs"]

    # 1. McNemar's test on classification accuracy
    chi2, p_mcn, contingency = mcnemar_test(y, preds_a, preds_b)
    acc_a = float(np.mean(preds_a == y))
    acc_b = float(np.mean(preds_b == y))

    # 2. Paired bootstrap on accuracy (per-sample correct/incorrect)
    correct_a = (preds_a == y).astype(float)
    correct_b = (preds_b == y).astype(float)
    diff_acc, ci_acc_lo, ci_acc_hi, p_boot_acc = paired_bootstrap(correct_a, correct_b)

    # 3. Paired bootstrap on log-loss (per-sample)
    eps = 1e-15
    probs_a_c = np.clip(probs_a, eps, 1 - eps)
    probs_b_c = np.clip(probs_b, eps, 1 - eps)
    ll_per_sample_a = y * np.log(probs_a_c) + (1 - y) * np.log(1 - probs_a_c)
    ll_per_sample_b = y * np.log(probs_b_c) + (1 - y) * np.log(1 - probs_b_c)
    # Negate because we want "lower is better" for loss
    diff_ll, ci_ll_lo, ci_ll_hi, p_boot_ll = paired_bootstrap(
        -ll_per_sample_a, -ll_per_sample_b
    )

    # 4. Likelihood ratio test
    total_ll_a = compute_log_likelihood(y, probs_a)
    total_ll_b = compute_log_likelihood(y, probs_b)
    df_a = len(cv_a["weights_per_fold"][0])
    df_b = len(cv_b["weights_per_fold"][0])
    lr_stat, p_lr = likelihood_ratio_test(total_ll_a, total_ll_b, abs(df_a - df_b) or 1)

    return {
        "corpus_a": name_a,
        "corpus_b": name_b,
        "n_samples": len(y),
        "accuracy": {
            "a": acc_a,
            "b": acc_b,
            "diff_a_minus_b": diff_acc,
            "ci_95": [ci_acc_lo, ci_acc_hi],
            "p_bootstrap": p_boot_acc,
            "winner": name_a if diff_acc > 0 else name_b if diff_acc < 0 else "tie",
        },
        "mcnemar": {
            "chi2": chi2,
            "p_value": p_mcn,
            "a_right_b_wrong": contingency["b_a_right_b_wrong"],
            "a_wrong_b_right": contingency["c_a_wrong_b_right"],
        },
        "log_loss": {
            "a": float(-np.mean(ll_per_sample_a)),
            "b": float(-np.mean(ll_per_sample_b)),
            "diff_a_minus_b": diff_ll,  # positive = A has higher loss (worse)
            "ci_95": [ci_ll_lo, ci_ll_hi],
            "p_bootstrap": p_boot_ll,
            "winner": name_a if diff_ll < 0 else name_b if diff_ll > 0 else "tie",
        },
        "likelihood_ratio": {
            "lr_stat": lr_stat,
            "p_value": p_lr,
            "ll_a": total_ll_a,
            "ll_b": total_ll_b,
            "df_a": df_a,
            "df_b": df_b,
        },
    }


# ---------------------------------------------------------------------------
# 5. Full pipeline for one dataset
# ---------------------------------------------------------------------------


def load_corpus_data(source, battles_file, traits_file, name, mode="ira"):
    """Load scores and build feature matrix for one corpus.

    Returns: (X, y, valid_idx, trait_ids) or (None, None, None, None)
    """
    print(f"\n  [{name}]")
    print(f"    Source: {source} (mode={mode})")
    print(f"    Traits: {traits_file}")

    traits = load_traits(traits_file)
    trait_ids = [t["id"] for t in traits]

    if mode == "bt":
        scores = parse_bt_scores(source)
    else:
        scores = parse_ira_batch_outputs(source)
    if not scores:
        return None, None, None, None

    sample_indices = sorted({idx for (idx, _, _) in scores})
    max_idx = max(sample_indices) + 1

    samples = load_samples(battles_file, max_samples=max_idx)
    X, y, sample_ids, valid_idx = build_dataset(scores, samples, trait_ids)
    print(f"    Dataset: {X.shape[0]} x {X.shape[1]}")

    n_a = int((y == 1.0).sum())
    n_b = int((y == 0.0).sum())
    n_tie = int((y == 0.5).sum())
    print(f"    Labels: A={n_a} B={n_b} tie={n_tie}")

    if X.shape[0] < 20:
        print("    SKIP: too few samples")
        return None, None, None, None

    unique_labels = set(y[y != 0.5])
    if len(unique_labels) < 2:
        print("    SKIP: single class")
        return None, None, None, None

    return X, y, valid_idx, trait_ids


def fit_and_cv(X, y, trait_ids, name):
    """Run full model fit + paired CV. Returns (cv_result, meta)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Full model fit for weights
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X[y != 0.5])
    y_nt = y[y != 0.5]
    mdl = LogisticRegression(
        C=1.0, penalty="l2", fit_intercept=False, solver="lbfgs", max_iter=1000
    )
    mdl.fit(X_s, y_nt)
    full_weights = mdl.coef_[0] / scaler.scale_

    # Paired CV
    cv = paired_cv(X, y, n_folds=5, alpha=1.0, random_state=42)
    if cv is None:
        print(f"    SKIP: CV failed for {name}")
        return None, None

    acc = float(np.mean(cv["fold_accs"]))
    ll = float(np.mean(cv["fold_lls"]))
    print(f"    [{name}] CV acc: {acc:.3f} +/- {np.std(cv['fold_accs']):.3f}")
    print(f"    [{name}] CV ll:  {ll:.3f} +/- {np.std(cv['fold_lls']):.3f}")

    n_a = int((y == 1.0).sum())
    n_b = int((y == 0.0).sum())
    n_tie = int((y == 0.5).sum())

    meta = {
        "corpus": name,
        "n_traits": len(trait_ids),
        "trait_ids": trait_ids,
        "n_samples": int(X.shape[0]),
        "n_a": n_a,
        "n_b": n_b,
        "n_tie": n_tie,
        "cv_accuracy_mean": acc,
        "cv_accuracy_std": float(np.std(cv["fold_accs"])),
        "cv_logloss_mean": ll,
        "cv_logloss_std": float(np.std(cv["fold_lls"])),
        "weights": sorted(
            [{"trait": tid, "w": float(w)} for tid, w in zip(trait_ids, full_weights)],
            key=lambda x: abs(x["w"]),
            reverse=True,
        ),
    }

    return cv, meta


def run_dataset(dataset_name, battles_file, corpora_cfg, output_dir, mode="ira"):
    """Run BT regression + pairwise comparisons for one dataset.

    corpora_cfg: list of (name, source_path, traits_file)
        source_path is an IRA dir (mode=ira) or bt_scores.jsonl path (mode=bt)
    """
    print(f"\n{'=' * 70}")
    print(f"DATASET: {dataset_name}")
    print(f"Battles: {battles_file}")
    print(f"Mode: {mode}")
    print(f"{'=' * 70}")

    # Phase 1: Load all corpora data
    corpus_data = {}  # name -> (X, y, valid_idx, trait_ids)
    for name, source_path, traits_file in corpora_cfg:
        result = load_corpus_data(
            source_path, battles_file, traits_file, name, mode=mode
        )
        if result[0] is not None:
            corpus_data[name] = result

    if len(corpus_data) < 2:
        print(f"\n  Only {len(corpus_data)} corpus(a) loaded -- skipping")
        return {"dataset": dataset_name, "corpora": {}, "comparisons": []}

    # Phase 2: Find common sample indices across all corpora
    all_valid_sets = [set(data[2]) for data in corpus_data.values()]
    common_idx = sorted(set.intersection(*all_valid_sets))
    print(f"\n  Common samples across {len(corpus_data)} corpora: {len(common_idx)}")

    # Phase 3: Align each corpus to common samples, then run CV
    cv_results = {}
    meta_results = {}

    for name, (X_full, y_full, valid_idx, trait_ids) in corpus_data.items():
        # Map valid_idx to row positions
        idx_to_row = {vi: row for row, vi in enumerate(valid_idx)}
        common_rows = [idx_to_row[ci] for ci in common_idx if ci in idx_to_row]
        X_aligned = X_full[common_rows]
        y_aligned = y_full[common_rows]
        print(f"    [{name}] Aligned: {X_aligned.shape[0]} samples")

        cv, meta = fit_and_cv(X_aligned, y_aligned, trait_ids, name)
        if cv is not None:
            cv_results[name] = cv
            meta_results[name] = meta

    if len(cv_results) < 2:
        print(f"\n  Only {len(cv_results)} corpus(a) succeeded -- skipping comparisons")
        return {"dataset": dataset_name, "corpora": meta_results, "comparisons": []}

    # Pairwise comparisons
    names = sorted(cv_results.keys())
    comparisons = []

    print(f"\n{'=' * 70}")
    print(f"PAIRWISE COMPARISONS: {dataset_name}")
    print(f"{'=' * 70}")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            print(f"\n  --- {na} vs {nb} ---")
            comp = compare_two_corpora(na, cv_results[na], nb, cv_results[nb])
            comparisons.append(comp)

            a_acc = comp["accuracy"]["a"]
            b_acc = comp["accuracy"]["b"]
            diff = comp["accuracy"]["diff_a_minus_b"]
            ci = comp["accuracy"]["ci_95"]
            p_boot = comp["accuracy"]["p_bootstrap"]
            p_mcn = comp["mcnemar"]["p_value"]

            print(f"    Accuracy: {na}={a_acc:.3f}  {nb}={b_acc:.3f}  diff={diff:+.3f}")
            print(f"    Bootstrap 95% CI: [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p_boot:.4f}")
            print(
                f"    McNemar: chi2={comp['mcnemar']['chi2']:.2f}  "
                f"p={p_mcn:.4f}  "
                f"({na}+/{nb}-={comp['mcnemar']['a_right_b_wrong']}, "
                f"{na}-/{nb}+={comp['mcnemar']['a_wrong_b_right']})"
            )

            a_ll = comp["log_loss"]["a"]
            b_ll = comp["log_loss"]["b"]
            ll_diff = comp["log_loss"]["diff_a_minus_b"]
            ll_ci = comp["log_loss"]["ci_95"]
            ll_p = comp["log_loss"]["p_bootstrap"]
            print(
                f"    Log-loss: {na}={a_ll:.4f}  {nb}={b_ll:.4f}  diff={ll_diff:+.4f}"
            )
            print(
                f"    Bootstrap 95% CI: [{ll_ci[0]:+.4f}, {ll_ci[1]:+.4f}]  "
                f"p={ll_p:.4f}"
            )

            lr = comp["likelihood_ratio"]
            print(
                f"    LR test: stat={lr['lr_stat']:.2f}  p={lr['p_value']:.4f}  "
                f"(ll_{na}={lr['ll_a']:.1f}, ll_{nb}={lr['ll_b']:.1f})"
            )

            sig = (
                "***"
                if p_boot < 0.001
                else "**"
                if p_boot < 0.01
                else "*"
                if p_boot < 0.05
                else "ns"
            )
            winner = comp["accuracy"]["winner"]
            print(f"    => Winner (accuracy): {winner} ({sig})")

    return {
        "dataset": dataset_name,
        "corpora": {k: v for k, v in meta_results.items()},
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------


def print_summary_table(all_dataset_results):
    """Print a concise comparison table across all datasets."""
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: BT Regression — Corpus Comparison")
    print(f"{'=' * 80}")

    # Header
    print(
        f"\n{'Dataset':<18} {'Corpus':<14} {'K':>3} {'N':>4} {'CV Acc':>8} {'CV LL':>8}"
    )
    print("-" * 65)

    for dr in all_dataset_results:
        ds = dr["dataset"]
        for name, meta in sorted(dr["corpora"].items()):
            acc = meta["cv_accuracy_mean"]
            ll = meta["cv_logloss_mean"]
            print(
                f"{ds:<18} {name:<14} {meta['n_traits']:>3} "
                f"{meta['n_samples']:>4} "
                f"{acc:>7.3f} {ll:>8.4f}"
            )

    # Pairwise significance
    print(f"\n\n{'=' * 80}")
    print("PAIRWISE SIGNIFICANCE (accuracy, paired bootstrap)")
    print(f"{'=' * 80}")
    print(
        f"\n{'Dataset':<18} {'A':<14} {'B':<14} "
        f"{'Acc_A':>6} {'Acc_B':>6} {'Diff':>7} "
        f"{'95% CI':>17} {'p':>8} {'Sig':>4}"
    )
    print("-" * 100)

    for dr in all_dataset_results:
        ds = dr["dataset"]
        for comp in dr["comparisons"]:
            na = comp["corpus_a"]
            nb = comp["corpus_b"]
            a_acc = comp["accuracy"]["a"]
            b_acc = comp["accuracy"]["b"]
            diff = comp["accuracy"]["diff_a_minus_b"]
            ci = comp["accuracy"]["ci_95"]
            p = comp["accuracy"]["p_bootstrap"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(
                f"{ds:<18} {na:<14} {nb:<14} "
                f"{a_acc:>6.3f} {b_acc:>6.3f} {diff:>+7.3f} "
                f"[{ci[0]:>+.3f},{ci[1]:>+.3f}] "
                f"{p:>8.4f} {sig:>4}"
            )

    # McNemar
    print(
        f"\n\n{'Dataset':<18} {'A':<14} {'B':<14} "
        f"{'A+/B-':>5} {'A-/B+':>5} {'p_McN':>8} {'Sig':>4}"
    )
    print("-" * 80)

    for dr in all_dataset_results:
        ds = dr["dataset"]
        for comp in dr["comparisons"]:
            na = comp["corpus_a"]
            nb = comp["corpus_b"]
            ab = comp["mcnemar"]["a_right_b_wrong"]
            ba = comp["mcnemar"]["a_wrong_b_right"]
            p = comp["mcnemar"]["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{ds:<18} {na:<14} {nb:<14} {ab:>5} {ba:>5} {p:>8.4f} {sig:>4}")

    # Log-loss
    print(
        f"\n\n{'Dataset':<18} {'A':<14} {'B':<14} "
        f"{'LL_A':>8} {'LL_B':>8} {'Diff':>8} "
        f"{'p':>8} {'Sig':>4}"
    )
    print("-" * 95)

    for dr in all_dataset_results:
        ds = dr["dataset"]
        for comp in dr["comparisons"]:
            na = comp["corpus_a"]
            nb = comp["corpus_b"]
            a_ll = comp["log_loss"]["a"]
            b_ll = comp["log_loss"]["b"]
            diff = comp["log_loss"]["diff_a_minus_b"]
            p = comp["log_loss"]["p_bootstrap"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(
                f"{ds:<18} {na:<14} {nb:<14} "
                f"{a_ll:>8.4f} {b_ll:>8.4f} {diff:>+8.4f} "
                f"{p:>8.4f} {sig:>4}"
            )


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------


def load_dataset_config(config_file: str) -> dict:
    """Read the datasets and corpora to compare from a JSON config.

    Shape — one entry per preference dataset, each listing the rubric corpora to
    compare on it:

        {
          "my_dataset": {
            "battles": "path/to/battles.jsonl",
            "corpora": [
              {"name": "discovered", "scores": "...", "traits": "..."},
              {"name": "baseline",   "scores": "...", "traits": "..."}
            ]
          }
        }

    `scores` is whatever the chosen --mode reads: an IRA output directory for
    `--mode ira`, or a bt_scores.jsonl path for `--mode bt`. Since the two modes
    need different score files, keep a config per mode rather than one for both.

    At least two corpora per dataset are required — the whole point is the paired
    comparison between them, and `run_dataset` skips any dataset with fewer.
    """
    cfg = json.loads(Path(config_file).read_text())
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"{config_file}: expected a non-empty JSON object of datasets")

    out = {}
    for ds_name, entry in cfg.items():
        if not entry.get("battles"):
            raise ValueError(f"{config_file}: dataset '{ds_name}' has no 'battles'")
        corpora = entry.get("corpora") or []
        if len(corpora) < 2:
            raise ValueError(
                f"{config_file}: dataset '{ds_name}' lists {len(corpora)} corpora; "
                "at least 2 are needed for a paired comparison"
            )
        parsed = []
        for i, c in enumerate(corpora):
            missing = [k for k in ("name", "scores", "traits") if not c.get(k)]
            if missing:
                raise ValueError(
                    f"{config_file}: dataset '{ds_name}' corpus {i} is missing "
                    f"{', '.join(missing)} (each needs name, scores, traits)"
                )
            parsed.append((c["name"], c["scores"], c["traits"]))
        names = [p[0] for p in parsed]
        if len(set(names)) != len(names):
            raise ValueError(
                f"{config_file}: dataset '{ds_name}' has duplicate corpus names — {names}"
            )
        out[ds_name] = {"battles": entry["battles"], "corpora": parsed}
    return out


def main():
    parser = argparse.ArgumentParser(
        description="BT composite reward regression with paired significance tests"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config of datasets and the corpora to compare on each: "
        '{"<dataset>": {"battles": <battles.jsonl>, "corpora": '
        '[{"name":, "scores":, "traits":}, ...]}}, two corpora minimum. '
        "`scores` must match --mode. See the module docstring for a full example.",
    )
    parser.add_argument(
        "--mode",
        choices=["ira", "bt"],
        default="ira",
        help="Score source: ira (200 samples, 3 judges) or bt (2000+ samples, 1 judge)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of the config's datasets to run (default: all of them)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            get_config().runtime("output_root"), "audit", "bt-regress"
        ),
        help="Output directory (default: runtime.output_root/audit/bt-regress)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="L2 regularization strength"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000, help="Bootstrap iterations"
    )
    args = parser.parse_args()

    ds_map = load_dataset_config(args.config)
    if args.datasets is None:
        args.datasets = list(ds_map.keys())

    os.makedirs(args.output, exist_ok=True)
    all_results = []

    for ds_name in args.datasets:
        if ds_name not in ds_map:
            print(f"Unknown dataset: {ds_name} (not in {args.config})")
            continue
        cfg = ds_map[ds_name]
        result = run_dataset(
            ds_name, cfg["battles"], cfg["corpora"], args.output, mode=args.mode
        )
        all_results.append(result)

    # Summary table
    print_summary_table(all_results)

    # Save
    out_file = os.path.join(args.output, "bt_regression_comparison.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
