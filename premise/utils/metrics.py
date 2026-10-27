# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Metrics and score parsing for PReMISE experiments."""

import re
from collections import defaultdict


def parse_score(response_text: str) -> int | None:
    """Parse a numeric score (0-10) from judge response text."""
    if not response_text:
        return None
    for line in reversed(response_text.strip().split("\n")):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score_str = line.split(":", 1)[1].strip()
                if "/" in score_str:
                    score_str = score_str.split("/")[0]
                return int(score_str)
            except (ValueError, IndexError):
                pass
    # Fallback: last integer in response
    numbers = re.findall(r"\b(\d+)\b", response_text)
    if numbers:
        last = int(numbers[-1])
        if 0 <= last <= 10:
            return last
    return None


def parse_verdict(response_text: str) -> str:
    """Parse PASS/FAIL from a judge verdict response."""
    if not response_text:
        return "FAIL"
    for line in reversed(response_text.strip().split("\n")):
        line = line.strip().upper()
        if "VERDICT:" in line:
            if "PASS" in line:
                return "PASS"
            return "FAIL"
    upper = response_text.upper()
    if "VERDICT: PASS" in upper or "VERDICT:PASS" in upper:
        return "PASS"
    return "FAIL"


def score_to_outcome(score: int | None) -> int | None:
    """Convert 0-10 score to outcome: 0=loss, 1=tie, 2=win."""
    if score is None:
        return None
    if score > 5:
        return 2
    elif score < 5:
        return 0
    return 1


def score_to_winner(score: int | None) -> str:
    """Convert 0-10 score to winner label."""
    if score is None:
        return "unknown"
    if score > 5:
        return "candidate"
    elif score < 5:
        return "baseline"
    return "tie"


def compute_krippendorff_alpha(
    judgments: list[list[int | float | None]],
    *,
    level: str = "nominal",
) -> float:
    """Compute Krippendorff's alpha for nominal or ordinal ratings.

    Args:
        judgments: list of lists (judges × items). Values are category labels;
                   None for missing. Each inner list must be the same length.
        level: ``"nominal"`` for exact category agreement or ``"ordinal"`` for
               frequency-ranked squared distance between ordered categories.
    Returns:
        Alpha value (-1 to 1). 1 = perfect agreement, 0 = chance.

    The ordinal distance follows Krippendorff's cumulative-frequency
    definition. This matters for PReMISE's 0--10 judge ratings: a 5-vs-6
    disagreement should count less than a 3-vs-9 disagreement.
    """
    if level not in {"nominal", "ordinal"}:
        raise ValueError("level must be 'nominal' or 'ordinal'")
    if not judgments or not judgments[0]:
        return 0.0
    if any(len(row) != len(judgments[0]) for row in judgments):
        raise ValueError("all judgment rows must contain the same number of items")

    n_judges = len(judgments)
    n_items = len(judgments[0])

    # Filter to items with at least 2 non-None ratings
    valid_items = []
    for i in range(n_items):
        ratings = [
            judgments[j][i] for j in range(n_judges) if judgments[j][i] is not None
        ]
        if len(ratings) >= 2:
            valid_items.append(i)

    if not valid_items:
        return 0.0

    # Collect all categories used
    categories = set()
    for j in range(n_judges):
        for i in valid_items:
            if judgments[j][i] is not None:
                categories.add(judgments[j][i])
    categories = sorted(categories)

    if len(categories) <= 1:
        return 1.0

    freq = defaultdict(int)
    total_ratings = 0
    for j in range(n_judges):
        for i in valid_items:
            value = judgments[j][i]
            if value is not None:
                freq[value] += 1
                total_ratings += 1

    category_index = {category: index for index, category in enumerate(categories)}

    def distance_squared(first, second) -> float:
        if first == second:
            return 0.0
        if level == "nominal":
            return 1.0
        left, right = sorted((category_index[first], category_index[second]))
        between = sum(freq[categories[index]] for index in range(left, right + 1))
        span = between - (freq[first] + freq[second]) / 2.0
        return float(span * span)

    # Observed disagreement (D_o), over ordered within-item rating pairs.
    n_pairs_total = 0
    observed_distance = 0.0
    for i in valid_items:
        ratings = [
            judgments[j][i] for j in range(n_judges) if judgments[j][i] is not None
        ]
        m = len(ratings)
        if m < 2:
            continue
        n_pairs = m * (m - 1)
        n_pairs_total += n_pairs
        for a in range(len(ratings)):
            for b in range(len(ratings)):
                if a != b:
                    observed_distance += distance_squared(ratings[a], ratings[b])

    if n_pairs_total == 0:
        return 0.0

    D_o = observed_distance / n_pairs_total

    # Expected disagreement (D_e), drawing two distinct ratings from the pool.
    expected_distance = 0.0
    for c1 in categories:
        for c2 in categories:
            multiplier = freq[c1] * (freq[c2] - (1 if c1 == c2 else 0))
            expected_distance += multiplier * distance_squared(c1, c2)
    D_e = expected_distance / (total_ratings * (total_ratings - 1))

    if D_e == 0:
        return 1.0

    return 1.0 - (D_o / D_e)
