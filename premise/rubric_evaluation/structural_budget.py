# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Structural Budget for Policy-Level Rubrics — spec + validator.

A candidate rubric passes the budget if:
  - total words <= 150
  - main definition (first paragraph/sentence block) <= 60 words
  - at most 2 anti-gaming/do-not-reward clauses
  - at most 1 applicability gate ("Score 0 if ..." or "Not applicable if ...")
  - at most 3 numbered criteria (detected via "(1)/(2)/(3)" or "1. / 2." patterns)
  - no nested conditionals (forbidden token patterns)
"""

import re
from dataclasses import dataclass, field

BUDGET = {
    "total_max_words": 150,
    "main_def_max_words": 60,
    "anti_gaming_max_count": 2,
    "applicability_gate_max_count": 1,
    "numbered_criteria_max_count": 3,
}

# Regex patterns
_ANTI_GAMING_PATTERNS = [
    r"do not reward",
    r"must not",
    r"fails this trait",
    r"fail(s)? this rubric",
    r"rejected when",
    r"not credit",
    r"no credit",
]
_APPLICABILITY_PATTERNS = [
    r"score\s*0\s*(if|when)",
    r"not applicable (if|when)",
    r"n\/?a (if|when)",
]
_NUMBERED_CRITERIA = re.compile(r"(\(\s*[1-9]\s*\)|(?<![.\d])[1-9]\.\s)")
_NESTING_PATTERNS = [
    r"if\s+[^.]*?then\s+if",  # "if A then if B"
    r"unless\s+[^.]*?unless",  # "unless ... unless"
    r"(?<![.\d])[1-9]\.\s*[^.]*?\(\s*[a-z]\s*\)",  # "1. ... (a)"
]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def extract_main_definition(text: str) -> str:
    """Main definition = text up to the first 'Score 0', 'Specifically:', or numbered-criteria list."""
    if not text:
        return ""
    candidates = [
        re.search(r"score\s*0\s*(if|when)", text, re.IGNORECASE),
        re.search(r"specifically\s*:", text, re.IGNORECASE),
        _NUMBERED_CRITERIA.search(text),
        re.search(r"do not reward|must not|fails this trait", text, re.IGNORECASE),
    ]
    cut = min((m.start() for m in candidates if m), default=len(text))
    return text[:cut].strip()


def count_pattern_matches(text: str, patterns: list[str]) -> int:
    t = text.lower()
    total = 0
    for p in patterns:
        total += len(re.findall(p, t))
    return total


@dataclass
class BudgetReport:
    passes: bool
    total_words: int
    main_def_words: int
    anti_gaming_count: int
    applicability_gate_count: int
    numbered_criteria_count: int
    nesting_violations: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


def validate(description: str, budget: dict | None = None) -> BudgetReport:
    b = budget or BUDGET
    total_w = word_count(description)
    main_def = extract_main_definition(description)
    main_w = word_count(main_def)
    antigaming = count_pattern_matches(description, _ANTI_GAMING_PATTERNS)
    applicability = count_pattern_matches(description, _APPLICABILITY_PATTERNS)
    numbered = len(_NUMBERED_CRITERIA.findall(description or ""))
    nesting = [
        p for p in _NESTING_PATTERNS if re.search(p, description or "", re.IGNORECASE)
    ]

    violations = []
    if total_w > b["total_max_words"]:
        violations.append(f"total_words={total_w} > {b['total_max_words']}")
    if main_w > b["main_def_max_words"]:
        violations.append(f"main_def_words={main_w} > {b['main_def_max_words']}")
    if antigaming > b["anti_gaming_max_count"]:
        violations.append(
            f"anti_gaming_count={antigaming} > {b['anti_gaming_max_count']}"
        )
    if applicability > b["applicability_gate_max_count"]:
        violations.append(
            f"applicability_gate_count={applicability} > {b['applicability_gate_max_count']}"
        )
    if numbered > b["numbered_criteria_max_count"]:
        violations.append(
            f"numbered_criteria_count={numbered} > {b['numbered_criteria_max_count']}"
        )
    if nesting:
        violations.append(f"nested_conditionals={nesting}")

    return BudgetReport(
        passes=not violations,
        total_words=total_w,
        main_def_words=main_w,
        anti_gaming_count=antigaming,
        applicability_gate_count=applicability,
        numbered_criteria_count=numbered,
        nesting_violations=nesting,
        violations=violations,
    )


def budget_prompt_spec() -> str:
    """Budget block to inject into generator prompts."""
    return """You must respect the following complexity budget. Any violation causes rejection:

- Total length of the rubric: at most 150 words.
- Main definition (first paragraph, before any numbered criteria or 'Score 0 if'): at most 60 words, single paragraph, no nested if/then.
- Anti-gaming / do-not-reward clauses: at most 2 total. Each is a flat assertion with a concrete trigger.
- Applicability gate: at most 1 "Score 0 if <flat predicate>" clause.
- Numbered criteria: at most 3 total. Each is a flat predicate. NO nesting ("if ... then if" is forbidden).
- No nested conditionals anywhere in the rubric.

Write a single rubric definition (plain text, no markdown headers, no bullet lists except for the numbered criteria if used)."""


def main() -> None:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(
        description="Check rubrics against the structural budget, or print the "
        "budget text to paste into a generator prompt."
    )
    ap.add_argument(
        "traits",
        nargs="?",
        help="Rubric JSON to check ({rubrics: [...]} or a single {description}). "
        "Omit to print the budget spec instead.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the verdict line per rubric, not the full breakdown",
    )
    args = ap.parse_args()

    if not args.traits:
        print(budget_prompt_spec())
        return

    with open(args.traits) as f:
        data = json.load(f)
    # Every rubric in the file, not just the first — a budget check that silently
    # ignored rubrics[1:] would report a passing file that isn't one.
    if isinstance(data, dict) and "rubrics" in data:
        rubrics = data["rubrics"]
    elif isinstance(data, list):
        rubrics = data
    else:
        rubrics = [data]

    n_fail = 0
    for i, r in enumerate(rubrics):
        desc = r.get("description", "") if isinstance(r, dict) else str(r)
        rid = (r.get("id") if isinstance(r, dict) else None) or f"rubric[{i}]"
        result = validate(desc)
        if result.violations:
            n_fail += 1
        if args.quiet:
            status = "FAIL" if result.violations else "pass"
            print(
                f"{status}  {rid}"
                + (f"  — {'; '.join(result.violations)}" if result.violations else "")
            )
        else:
            print(f"=== {rid} ===")
            print(json.dumps(vars(result), indent=2))

    print(f"\n{len(rubrics) - n_fail}/{len(rubrics)} rubrics within budget")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
