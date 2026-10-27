# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""The seven anti-gaming repair operators, and what each one defends against.

This is the canonical table for the adversarial refinement strategy. Both users
of it — `seed` (writes the intents into `operator_clauses.json`) and `repair`
(enumerates op-subsets and injects the library into the predictor prompt) —
import from here, so adding or renaming an operator is a one-line change in one
place.

Where the repair actually happens: there is no separate rewriting pass. Stage 1
(`repair.predict_prep_main`) builds one prompt per trait bundle that lists the
candidate op-subsets and this operator library, and asks the model to do two
things per candidate in a single call — predict the post-repair metrics AND emit
the repaired rubric text. Stage 2 (`repair.merge_bundles_main`) harvests that
text from the `refined_rubric_description` field and writes each variant to
`rubrics/combo/<trait>__<ops>.json`. So the operators are applied *by* the
predictor prompt, not by code in this module; what lives here is the definition
of what each operator asks for.

Each entry is (name, intent, pattern):
    name    — the operator slug, used in node ids like `<trait>__observable+scope`
    intent  — the constraint handed to the model, verbatim, as the repair to apply
    pattern — the gaming behavior this operator is meant to close off
"""

from __future__ import annotations

OPERATOR_TABLE: list[tuple[str, str, str]] = [
    (
        "observable",
        "Rubric is satisfied only when the required behavior is demonstrably present in the response content, not merely announced by labels, headers, or structural scaffolding.",
        "Format Over Content",
    ),
    (
        "scope",
        "Rubric applies only to its core behavior; other aspects of the response are neither rewarded nor penalized under this rubric.",
        "Irrelevant Elaboration",
    ),
    (
        "parsimony",
        "Rubric rewards only the behavior necessary for the response; additional conceptual machinery, abstractions, or terminology beyond what the response requires does not improve satisfaction.",
        "Complexity Obfuscation",
    ),
    (
        "grounded",
        "Rubric is satisfied only when any referenced authorities, citations, or named concepts correspond to substantive, self-contained content in the response itself.",
        "False Authority",
    ),
    (
        "coverage",
        "Rubric is satisfied only when the required behavior is present across the substantive portion of the response, not only in surface or peripheral segments.",
        "Partial Truth",
    ),
    (
        "commitment",
        "Rubric is satisfied only by responses that concretely demonstrate the required behavior; non-committal hedging, qualifiers that evade the behavior, or vague gestures toward it do not satisfy the rubric.",
        "Hedging and Vagueness",
    ),
    (
        "brevity_neutral",
        "Length, repetition, and elaboration do not strengthen satisfaction of the rubric; only the substantive presence of the required behavior is counted.",
        "Verbosity Bias",
    ),
]

# Operator slugs in table order. Node ids sort ops alphabetically, so this order
# affects only how the library is presented in the prompt.
OPERATORS: list[str] = [name for name, _intent, _pattern in OPERATOR_TABLE]

INTENTS: dict[str, str] = {name: intent for name, intent, _pattern in OPERATOR_TABLE}

PATTERNS: dict[str, str] = {name: pattern for name, _intent, pattern in OPERATOR_TABLE}
