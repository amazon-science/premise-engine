# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Adversarial repair (paper row `+VFR unconstrained`): rewrite, then select.

Repairs the rubric *text*, in stage order:

    workflow.py guided resumable seed -> predict -> merge -> materialize path
    seed.py     Stage 0    scaffold the corpus root from a v0 traits file
    repair.py   Stages 1-2 apply the operators + predict, then harvest the text
    filters.py  Stages 3-4 prune on predicted metrics, batch-eval the survivors
    pareto.py   Stage 5    paper-settings eval + the (vfr, flip) domination test
    operators.py           the seven anti-gaming operators, single-sourced
    plots.py               frontier plots; reporting only

`repair.py` remains the only module that asks a model to author rubric text.
The guided workflow materializes reviewed variants from those authored nodes;
advanced stages can still run separately and coordinate through `..tree`.
"""
