# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Two rubric repair strategies, as described in the paper.

Both repair a discovered rubric and end in a selection rule; they differ in
whether they rewrite trait text first. One subpackage each:

    preference/   paper row `+pref repair` — repair the rubric *set*
    adversarial/  paper row `+VFR ...`     — repair the rubric *text*
    tree.py       the node/rubric-file structure the adversarial half runs on

1. Preference repair (`preference.repair`) ranks the discovered trait pool by
   per-trait preference fit and keeps the top K. No rewriting, no model calls;
   the trait text is left exactly as discovered.

2. Adversarial repair rewrites each trait under seven anti-gaming operators,
   predicts which variants resist gaming, drops the rest, measures the survivors,
   then takes the Pareto front on (verified fool rate, flip rate):

       seed.py     Stage 0    scaffold the corpus root
       repair.py   Stages 1-2 rewrite the text  ← the only module that writes it
       filters.py  Stages 3-4 prune, then batch-eval the survivors
       pareto.py   Stage 5    paper-settings eval + the domination test
       plots.py    reporting only

   The rewrite is `repair.py` and nothing else authors rubric text: one model call
   both applies the operators and predicts the outcome, and Stage 2 harvests the
   result. Everything after it filters, measures, and copies that text through.

The first optimizes for predictive power, the second for robustness. They start
from the same discovery output, can be run independently, and compose in that
order — the paper's `PReMISE-pref+refined` variant is preference repair followed
by adversarial repair on the survivors.

Note the paper also reports a *constrained* adversarial variant, which bounds how
much preference-fit accuracy a rewritten variant may lose against v0. That
constraint is not implemented here: `filters`' predicted-metric filters and
`pareto`'s domination test consult only vfr/alpha/flip/meta, so this package
reproduces the unconstrained variant.
"""
