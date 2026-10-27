# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Verifier prompt versions used by the adversarial evaluation.

The paper reports v0 (original) and v13 (optimized, high-recall) results.
`PROMPTS["v0"]` matches the original verifier; `PROMPTS["v13"]` is the optimized
default exposed elsewhere as `premise.utils.prompts.VERIFIER_PROMPT`.
"""

from premise.utils.prompts import VERIFIER_PROMPT, VERIFIER_PROMPT_V0

PROMPTS: dict[str, str] = {
    "v0": VERIFIER_PROMPT_V0,
    "v13": VERIFIER_PROMPT,
}
