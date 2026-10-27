# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Preference repair (paper row `+pref repair`): rank the trait pool, keep top K.

Repairs the rubric *set* rather than its text — see `repair.py`. One module; the
package exists so it sits symmetrically beside `adversarial/`.
"""
