# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""The refinement tree: one node per rubric variant, plus the rubric files on disk.

Both repair strategies' adversarial half reads and writes this structure, so it
lives at the package root rather than inside `adversarial/`. A node is keyed by
`canonical_node_id(trait, ops)` — the trait it came from and the sorted operator
subset applied to it — which is what makes a variant addressable across stages
that run days apart in separate processes.

`tree.json` is the single source of truth; `atomic_save` writes it through a
temp file and rename so a crash mid-write cannot truncate it. Rubric text goes
to `rubrics/combo/<trait>__<ops>.json` via `write_rubric_file`, and the node
records the path rather than the text.

Extracted from `pipeline.py` when the stages moved under `adversarial/`; the
functions are unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime

# Default corpus root for the adversarial-repair stages. Every stage takes
# --root, so this only decides where a run lands when none is given; it sits
# under experiments/, which .gitignore excludes.
ROOT = "experiments/adversarial_repair"


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------


def now_iso():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_node_id(trait, ops):
    if not ops:
        return f"{trait}__v0"
    return f"{trait}__" + "+".join(sorted(ops))


def load_tree(path):
    with open(path) as f:
        return json.load(f)


def atomic_save(tree, path):
    tmp = path + ".tmp"
    tree["updated_at"] = now_iso()
    with open(tmp, "w") as f:
        json.dump(tree, f, indent=2)
    os.replace(tmp, path)


def get_node(tree, nid):
    for n in tree["nodes"]:
        if n["id"] == nid:
            return n
    return None


def upsert(tree, node):
    for i, n in enumerate(tree["nodes"]):
        if n["id"] == node["id"]:
            tree["nodes"][i] = {**n, **node, "updated_at": now_iso()}
            return tree["nodes"][i]
    node.setdefault("created_at", now_iso())
    node["updated_at"] = now_iso()
    tree["nodes"].append(node)
    return node


def write_rubric_file(root, trait, ops, description, v0_data=None):
    """Write a standalone rubric JSON for a refined node, returning its path."""
    out_dir = os.path.join(root, "rubrics", "combo")
    os.makedirs(out_dir, exist_ok=True)
    name = "+".join(sorted(ops)) or "v0"
    out_path = os.path.join(out_dir, f"{trait}__{name}.json")
    if v0_data and v0_data.get("rubrics"):
        r0 = v0_data["rubrics"][0]
        corpus = r0.get("corpus", "discovered")
        category = r0.get("category", "refined")
    else:
        corpus, category = "discovered", "refined"
    data = {
        "schema_version": "1.0.0",
        "trait_id": trait,
        "ops": sorted(ops),
        "rubrics": [
            {
                "id": trait,
                "corpus": corpus,
                "category": category,
                "description": description,
            }
        ],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def real_eval_nodes(tree, trait=None, include_rejected=True):
    """Nodes that have real metrics (not predicted-only)."""
    res = []
    for n in tree["nodes"]:
        if trait and n["trait_id"] != trait:
            continue
        if n["metrics"].get("meta") is None:
            continue
        if not include_rejected and n.get("status", "").startswith("rejected"):
            continue
        res.append(n)
    return res


def _run_tool(cmd, log_path):
    """Run a command, log stdout+stderr to log_path. Returns return code."""
    with open(log_path, "w") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
