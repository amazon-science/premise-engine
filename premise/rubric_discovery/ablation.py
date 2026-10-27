# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Single-pass Ward-linkage baseline for consolidation ablation.

Compares against PReMISE's iterative multi-level consolidation:
  PReMISE:  embed -> cluster -> LLM sub-group -> LLM describe -> repeat
  Baseline: embed -> Ward agglomerative (single pass) -> LLM describe per cluster

Input:   the `extracted_criteria.jsonl` a `premise discover run` wrote
Output:  --target-k baseline rubrics, each with source_count = cluster size

Costs one embedding call per raw criterion plus one description call per cluster,
so the embedding pass dominates and scales with the size of the extraction.

Usage:
    premise discover ablation \
        --extract-file results/discovery/extracted_criteria.jsonl \
        --output results/ablation/ \
        --target-k 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# Reuse helpers from the discovery pipeline (generate_embeddings, generate_consolidated_description, Trait, prompts)
from premise.config import get_config
from premise.rubric_discovery import run as premise
from premise.utils.bedrock import DEFAULT_REGION, get_bedrock_client


def load_raw_criteria(extract_file: Path):
    """Load raw criteria as Trait objects (level 0).

    extracted_criteria.jsonl schema (as written by `premise discover run`):
      {"criterion_name": str, "criterion_description": str,
       "record_id": str, "model_a_score": int, "model_b_score": int}
    """
    traits = []
    with open(extract_file) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            name = rec.get("criterion_name", rec.get("name", ""))
            desc = rec.get("criterion_description", rec.get("description", ""))
            full = f"{name}: {desc}" if name else desc
            if not full.strip() or full.strip() == ":":
                continue  # skip empty/malformed rows
            traits.append(
                premise.Trait(
                    trait_id=f"criterion_{i:05d}",
                    description=full,
                    source_count=1,
                    source_ids=[rec.get("record_id", rec.get("battle_id", ""))],
                )
            )
    return traits


def ward_cluster(embeddings: np.ndarray, k: int) -> list:
    """Single-pass Ward agglomerative clustering, cut at k clusters.

    Returns: list of lists of indices, one per cluster.
    """
    normed = normalize(embeddings)
    # Ward linkage with euclidean metric (cosine with L2-normalized embeddings
    # is equivalent to euclidean up to monotonic transform)
    clustering = AgglomerativeClustering(
        n_clusters=k,
        metric="euclidean",
        linkage="ward",
    )
    labels = clustering.fit_predict(normed)
    clusters = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    return clusters


def describe_cluster(
    client, model_id: str, traits: list, cluster_indices: list
) -> premise.Trait:
    """Call LLM once to describe a whole cluster as one rubric."""
    if len(cluster_indices) == 0:
        return None
    if len(cluster_indices) == 1:
        return traits[cluster_indices[0]]

    # Reuse PReMISE's describe pipeline by constructing a single SubGroup
    # covering the entire cluster (this is exactly what "no LLM sub-grouping" means)
    result = premise.generate_consolidated_description(
        client, model_id, traits, cluster_indices
    )
    if result is None:
        # Fall back to the most-descriptive trait in the cluster
        best = max(cluster_indices, key=lambda i: len(traits[i].description))
        return traits[best]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extract-file",
        required=True,
        type=Path,
        help="Path to extracted_criteria.jsonl",
    )
    ap.add_argument("--output", required=True, type=Path, help="Output directory")
    ap.add_argument(
        "--target-k",
        type=int,
        default=20,
        help="Target cluster count (matches PReMISE target_traits)",
    )
    ap.add_argument("--embedding-model", default=get_config().model("embedding"))
    ap.add_argument("--judge-model", default=get_config().model("judge"))
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Bedrock client (reuse PReMISE's setup)
    client = get_bedrock_client(args.region)

    # Stage 1: load raw criteria as level-0 traits
    print(f"Loading raw criteria from {args.extract_file}")
    traits = load_raw_criteria(args.extract_file)
    print(f"  Loaded {len(traits)} raw criteria")

    # Stage 2a: embed
    cache_path = args.output / "embeddings_ward.npy"
    texts = [t.description for t in traits]
    embeddings = premise.generate_embeddings(
        client,
        texts,
        args.embedding_model,
        cache_path=str(cache_path),
        max_workers=10,
    )

    # Stage 2b: single-pass Ward clustering
    print(f"\nRunning Ward agglomerative clustering (k={args.target_k})...")
    clusters = ward_cluster(embeddings, args.target_k)
    cluster_sizes = [len(c) for c in clusters]
    print(
        f"  Formed {len(clusters)} clusters; sizes: "
        f"min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
        f"mean={np.mean(cluster_sizes):.0f}, median={int(np.median(cluster_sizes))}"
    )

    # Save cluster assignments (for qualitative inspection later)
    cluster_log = args.output / "cluster_assignments.json"
    with open(cluster_log, "w") as f:
        json.dump(
            {
                f"cluster_{i}": {
                    "size": len(c),
                    "trait_ids": [traits[j].trait_id for j in c],
                    "sample_descriptions": [traits[j].description for j in c[:5]],
                }
                for i, c in enumerate(clusters)
            },
            f,
            indent=2,
        )
    print(f"  Saved cluster assignments: {cluster_log}")

    # Stage 3: LLM describes each cluster
    print(f"\nLLM-describing {len(clusters)} clusters...")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rubrics = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for i, cluster_indices in enumerate(clusters):
            futures[
                executor.submit(
                    describe_cluster, client, args.judge_model, traits, cluster_indices
                )
            ] = (i, cluster_indices)
        for future in as_completed(futures):
            i, ci = futures[future]
            r = future.result()
            if r is None:
                print(f"  Cluster {i}: describe failed (size {len(ci)})")
                continue
            # Override source_count/source_ids to reflect cluster size
            source_ids = []
            for j in ci:
                source_ids.extend(traits[j].source_ids)
            r.source_count = len(ci)
            r.source_ids = list(set(source_ids))
            rubrics.append(r)
            print(f"  Cluster {i}: size={len(ci)}, rubric={r.trait_id}")

    # Save final rubric set
    final_file = args.output / "ward_ablation_traits.json"
    output = {
        "method": "ward_agglomerative_single_pass",
        "target_k": args.target_k,
        "n_raw_criteria": len(traits),
        "n_clusters_formed": len(clusters),
        "n_rubrics_output": len(rubrics),
        "rubrics": [
            {
                "trait_id": r.trait_id,
                "description": r.description,
                "source_count": r.source_count,
            }
            for r in rubrics
        ],
    }
    with open(final_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(rubrics)} Ward-baseline rubrics: {final_file}")

    # Summary stats
    print("\n=== Summary ===")
    print(f"Raw criteria: {len(traits)}")
    print(f"Clusters: {len(clusters)} (target k={args.target_k})")
    print(f"Final rubrics: {len(rubrics)}")
    print(f"Mean cluster size: {np.mean(cluster_sizes):.1f}")
    print(f"Cluster size range: {min(cluster_sizes)}-{max(cluster_sizes)}")


if __name__ == "__main__":
    main()
