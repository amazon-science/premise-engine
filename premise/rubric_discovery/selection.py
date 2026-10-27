# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Final top-K selection strategies for the discovery pipeline.

Four strategies, dispatched via --final-topk-selection:

- source_count: top-K of the input pool by source_count (current behavior).
- logdet: greedy log-determinant volume maximization over normalized
          embeddings, with min-increment / patience early-stop. Picks the
          trait that most expands the volume the selected set spans, so
          near-duplicates stop earning their slot.
- llm:  judge prompt that selects exactly K with paraphrastic-merge
        identification; merged_into annotations and summed source_counts
        on kept traits.
- mmr:  genericity filter plus greedy maximal-marginal-relevance over
        log-scaled source_count.

All four return:
    selected_ids:   List[str]            -- trait_ids picked, in rank order.
    annotations:    Dict[trait_id, dict] -- per-trait keys to merge into the
                                            saved JSON (selected_by_strategy
                                            plus strategy-specific keys).
    extras:         Dict[str, Any]       -- strategy-specific artifacts
                                            (logdet_curve, llm_merges, ...).
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from premise.config import get_config
from premise.utils import DEFAULT_REGION, get_bedrock_client, invoke_model

if TYPE_CHECKING:
    # Import-time circular: run.py imports this module. Only needed for the
    # List["Trait"] annotations below.
    from premise.rubric_discovery.run import Trait


# ============================================================
# Strategy: source_count (current behavior, no LLM/embeddings)
# ============================================================


def select_by_source_count(
    traits: list[Trait],
    target_traits: int,
) -> tuple[list[str], dict[str, dict], dict[str, Any]]:
    """Top-K by source_count. Identical to the pre-existing slice."""
    sorted_traits = sorted(traits, key=lambda t: -t.source_count)
    selected = sorted_traits[:target_traits]
    selected_ids = [t.trait_id for t in selected]
    selected_set = set(selected_ids)
    annotations = {
        t.trait_id: {"selected_by_strategy": t.trait_id in selected_set} for t in traits
    }
    return selected_ids, annotations, {"strategy": "source_count"}


# ============================================================
# Strategy: logdet (greedy log-determinant volume maximization)
# ============================================================


def _log_volume(X: np.ndarray, eps: float = 0.5) -> float:
    """½ log det(I + (1/(ε² n)) XᵀX), via SVD with 95% energy retention.

    X has shape [n, d] (n traits, d-dim normalized embeddings). The value
    grows with the volume the rows span, so it rises when a row adds a new
    direction and barely moves when a row duplicates one already present.
    """
    n = X.shape[0]
    if n == 0:
        return 0.0
    # Computed from the singular values rather than an explicit determinant:
    # equivalent, and cheap enough to call once per candidate per pick.
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # Energy-based truncation: keep components covering >= 95% of variance.
    if len(s) > 0:
        energy = np.cumsum(s**2)
        total = energy[-1] if energy[-1] > 0 else 1.0
        cutoff = np.searchsorted(energy, 0.95 * total) + 1
        s = s[:cutoff]
    coeff = 1.0 / (eps**2 * n)
    return 0.5 * float(np.sum(np.log1p(coeff * (s**2))))


def _embed_traits(
    traits: list[Trait],
    region: str,
    embedding_model: str,
    max_workers: int = 10,
) -> np.ndarray:
    """Embed trait descriptions via Titan (matches the consolidation cache shape).

    Re-embeds rather than reusing the consolidation cache, keeping selection
    decoupled from consolidation's embedding space.
    """
    # Lazy import to avoid a circular dependency at module load time.
    from premise.rubric_discovery.run import generate_embeddings

    client = get_bedrock_client(region)
    texts = [t.description for t in traits]
    embeddings = generate_embeddings(
        client,
        texts,
        embedding_model,
        cache_path=None,
        max_workers=max_workers,
    )
    return embeddings


def select_by_logdet(
    traits: list[Trait],
    target_traits: int,
    *,
    eps: float = 0.5,
    min_increment: float = 0.001,
    patience: int = 3,
    region: str = DEFAULT_REGION,
    embedding_model: str | None = None,
) -> tuple[list[str], dict[str, dict], dict[str, Any]]:
    """Greedy log-volume selection. Stop on patience sub-threshold picks.

    Annotates every input trait with:
      selected_by_strategy: bool
      logdet_rank: int | None       (1-based pick order; None if not selected)
      logdet_volume_increment: float | None
    """
    if not traits:
        return [], {}, {"strategy": "logdet", "curve": []}

    embeddings = _embed_traits(
        traits, region, embedding_model or get_config().model("embedding")
    )
    # Normalize for cosine geometry.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = embeddings / norms

    n = len(traits)
    remaining = set(range(n))
    selected: list[int] = []
    increments: list[float] = []
    curve: list[dict[str, Any]] = []

    prev_rate = 0.0
    sub_threshold_streak = 0

    while remaining and len(selected) < target_traits:
        # Score each candidate by the log-volume increment it provides.
        best_idx = None
        best_rate = -math.inf
        for idx in remaining:
            trial = X[selected + [idx]]
            r = _log_volume(trial, eps=eps)
            if r > best_rate:
                best_rate = r
                best_idx = idx
        if best_idx is None:
            break

        increment = best_rate - prev_rate
        # Apply the patience rule on increments.
        if increment < min_increment:
            sub_threshold_streak += 1
            if sub_threshold_streak >= patience:
                # Stop *before* committing this pick — its increment is
                # below the floor and we've exhausted patience.
                break
        else:
            sub_threshold_streak = 0

        selected.append(best_idx)
        remaining.remove(best_idx)
        increments.append(increment)
        prev_rate = best_rate
        curve.append(
            {
                "k": len(selected),
                "trait_id": traits[best_idx].trait_id,
                "log_volume": best_rate,
                "increment": increment,
            }
        )

    selected_ids = [traits[i].trait_id for i in selected]
    rank_by_id = {tid: r for r, tid in enumerate(selected_ids, 1)}
    inc_by_id = {traits[idx].trait_id: increments[r] for r, idx in enumerate(selected)}
    annotations: dict[str, dict] = {}
    for t in traits:
        annotations[t.trait_id] = {
            "selected_by_strategy": t.trait_id in rank_by_id,
            "logdet_rank": rank_by_id.get(t.trait_id),
            "logdet_volume_increment": inc_by_id.get(t.trait_id),
        }
    return selected_ids, annotations, {"strategy": "logdet", "curve": curve}


# ============================================================
# Strategy: llm (configured reasoning model)
# ============================================================

LLM_TOPK_PROMPT = """You are auditing a candidate rubric. You will be given N evaluation traits that were discovered from preference data. Each trait has an ID, a description (the rubric statement), and a source_count (how many preference comparisons originally contributed to it).

YOUR TASK:
Select exactly {k} traits that together form a strong, non-redundant rubric for evaluating AI responses. Specifically:

1. SELECTION CRITERIA. The selected {k} should be:
   - DISTINCT: each measures a substantively different evaluation axis. Two traits are NOT distinct if they describe the same underlying quality with different wording (e.g., "validates assumptions about the user's intent" and "asks for clarification on ambiguous requests"). Treat these as paraphrastic duplicates.
   - HIGH-EVIDENCE: when choosing among near-equivalent candidates, prefer the one with higher source_count.
   - WELL-FORMED: each is a concrete, measurable rubric statement (not a vague slogan, not trivial, not specific to a single battle).

2. ENTITY-FREE. REJECT traits whose description bakes in sample-level entities — specific named papers/authors (e.g. "Vaswani et al. 2017"), specific products/APIs/files, specific chemical species or compounds (e.g. "CO2", "hydrocarbon combustion"), specific algorithms or named techniques tied to a single prompt's subject (e.g. "demand paging", "transformer architecture"), or specific datasets/benchmarks. Subject-area scope is OK ("verifies mathematical derivations", "explains chemistry mechanisms clearly"); naming a particular compound, paper, or system is NOT.

3. PARAPHRASTIC MERGING. If two or more candidates measure the same underlying quality:
   - KEEP the one with the highest source_count.
   - If tied on source_count, keep the one with the more concrete description.
   - List the others as MERGED into the kept trait.
   - The merged trait gets the kept trait's description verbatim (no rewriting).

4. NEVER:
   - Invent a new trait.
   - Rewrite an existing description.
   - Select more than {k}.

5. UNDERFLOW. It is BETTER to return fewer than {k} than to include entity-baked or paraphrastic fillers. If the pool has < {k} entity-free, policy-level traits, return only those and explain in `notes`.

CANDIDATE TRAITS:
{traits_block}

OUTPUT FORMAT (strict JSON):
{{
  "selected": ["trait_id_1", "trait_id_2", ...],
  "merges": [
    {{
      "kept": "trait_id_X",
      "merged_in": ["trait_id_Y", "trait_id_Z"],
      "reason": "brief: cite the paraphrastic overlap"
    }}
  ],
  "notes": ""
}}

The `selected` list must contain exactly {k} unique trait_ids drawn from the candidate pool. The `merges` list may be empty.

Return ONLY the JSON object."""


LLM_TOPK_RETRY_PROMPT = """The previous response had errors:
{errors}

Re-do the selection following the original instructions. Make sure:
- `selected` has exactly {k} unique trait_ids drawn from the candidate pool.
- All trait_ids in `merges[*].kept` and `merges[*].merged_in` are from the candidate pool.

Return ONLY the corrected JSON object."""


def _format_traits_block(traits: list[Trait]) -> str:
    lines = []
    for t in traits:
        lines.append(
            f"- id: {t.trait_id} | source_count: {t.source_count} | "
            f"description: {t.description}"
        )
    return "\n".join(lines)


def _validate_llm_response(
    parsed: dict,
    traits: list[Trait],
    k: int,
) -> list[str]:
    """Return a list of error strings; empty list = OK."""
    errors = []
    pool_ids = {t.trait_id for t in traits}

    selected = parsed.get("selected")
    if not isinstance(selected, list):
        errors.append("'selected' is missing or not a list.")
    else:
        if len(selected) != k:
            errors.append(f"'selected' has {len(selected)} ids; expected {k}.")
        if len(set(selected)) != len(selected):
            errors.append("'selected' contains duplicate trait_ids.")
        bad = [s for s in selected if s not in pool_ids]
        if bad:
            errors.append(
                f"'selected' contains ids not in the candidate pool: {bad[:5]}"
            )

    merges = parsed.get("merges", [])
    if not isinstance(merges, list):
        errors.append("'merges' is not a list.")
    else:
        for i, m in enumerate(merges):
            if not isinstance(m, dict):
                errors.append(f"merges[{i}] is not an object.")
                continue
            kept = m.get("kept")
            mlist = m.get("merged_in", [])
            if kept not in pool_ids:
                errors.append(f"merges[{i}].kept '{kept}' not in pool.")
            if not isinstance(mlist, list) or any(x not in pool_ids for x in mlist):
                errors.append(f"merges[{i}].merged_in contains ids not in pool.")
    return errors


def _parse_json_block(text: str) -> dict | None:
    if not text:
        return None
    # Models sometimes wrap in ```json fences.
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def select_by_llm(
    traits: list[Trait],
    target_traits: int,
    *,
    region: str = DEFAULT_REGION,
) -> tuple[list[str], dict[str, dict], dict[str, Any]]:
    """LLM-driven top-K selection with paraphrastic merging.

    Uses the configured reasoning model for the diversity and merge judgment.
    """
    model_id = get_config().model("reasoning")
    client = get_bedrock_client(region)
    k = min(target_traits, len(traits))

    if not traits:
        return [], {}, {"strategy": "llm", "merges": []}

    traits_block = _format_traits_block(traits)
    prompt = LLM_TOPK_PROMPT.format(k=k, traits_block=traits_block)

    response = invoke_model(
        client,
        model_id,
        prompt,
        temperature=0.0,
        max_tokens=8000,
    )
    parsed = _parse_json_block(response or "")
    errors = []
    if parsed is None:
        errors = ["Response was not valid JSON."]
    else:
        errors = _validate_llm_response(parsed, traits, k)

    if errors:
        # One retry with feedback.
        retry_prompt = (
            prompt
            + "\n\n--- PREVIOUS RESPONSE WAS REJECTED ---\n\n"
            + LLM_TOPK_RETRY_PROMPT.format(
                errors="\n".join(f"- {e}" for e in errors),
                k=k,
            )
        )
        response = invoke_model(
            client,
            model_id,
            retry_prompt,
            temperature=0.0,
            max_tokens=8000,
        )
        parsed = _parse_json_block(response or "")
        if parsed is None:
            print(
                "  WARN: LLM selection failed twice (no valid JSON); "
                "falling back to source_count strategy."
            )
            return select_by_source_count(traits, target_traits)
        errors = _validate_llm_response(parsed, traits, k)
        if errors:
            print(
                "  WARN: LLM selection malformed after retry; "
                f"errors: {errors[:3]}; trusting raw response anyway."
            )

    selected = list(dict.fromkeys(parsed.get("selected", [])))
    pool_ids = {t.trait_id for t in traits}
    selected = [s for s in selected if s in pool_ids]
    if len(selected) < k:
        print(
            f"  WARN: LLM returned {len(selected)} selections (expected {k}); "
            "logging underflow and proceeding."
        )

    merges_raw = parsed.get("merges", []) or []
    merges_clean = []
    merged_into_map: dict[str, str] = {}  # dropped_id -> kept_id
    merge_reason_map: dict[str, str] = {}
    summed_source: dict[str, int] = {}
    for m in merges_raw:
        if not isinstance(m, dict):
            continue
        kept = m.get("kept")
        mlist = [x for x in m.get("merged_in", []) if x in pool_ids and x != kept]
        if kept not in pool_ids or not mlist:
            continue
        reason = str(m.get("reason", ""))[:300]
        merges_clean.append({"kept": kept, "merged_in": mlist, "reason": reason})
        for d in mlist:
            merged_into_map[d] = kept
            merge_reason_map[d] = reason
        # Sum source_counts of merged-in into the kept trait.
        kept_trait = next(t for t in traits if t.trait_id == kept)
        extras = sum(t.source_count for t in traits if t.trait_id in mlist)
        summed_source[kept] = summed_source.get(kept, kept_trait.source_count) + extras

    selected_set = set(selected)
    annotations: dict[str, dict] = {}
    for t in traits:
        ann: dict[str, Any] = {
            "selected_by_strategy": t.trait_id in selected_set,
        }
        if t.trait_id in merged_into_map:
            ann["merged_into"] = merged_into_map[t.trait_id]
            ann["merge_reason"] = merge_reason_map.get(t.trait_id, "")
        if t.trait_id in summed_source:
            ann["source_count_after_merge"] = summed_source[t.trait_id]
        annotations[t.trait_id] = ann

    extras = {
        "strategy": "llm",
        "merges": merges_clean,
        "notes": str(parsed.get("notes", ""))[:500],
        "underflow": len(selected) < k,
    }
    return selected, annotations, extras


# ============================================================
# Public dispatcher
# ============================================================


# ============================================================
# Strategy: mmr (genericity filter + greedy max-marginal-relevance)
# ============================================================


GENERICITY_PROMPT = """You are auditing a discovered rubric pool. For each rubric description, label whether it is GENERIC (a general evaluation axis applicable across many prompts and domains) or PROMPT_SPECIFIC (clearly tied to a single prompt's answer key — references specific named entities, named tools/papers/products, very narrow task structures like "configure transfer learning model with X dropout", or domain-specific facts that only matter for one question).

Examples:
  GENERIC: "The response should reproduce all specified content, structure, and formatting exactly as provided"
  GENERIC: "The response should use specific, measurable, and observable details such as named examples, quantified metrics"
  GENERIC: "The response should adhere to the user's explicit and implicit requirements"
  PROMPT_SPECIFIC: "The response should correctly identify a gas that is a direct chemical byproduct of hydrocarbon combustion in household appliances"
  PROMPT_SPECIFIC: "The response should configure a transfer learning model with a pre-trained base, global average pooling, dropout, sigmoid output"
  PROMPT_SPECIFIC: "The response should explicitly reference the Vaswani et al. 2017 paper"
  PROMPT_SPECIFIC: "The response should list at least four distinct non-traditional career paths for law degree holders"

RUBRICS TO LABEL:
{block}

OUTPUT FORMAT (strict JSON):
{{"labels": [{{"id": "<id>", "label": "generic" | "prompt_specific"}}, ...]}}

Rules:
- Return EXACTLY {n} labels in the same order as input.
- Use lowercase "generic" or "prompt_specific" only.
- Return ONLY the JSON object."""


def _label_genericity(
    traits: list[Trait],
    region: str,
) -> dict[str, str]:
    """LLM-label each trait as 'generic' or 'prompt_specific'.

    Returns {trait_id: label}. Falls back to all-generic on parse failure
    so we never silently drop candidates due to an LLM error.
    """
    if not traits:
        return {}
    block = "\n".join(f"- {t.trait_id}: {t.description}" for t in traits)
    prompt = GENERICITY_PROMPT.format(block=block, n=len(traits))
    client = get_bedrock_client(region)
    text = invoke_model(
        client,
        get_config().model("genericity"),
        prompt,
        temperature=0.0,
        max_tokens=4000 + 30 * len(traits),
    )
    if not text:
        print("  WARN: genericity-label call returned empty; treating all as generic")
        return {t.trait_id: "generic" for t in traits}
    parsed = _parse_json_block(text)
    if not parsed:
        print("  WARN: genericity-label parse failed; treating all as generic")
        return {t.trait_id: "generic" for t in traits}
    out: dict[str, str] = {}
    for entry in parsed.get("labels", []):
        if isinstance(entry, dict) and entry.get("id") and entry.get("label"):
            lab = str(entry["label"]).strip().lower()
            if lab in ("generic", "prompt_specific"):
                out[entry["id"]] = lab
    # Fill any missing as generic.
    for t in traits:
        out.setdefault(t.trait_id, "generic")
    return out


ABSTRACTION_PROMPT = """You will receive a list of rubrics that were flagged as PROMPT_SPECIFIC — each is tied to a particular prompt's answer key (e.g., specific named entities, narrow domain facts, single-task structures).

Your job: identify whether some of these prompt-specific rubrics share a *latent generalizable evaluation axis*. If two or more prompt-specific rubrics, despite their narrow surface form, are really expressing the same abstract evaluative principle, propose ONE generalized rubric that subsumes them. The abstracted rubric must:
  - Be specific enough to be objectively scorable by an evaluator on any response.
  - Be general enough to apply across many prompts in many domains.
  - Use a should-statement description (e.g., "The response should ...").
  - NOT reference any prompt-specific named entity, tool, paper, product, formula, or fact.

Examples of good abstractions:
  - From "should reference the Vaswani et al. 2017 paper" + "should cite legal provisions valid as of 2024" → "The response should ground all factual claims in named, currently-authoritative sources for the prompt's domain."
  - From "should configure a transfer-learning model with global average pooling and dropout" + "should use idiomatic Racket place/thread patterns" → "The response should follow domain-idiomatic implementation conventions for the specific framework or language requested."
  - From "should compute probability ≈ 1/(n ln 10) for primes of magnitude 10^n" + "should generate lesion masks accounting for low-contrast medical imaging" → "The response should apply the correct quantitative method or formula appropriate to the problem domain."

Do NOT force abstractions. If a group of rubrics doesn't share a clean latent axis, leave them out. Conservative is better than wrong.
Do NOT propose abstractions that duplicate generic axes already covered (e.g., 'be factually accurate', 'be well-organized') — those are already in the kept pool.

PROMPT-SPECIFIC RUBRICS (id : description):
{block}

OUTPUT FORMAT (strict JSON):
{{
  "abstractions": [
    {{
      "abstract_id": "snake_case_id",
      "abstract_description": "should-statement, general but evaluable",
      "subsumes": ["original_id_1", "original_id_2", ...]
    }}
  ]
}}

Rules:
- Each abstraction MUST subsume at least 2 of the input rubrics.
- abstract_id is unique snake_case (2-5 words).
- It is acceptable to return zero abstractions if none apply.
- Return ONLY the JSON object."""


def _rescue_abstractions(
    prompt_specific_traits: list[Trait],
    region: str,
    min_subsumes: int = 2,
) -> list[Trait]:
    """Ask an LLM to propose generalizable abstractions over prompt-specific traits.

    Each returned Trait carries a synthetic trait_id, a should-statement description,
    and source_count = sum of subsumed traits' source_counts (so the rescue inherits
    the empirical mass of the originals).

    Returns [] on parse failure or no proposals.
    """
    if len(prompt_specific_traits) < min_subsumes:
        return []
    # Lazy import to avoid circular dep when run.py is the caller.
    from premise.rubric_discovery.run import Trait

    block = "\n".join(
        f"- {t.trait_id}: {t.description}" for t in prompt_specific_traits
    )
    prompt = ABSTRACTION_PROMPT.format(block=block)
    client = get_bedrock_client(region)
    text = invoke_model(
        client,
        get_config().model("genericity"),
        prompt,
        temperature=0.0,
        max_tokens=4000 + 50 * len(prompt_specific_traits),
    )
    if not text:
        print("  WARN: abstraction-rescue call returned empty")
        return []
    parsed = _parse_json_block(text)
    if not parsed:
        print("  WARN: abstraction-rescue parse failed")
        return []

    by_id = {t.trait_id: t for t in prompt_specific_traits}
    rescued: list[Trait] = []
    for entry in parsed.get("abstractions", []):
        if not isinstance(entry, dict):
            continue
        a_id = entry.get("abstract_id")
        a_desc = entry.get("abstract_description")
        subsumes = entry.get("subsumes") or []
        if not a_id or not a_desc:
            continue
        if not isinstance(subsumes, list) or len(subsumes) < min_subsumes:
            continue
        valid_subsumes = [s for s in subsumes if s in by_id]
        if len(valid_subsumes) < min_subsumes:
            continue
        # Inherit summed source_count and concatenated source_ids.
        total_sc = sum(by_id[s].source_count for s in valid_subsumes)
        all_src_ids: list[str] = []
        for s in valid_subsumes:
            all_src_ids.extend(by_id[s].source_ids or [s])
        rescued.append(
            Trait(
                trait_id=f"abstracted__{a_id}",
                description=a_desc,
                source_count=total_sc,
                source_ids=all_src_ids,
            )
        )
    if rescued:
        print(
            f"  abstraction-rescue: salvaged {len(rescued)} generalized axes "
            f"from {len(prompt_specific_traits)} prompt-specific candidates"
        )
        for r in rescued:
            print(f"    + [src={r.source_count:>4}] {r.trait_id}")
    else:
        print("  abstraction-rescue: no generalizable axes proposed")
    return rescued


def select_by_mmr(
    traits: list[Trait],
    target_traits: int,
    *,
    lambda_: float = 0.75,
    skip_genericity_filter: bool = False,
    abstraction_rescue: bool = True,
    region: str = DEFAULT_REGION,
    embedding_model: str | None = None,
) -> tuple[list[str], dict[str, dict], dict[str, Any]]:
    """Genericity filter + abstraction rescue + greedy MMR selection.

    Steps:
      1. LLM genericity filter drops PROMPT_SPECIFIC traits.
      1b. (optional, default on) Abstraction rescue: ask the LLM whether any
          generalizable evaluation axes can be extracted from the dropped
          prompt-specific bucket. Rescued traits get a synthetic
          'abstracted__<id>' trait_id and inherit the summed source_count
          of the originals they subsume.
      2. Embed kept (generic + rescued) candidates via Titan.
      3. Greedy MMR: pick the highest log-source trait first, then iteratively
         pick the trait that maximizes
            lambda * normalized_log_source - (1-lambda) * max_cosine_to_selected
         until K traits are selected (or pool exhausted).

    Annotates each trait with:
      selected_by_strategy: bool
      mmr_rank: int | None         (1-based selection order)
      mmr_genericity: 'generic' | 'prompt_specific' | 'abstracted' | None
    """
    from sklearn.preprocessing import normalize

    extras: dict[str, Any] = {"strategy": "mmr", "lambda": lambda_}

    if not traits:
        return (
            [],
            {},
            {
                **extras,
                "n_pool_input": 0,
                "n_after_genericity": 0,
                "n_dropped_prompt_specific": 0,
            },
        )

    # 1) Genericity filter.
    labels: dict[str, str] = {}
    rescued_traits: list = []  # type: ignore[var-annotated]
    if not skip_genericity_filter:
        print(
            f"  MMR: genericity-labeling {len(traits)} candidates "
            f"via {get_config().model('genericity')}..."
        )
        labels = _label_genericity(traits, region)
        n_specific = sum(1 for v in labels.values() if v == "prompt_specific")
        print(f"    prompt-specific (dropped): {n_specific}")
        print(f"    generic (kept):            {len(traits) - n_specific}")
        kept = [t for t in traits if labels.get(t.trait_id, "generic") == "generic"]

        # 1b) Abstraction rescue: try to salvage generalizable axes from the
        #     prompt-specific bucket. Each rescued trait inherits the summed
        #     source_count of the originals it subsumes.
        if abstraction_rescue and n_specific >= 2:
            specific_traits = [
                t for t in traits if labels.get(t.trait_id) == "prompt_specific"
            ]
            rescued_traits = _rescue_abstractions(specific_traits, region)
            kept = kept + rescued_traits

        # Safety: if filter (even after rescue) would drop us below K,
        # fall back to keeping all originals.
        if len(kept) < target_traits:
            print(
                f"    WARN: only {len(kept)} kept candidates < target K={target_traits}; "
                f"falling back to full pool"
            )
            kept = list(traits) + rescued_traits
            for t in traits:
                labels[t.trait_id] = "generic"
    else:
        kept = list(traits)
        labels = {t.trait_id: "generic" for t in traits}

    extras["n_pool_input"] = len(traits)
    extras["n_after_genericity"] = len(kept) - len(rescued_traits)
    extras["n_dropped_prompt_specific"] = len(traits) - extras["n_after_genericity"]
    extras["n_abstraction_rescued"] = len(rescued_traits)
    extras["abstraction_rescue_enabled"] = (
        abstraction_rescue and not skip_genericity_filter
    )

    # 2) Embed kept candidates.
    embeddings = _embed_traits(
        kept, region, embedding_model or get_config().model("embedding")
    )
    norms = np.linalg.norm(embeddings, axis=1)
    n_zero = int((norms == 0).sum())
    if n_zero:
        print(f"  MMR: WARN: {n_zero}/{len(kept)} embeddings are zero")
    valid = norms > 0

    # 3) Greedy MMR over log-scaled source_count.
    sources = np.array([t.source_count for t in kept], dtype=float)
    s = np.log1p(sources)
    if s.max() == s.min():
        s_norm = np.ones_like(s)
    else:
        s_norm = (s - s.min()) / (s.max() - s.min())
    s_norm = np.where(valid, s_norm, -1.0)

    if not valid.any():
        return (
            [],
            {
                t.trait_id: {
                    "selected_by_strategy": False,
                    "mmr_genericity": labels.get(t.trait_id),
                    "mmr_rank": None,
                }
                for t in traits
            },
            extras,
        )

    safe_emb = embeddings.copy()
    safe_emb[~valid] = 0.0
    normed = normalize(safe_emb)

    selected: list[int] = []
    first = int(np.argmax(s_norm))
    selected.append(first)

    for _ in range(min(target_traits, len(kept)) - 1):
        sim_to_sel = normed @ normed[selected].T
        max_sim = sim_to_sel.max(axis=1)
        mmr_score = lambda_ * s_norm - (1.0 - lambda_) * max_sim
        mmr_score[selected] = -np.inf
        mmr_score[~valid] = -np.inf
        nxt = int(np.argmax(mmr_score))
        if mmr_score[nxt] == -np.inf:
            break
        selected.append(nxt)

    # Diversity diagnostics: mean / max pairwise cosine sim of picks.
    if len(selected) >= 2:
        sub = normed[selected]
        sim = sub @ sub.T
        np.fill_diagonal(sim, np.nan)
        extras["mean_pairwise_cosine_sim"] = float(np.nanmean(sim))
        extras["max_pairwise_cosine_sim"] = float(np.nanmax(sim))

    # Surface rescued (synthetic) traits so callers can merge them back into
    # their pool when assembling the final saved JSON.
    if rescued_traits:
        extras["rescued_traits"] = [
            {
                "trait_id": t.trait_id,
                "description": t.description,
                "source_count": t.source_count,
                "source_ids": list(t.source_ids),
            }
            for t in rescued_traits
        ]

    # Build per-trait annotations.
    selected_ids = [kept[i].trait_id for i in selected]
    selected_set = set(selected_ids)
    rank_by_id = {tid: rank for rank, tid in enumerate(selected_ids, 1)}
    rescued_ids = {t.trait_id for t in rescued_traits}
    annotations: dict[str, dict] = {}

    def _label_for(tid: str) -> str:
        if tid in rescued_ids:
            return "abstracted"
        return labels.get(tid, "generic")

    for t in kept:
        annotations[t.trait_id] = {
            "selected_by_strategy": t.trait_id in selected_set,
            "mmr_rank": rank_by_id.get(t.trait_id),
            "mmr_genericity": _label_for(t.trait_id),
        }
    # Traits dropped by the genericity filter still get annotations.
    for t in traits:
        if t.trait_id not in annotations:
            annotations[t.trait_id] = {
                "selected_by_strategy": False,
                "mmr_rank": None,
                "mmr_genericity": labels.get(t.trait_id),
            }
    return selected_ids, annotations, extras


SUPPORTED_STRATEGIES = ("source_count", "logdet", "llm", "mmr")


def run_selection(
    strategy: str,
    traits: list[Trait],
    target_traits: int,
    *,
    selection_pool_size: int,
    region: str = DEFAULT_REGION,
    embedding_model: str | None = None,
    logdet_eps: float = 0.5,
    logdet_min_increment: float = 0.001,
    logdet_patience: int = 3,
    mmr_lambda: float = 0.75,
    mmr_skip_genericity_filter: bool = False,
    mmr_disable_abstraction_rescue: bool = False,
) -> tuple[list[str], dict[str, dict], dict[str, Any]]:
    """Cap input pool to top selection_pool_size by source_count, then dispatch.

    Returns (selected_ids, per_trait_annotations, extras_dict).
    Trait_ids not in the capped pool get annotation
    {"selected_by_strategy": False} so callers can uniformly merge into
    the saved JSON.
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown selection strategy: {strategy!r}. "
            f"Supported: {SUPPORTED_STRATEGIES}"
        )

    sorted_pool = sorted(traits, key=lambda t: -t.source_count)
    pool = sorted_pool[:selection_pool_size]
    pool_ids = {t.trait_id for t in pool}

    if strategy == "source_count":
        selected_ids, annotations, extras = select_by_source_count(
            pool,
            target_traits,
        )
    elif strategy == "logdet":
        selected_ids, annotations, extras = select_by_logdet(
            pool,
            target_traits,
            eps=logdet_eps,
            min_increment=logdet_min_increment,
            patience=logdet_patience,
            region=region,
            embedding_model=embedding_model,
        )
    elif strategy == "mmr":
        selected_ids, annotations, extras = select_by_mmr(
            pool,
            target_traits,
            lambda_=mmr_lambda,
            skip_genericity_filter=mmr_skip_genericity_filter,
            abstraction_rescue=not mmr_disable_abstraction_rescue,
            region=region,
            embedding_model=embedding_model,
        )
    else:  # strategy == "llm"
        selected_ids, annotations, extras = select_by_llm(
            pool,
            target_traits,
            region=region,
        )

    # Pad annotations with `selected_by_strategy: False` for traits that
    # didn't make the input pool — keeps the saved JSON shape uniform.
    for t in traits:
        if t.trait_id not in pool_ids and t.trait_id not in annotations:
            annotations[t.trait_id] = {"selected_by_strategy": False}

    extras["pool_size"] = len(pool)
    extras["target_k"] = target_traits
    extras["actual_k"] = len(selected_ids)
    return selected_ids, annotations, extras


# ============================================================
# Atomicity rewrite pass (Stage 2.6)
# ============================================================

ATOMIC_REWRITE_PROMPT = """You are tightening a discovered evaluation rubric. The trait below may bundle multiple distinct evaluation qualities into one description; your job is to rewrite it as a single atomic should-statement.

CURRENT TRAIT
- name: {name}
- description: {description}

REWRITE RULES
1. The rewritten description must measure EXACTLY ONE quality.
2. Do NOT use "and", "with", "while", "including", em-dashes, or commas to join two distinct evaluation qualities. (Commas inside a list of examples are fine.)
3. If the current description bundles multiple qualities, pick the SINGLE most central and most measurable quality, and drop the others.
4. Keep it 15-30 words. The rewrite should be a should-statement that another evaluator could score the same way.
5. Update the trait_id (snake_case) to match the rewritten quality. The new id must NOT contain "_and_", "_with_", or "_including_".

EXAMPLES OF GOOD ATOMIC SHOULD-STATEMENTS
- "The response should not invent citations that do not exist."
- "The response should use named, real-world examples to ground its claims."
- "The response should preserve the original numerical units when restating quantities."

EXAMPLES OF BAD (BUNDLED) STATEMENTS — DO NOT PRODUCE THESE
- "The response should be accurate and use clear examples." (joins two qualities)
- "The response should be concise while preserving completeness." (joins two qualities)

OUTPUT FORMAT (strict JSON):
{{"trait_id": "snake_case_name", "trait_description": "atomic should-statement"}}

If the current description is already atomic and well-formed, return it unchanged (same id, same description).

Return ONLY the JSON object."""


def _looks_bundled(description: str, name: str) -> bool:
    """Heuristic: trigger rewrite only on traits whose form suggests bundling."""
    if not description:
        return False
    lower = description.lower()
    if (
        " and " in lower
        or " while " in lower
        or " with " in lower
        or " including " in lower
    ):
        return True
    # em-dash bundling
    if "—" in description:
        return True
    # name itself bundles
    n = (name or "").lower()
    if "_and_" in n or "_with_" in n or "_including_" in n:
        return True
    # commas joining qualities (>= 3 commas often indicates multiple clauses)
    return description.count(",") >= 3


def atomic_rewrite_traits(
    traits: list[Trait],
    *,
    region: str = DEFAULT_REGION,
) -> tuple[list[Trait], dict[str, dict]]:
    """Rewrite each trait as an atomic should-statement.

    Only fires on traits that look bundled (heuristic: contains 'and', 'with',
    'while', 'including', em-dash, or '_and_' in name). Atomic-looking traits
    are left untouched.

    Returns (new_traits, annotations) where annotations[trait_id] = {
        "atomic_rewrite_applied": bool,
        "atomic_rewrite_old_id": str | None,
        "atomic_rewrite_old_description": str | None,
    }.
    """
    from premise.rubric_discovery.run import Trait  # avoid circular import at load

    if not traits:
        return [], {}

    client = get_bedrock_client(region)
    out_traits = []
    annotations: dict[str, dict] = {}

    for t in traits:
        if not _looks_bundled(t.description, t.trait_id):
            annotations[t.trait_id] = {"atomic_rewrite_applied": False}
            out_traits.append(t)
            continue

        prompt = ATOMIC_REWRITE_PROMPT.format(
            name=t.trait_id,
            description=t.description,
        )
        response = invoke_model(
            client,
            get_config().model("atomic_rewrite"),
            prompt,
            temperature=0.0,
            max_tokens=600,
        )
        parsed = _parse_json_block(response or "")
        if not parsed:
            print(
                f"  WARN: atomic-rewrite parse failed for {t.trait_id}; keeping original"
            )
            annotations[t.trait_id] = {"atomic_rewrite_applied": False}
            out_traits.append(t)
            continue

        new_id = (parsed.get("trait_id") or "").strip() or t.trait_id
        new_desc = (parsed.get("trait_description") or "").strip()
        if not new_desc:
            annotations[t.trait_id] = {"atomic_rewrite_applied": False}
            out_traits.append(t)
            continue

        # Track the rewrite under the *new* trait_id so downstream metadata
        # reflects what got published.
        applied = (new_id != t.trait_id) or (new_desc != t.description)
        annotations[new_id] = {
            "atomic_rewrite_applied": applied,
            "atomic_rewrite_old_id": t.trait_id if applied else None,
            "atomic_rewrite_old_description": t.description if applied else None,
        }
        out_traits.append(
            Trait(
                trait_id=new_id,
                description=new_desc,
                source_count=t.source_count,
                source_ids=list(t.source_ids),
            )
        )

    n_changed = sum(1 for v in annotations.values() if v.get("atomic_rewrite_applied"))
    print(f"  Atomicity rewrite: changed {n_changed}/{len(traits)} traits")
    return out_traits, annotations
