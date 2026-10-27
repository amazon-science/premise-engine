# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
PReMISE Trait Discovery Pipeline (self-contained).

Discovers policy-level rubrics from pairwise arena battles through:
1. Criteria Extraction — LLM identifies differentiating criteria from each battle pair
2. Iterative Consolidation — embed → cluster → sub-group → describe → repeat

No external dependencies beyond boto3, numpy, sklearn.

Usage:
    # Quick run on a slice of your battles
    premise discover run \
        --input-preference-data <preferences.jsonl> \
        --output results/discovery/

    # Full settings: consolidate until 20 traits remain
    premise discover run \
        --input-preference-data <preferences.jsonl> \
        --output results/discovery/ \
        --judge-model <model-id> \
        --max-levels 10 --target-traits 20

`--input-preference-data` is any JSONL of pairwise preferences; see README for
the schema. The prior `--arena-data` name remains as a hidden compatibility
alias.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from premise.config import get_config
from premise.provenance import (
    active_config_identity,
    canonical_sha256,
    changed_identity_fields,
    file_sha256,
)
from premise.utils import (
    DEFAULT_BATCH_ROLE_ARN,
    DEFAULT_REGION,
    DEFAULT_S3_BUCKET,
    ProviderRequestError,
    compute_file_hash,
    download_batch_results,
    extract_batch_model_response,
    get_batch_model_input,
    get_bedrock_client,
    get_bedrock_management_client,
    invoke_model,
    submit_batch_job,
    upload_to_s3,
    wait_for_batch_job,
)

DISCOVERY_STATE_FILE = "discovery-workflow.json"
DISCOVERY_WORKFLOW_REVISION = 3


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_discovery_state(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _prepare_discovery_state(
    output_dir: str,
    identity: dict,
    *,
    skip_extraction: bool,
) -> tuple[Path, dict]:
    """Create or validate provenance before any discovery cache is reused."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    state_path = output_path / DISCOVERY_STATE_FILE

    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            changed = ", ".join(
                changed_identity_fields(state.get("identity"), identity)
            )
            raise ValueError(
                f"{state_path} does not match the current discovery invocation "
                f"(changed: {changed}); choose a new --output directory"
            )
        return state_path, state

    existing = sorted(
        entry.name
        for entry in output_path.iterdir()
        if entry.name != DISCOVERY_STATE_FILE
    )
    allowed = {"extracted_criteria.jsonl"} if skip_extraction else set()
    unsafe = [name for name in existing if name not in allowed]
    if unsafe or (existing and not skip_extraction):
        names = ", ".join(unsafe or existing)
        raise ValueError(
            f"{output_path} contains discovery artifacts without validated "
            f"provenance ({names}); choose a new --output directory"
        )
    if skip_extraction and "extracted_criteria.jsonl" not in existing:
        raise ValueError(
            "--skip-extraction requires extracted_criteria.jsonl in the "
            f"output directory: {output_path}"
        )

    state = {
        "schema_version": "1.0",
        "kind": "discovery",
        "identity": identity,
        "status": "running",
        "created_at": _utc_now(),
    }
    if skip_extraction:
        state["adopted_extraction"] = True
    _write_discovery_state(state_path, state)
    return state_path, state


# ============================================================
# Data Structures
# ============================================================


@dataclass
class ExtractedCriterion:
    """A single criterion extracted from a battle."""

    battle_id: str
    name: str
    description: str
    model_a_score: int
    model_b_score: int


@dataclass
class Trait:
    """A trait at any consolidation level."""

    trait_id: str
    description: str
    source_count: int = 1
    source_ids: list[str] = field(default_factory=list)


class CriteriaExtractionError(RuntimeError):
    """Raised when Stage 1 cannot produce trustworthy parsed criteria."""


# ============================================================
# Stage 1: Criteria Extraction
# ============================================================


def _focus_cue(focus_theme: str) -> str:
    """Optional thematic steer for criteria extraction. Default blank = no steer
    (extract whatever differentiates). When set, PRIORITIZE — not mandate —
    criteria related to the given theme, preserving the anti-fabrication guard."""
    theme = (focus_theme or "").strip()
    if not theme:
        return ""
    return (
        f"\n\nFOCUS THEME: When eligible criteria related to **{theme}** are genuinely "
        f"present and discriminating in this pair, prioritize surfacing them. Do NOT "
        f"fabricate {theme} criteria on pairs where they do not genuinely differentiate "
        f"— the eligibility and anti-fabrication rules above still take precedence.\n"
    )


CRITERIA_EXTRACTION_PROMPT = """Analyze these two AI model responses and identify evaluation criteria where they meaningfully differ.

USER PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

WINNER: {winner}

YOUR TASK:
Identify up to {k} key evaluation criteria where these responses meaningfully differ. A criterion is eligible only if BOTH:
  (a) both responses are scoreable on it (the criterion is applicable to this prompt type and visible in both responses), AND
  (b) the responses behave differently on it (i.e., the criterion discriminates).

Do NOT extract criteria that are irrelevant to the prompt type or invisible in one or both responses (e.g., "code efficiency" when neither response contains code; "math accuracy" on a creative writing pair).

If fewer than {k} criteria are genuinely eligible for this pair, return fewer — fabricating criteria is worse than returning a smaller list.

For each criterion:
1. Name it concisely (2-5 words). The name MUST describe a single quality — names like "X_and_Y" indicate two traits and are forbidden.
2. Write a description as a should-statement that can be judged objectively and deterministically — i.e., another evaluator could read the description and reliably score the same way.
3. Score each response 1-10

ATOMICITY (critical):
- Each criterion describes EXACTLY ONE quality.
- A description must NOT use "and", "with", "while", "including", or em-dashes to join two distinct evaluation qualities. If you find yourself writing two qualities, split them into two separate criteria, or pick the more measurable one.
- A response should be scoreable on this criterion independently of any other criterion.

SPECIFICITY (critical):
- Each criterion should describe a concrete, measurable behavior pattern — specific enough to explain WHY the winner was preferred in this pair.
- DO name task patterns and behavioral expectations when they explain the preference (e.g., "provide runnable code before explaining it", "answer within the first sentence", "lead with the most important finding").
- Do NOT name specific entities, people, products, or datasets from this pair (e.g., "Joe Wegg", "the WorkFlowAI project", "the Vaswani 2017 paper"). Describe the behavioral pattern, not the content.
- Test: does this criterion capture a specific POLICY CHOICE (something a user or org might actually decide to value), or is it just a motherhood statement that any good response would satisfy?
- BAD (too generic): "The response should be accurate." — every response should be accurate; this doesn't explain a preference.
- GOOD (policy-level): "The response should provide a direct answer before elaborating." — this captures a real tradeoff (directness vs. thoroughness) that explains why some responses are preferred.

EXAMPLES OF WELL-FORMED (atomic, specific) DESCRIPTIONS:
- "The response should provide runnable code before any explanation."
- "The response should answer the question in the first sentence without preamble."
- "The response should acknowledge uncertainty rather than guessing when information is insufficient."
- "The response should use numbered steps rather than prose paragraphs for procedures."

EXAMPLES OF BAD DESCRIPTIONS — do NOT produce these:
- "The response should be accurate and use clear examples." (bundled — two qualities joined by "and")
- "The response should be concise while preserving completeness." (bundled — two qualities joined by "while")
- "The response should not include extraneous details unrelated to the cause of Joe Wegg's recovery." (entity-specific — names content from this pair)
- "The response should be helpful." (too generic — motherhood statement, not a policy choice)
- "The response should be well-organized." (too generic — doesn't capture HOW to organize or what pattern is preferred)

OUTPUT FORMAT (strict JSON):
{{
  "criteria": [
    {{
      "name": "criterion_name",
      "description": "should-statement describing ONE specific behavioral pattern that explains the preference",
      "model_a_score": <1-10>,
      "model_b_score": <1-10>
    }}
  ]
}}

IMPORTANT:
- Return ONLY the JSON object
- Return at most {k} criteria; return fewer if {k} eligible criteria do not exist for this pair
- Each description should be an atomic should-statement that any evaluator could score the same way
- Trait names must NOT contain "_and_", "_with_", "_including_", or describe more than one quality
- Each description must not reference specific entities, people, or products from the responses — but DO describe specific behavioral patterns
- Each criterion must be applicable to BOTH responses (skip criteria that don't apply to this prompt type)
- Include at least one criterion where the LOSING response scores HIGHER than the winner, if any such axis genuinely exists (e.g., the loser may be more concise, more creative, or more cautious). Not every pair has loser-advantages; if none exists, all criteria may favor the winner.
- Focus on meaningful differences, not minor stylistic variations"""


CRITERIA_EXTRACTION_WITH_TRACE_PROMPT = """Analyze these two AI model responses and identify evaluation criteria where they meaningfully differ.

USER PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

WINNER: {winner}

REASONING TRACE (why the winner won):
{reasoning_trace}

YOUR TASK:
Using the reasoning trace above as evidence, identify up to {k} key evaluation criteria where these responses meaningfully differ. A criterion is eligible only if BOTH:
  (a) both responses are scoreable on it (the criterion is applicable to this prompt type and visible in both responses), AND
  (b) the responses behave differently on it (i.e., the criterion discriminates).

Do NOT extract criteria that are irrelevant to the prompt type or invisible in one or both responses (e.g., "code efficiency" when neither response contains code; "math accuracy" on a creative writing pair).

If fewer than {k} criteria are genuinely eligible for this pair, return fewer — fabricating criteria is worse than returning a smaller list.

For each criterion:
1. Name it concisely (2-5 words). The name MUST describe a single quality — names like "X_and_Y" indicate two traits and are forbidden.
2. Write a description as a should-statement that can be judged objectively and deterministically — i.e., another evaluator could read the description and reliably score the same way.
3. Score each response 1-10

ATOMICITY (critical):
- Each criterion describes EXACTLY ONE quality.
- A description must NOT use "and", "with", "while", "including", or em-dashes to join two distinct evaluation qualities. If you find yourself writing two qualities, split them into two separate criteria, or pick the more measurable one.
- A response should be scoreable on this criterion independently of any other criterion.

SPECIFICITY (critical):
- Each criterion should describe a concrete, measurable behavior pattern — specific enough to explain WHY the winner was preferred in this pair.
- DO name task patterns and behavioral expectations when they explain the preference (e.g., "provide runnable code before explaining it", "answer within the first sentence", "lead with the most important finding").
- Do NOT name specific entities, people, products, or datasets from this pair (e.g., "Joe Wegg", "the WorkFlowAI project", "the Vaswani 2017 paper"). Describe the behavioral pattern, not the content.
- Test: does this criterion capture a specific POLICY CHOICE (something a user or org might actually decide to value), or is it just a motherhood statement that any good response would satisfy?
- BAD (too generic): "The response should be accurate." — every response should be accurate; this doesn't explain a preference.
- GOOD (policy-level): "The response should provide a direct answer before elaborating." — this captures a real tradeoff (directness vs. thoroughness) that explains why some responses are preferred.

EXAMPLES OF WELL-FORMED (atomic, specific) DESCRIPTIONS:
- "The response should provide runnable code before any explanation."
- "The response should answer the question in the first sentence without preamble."
- "The response should acknowledge uncertainty rather than guessing when information is insufficient."
- "The response should use numbered steps rather than prose paragraphs for procedures."

EXAMPLES OF BAD DESCRIPTIONS — do NOT produce these:
- "The response should be accurate and use clear examples." (bundled — two qualities joined by "and")
- "The response should be concise while preserving completeness." (bundled — two qualities joined by "while")
- "The response should not include extraneous details unrelated to the cause of Joe Wegg's recovery." (entity-specific — names content from this pair)
- "The response should be helpful." (too generic — motherhood statement, not a policy choice)
- "The response should be well-organized." (too generic — doesn't capture HOW to organize or what pattern is preferred)

OUTPUT FORMAT (strict JSON):
{{
  "criteria": [
    {{
      "name": "criterion_name",
      "description": "should-statement describing ONE specific behavioral pattern that explains the preference",
      "model_a_score": <1-10>,
      "model_b_score": <1-10>
    }}
  ]
}}

IMPORTANT:
- Return ONLY the JSON object
- Return at most {k} criteria; return fewer if {k} eligible criteria do not exist for this pair
- Each description should be an atomic should-statement that any evaluator could score the same way
- Trait names must NOT contain "_and_", "_with_", "_including_", or describe more than one quality
- Each description must not reference specific entities, people, or products from the responses — but DO describe specific behavioral patterns
- Ground each criterion in the reasoning trace
- Each criterion must be applicable to BOTH responses (skip criteria that don't apply to this prompt type)
- Include at least one criterion where the LOSING response scores HIGHER than the winner, if any such axis genuinely exists (e.g., the loser may be more concise, more creative, or more cautious). Not every pair has loser-advantages; if none exists, all criteria may favor the winner.
- Focus on meaningful differences, not minor stylistic variations"""


REASONING_TRACE_PROMPT = """You are tasked with analyzing two conversations between an AI assistant and a user. Based on the content, please provide a detailed explanation of why the user might have preferred the winning conversation.
Please consider aspects such as clarity, coherence, helpfulness, tone, and overall quality.

[Conversation with Assistant A]
{response_a}

[Conversation with Assistant B]
{response_b}

[User Prompt]
{prompt}

[Winning Conversation]
{winner}

[Your Explanation]
"""


def _reasoning_trace_model() -> str:
    """Return the configured reasoning model used to produce mining traces."""
    return get_config().model("reasoning")


def extract_conversation_text(conversation: list) -> tuple[str, str]:
    """Extract prompt and response from conversation structure."""
    prompt = ""
    response = ""
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if isinstance(content, list) and content:
            text = (
                content[0].get("text", "")
                if isinstance(content[0], dict)
                else str(content[0])
            )
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        if role == "user" and not prompt:
            prompt = text or ""
        elif role == "assistant" and not response:
            response = text or ""
    return prompt, response


def _normalize_battle(battle: dict) -> dict:
    """Normalize documented direct pairs and legacy conversation-shaped rows."""
    normalized = dict(battle)
    if "full_conversation_a" in normalized and "conversation_a" not in normalized:
        normalized["conversation_a"] = normalized["full_conversation_a"]
    if "full_conversation_b" in normalized and "conversation_b" not in normalized:
        normalized["conversation_b"] = normalized["full_conversation_b"]
    if "conversation_a" not in normalized and "prompt" in normalized:
        prompt = str(normalized.get("prompt", ""))
        normalized["conversation_a"] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(normalized.get("response_a", ""))},
        ]
        normalized["conversation_b"] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(normalized.get("response_b", ""))},
        ]
    return normalized


def _load_battles(arena_file: str, max_samples: int | None = None) -> list[dict]:
    """Load and normalize preference pairs for every discovery stage."""
    battles = []
    with open(arena_file) as f:
        for line in f:
            if line.strip():
                battles.append(_normalize_battle(json.loads(line)))
                if max_samples and len(battles) >= max_samples:
                    break
    return battles


def _first_json_object(text: str) -> dict:
    """Return the first parseable JSON object embedded in model output."""
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(value, dict):
            return value
    detail = f": {last_error.msg}" if last_error is not None else ""
    raise CriteriaExtractionError(
        f"model response did not contain a parseable JSON object{detail}"
    )


def extract_criteria_from_battle(
    client,
    model_id: str,
    battle: dict,
    k_per_pair: int = 15,
    reasoning_trace: str | None = None,
    focus_theme: str = "",
) -> list[ExtractedCriterion]:
    """Extract differentiating criteria from a single battle."""
    battle_id = battle.get("id", "unknown")

    # Extract conversations
    conv_a = battle.get("conversation_a", [])
    conv_b = battle.get("conversation_b", [])
    prompt_a, response_a = extract_conversation_text(conv_a)
    prompt_b, response_b = extract_conversation_text(conv_b)
    prompt = prompt_a or prompt_b

    if not prompt or not response_a or not response_b:
        raise CriteriaExtractionError(
            f"battle {battle_id!r} is missing a prompt or one of its responses"
        )

    template = (
        CRITERIA_EXTRACTION_WITH_TRACE_PROMPT
        if reasoning_trace
        else CRITERIA_EXTRACTION_PROMPT
    )
    fmt_kwargs = {
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "winner": battle.get("winner", "unknown"),
        "k": k_per_pair,
    }
    if reasoning_trace:
        fmt_kwargs["reasoning_trace"] = reasoning_trace
    eval_prompt = template.format(**fmt_kwargs) + _focus_cue(focus_theme)

    # Scale max_tokens with k (~150 tokens per criterion + overhead).
    max_tokens = max(2000, 200 * k_per_pair + 400)
    response = invoke_model(
        client, model_id, eval_prompt, temperature=0.1, max_tokens=max_tokens
    )
    if not response:
        raise ProviderRequestError(
            f"battle {battle_id!r}: provider returned an empty response"
        )

    # Parse the first valid object rather than a greedy ``{.*}`` span. This
    # accepts fenced JSON and models that add short prose before/after it.
    parsed = _first_json_object(response)
    raw_criteria = parsed.get("criteria")
    if not isinstance(raw_criteria, list):
        raise CriteriaExtractionError(
            f"battle {battle_id!r}: JSON response must contain a criteria array"
        )

    criteria = []
    for index, item in enumerate(raw_criteria):
        if not isinstance(item, dict):
            raise CriteriaExtractionError(
                f"battle {battle_id!r}: criterion {index + 1} must be an object"
            )
        name = item.get("name")
        description = item.get("description")
        if not isinstance(name, str) or not name.strip():
            raise CriteriaExtractionError(
                f"battle {battle_id!r}: criterion {index + 1} has no name"
            )
        if not isinstance(description, str) or not description.strip():
            raise CriteriaExtractionError(
                f"battle {battle_id!r}: criterion {index + 1} has no description"
            )
        try:
            model_a_score = int(item["model_a_score"])
            model_b_score = int(item["model_b_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CriteriaExtractionError(
                f"battle {battle_id!r}: criterion {index + 1} has invalid scores"
            ) from exc
        if not 1 <= model_a_score <= 10 or not 1 <= model_b_score <= 10:
            raise CriteriaExtractionError(
                f"battle {battle_id!r}: criterion {index + 1} scores must be 1-10"
            )
        criteria.append(
            ExtractedCriterion(
                battle_id=battle_id,
                name=name.strip(),
                description=description.strip(),
                model_a_score=model_a_score,
                model_b_score=model_b_score,
            )
        )

    return criteria


def run_criteria_extraction(
    client,
    model_id: str,
    arena_file: str,
    max_samples: int | None = None,
    k_per_pair: int = 15,
    use_reasoning_trace: bool = False,
    focus_theme: str = "",
) -> list[ExtractedCriterion]:
    """Extract criteria from all battles in the arena file."""
    print("\n=== Stage 1: Criteria Extraction ===")

    battles = _load_battles(arena_file, max_samples)
    if not battles:
        raise CriteriaExtractionError(
            f"no preference pairs were loaded from {arena_file}"
        )

    print(
        f"  Processing {len(battles)} battles (k_per_pair={k_per_pair}, "
        f"reasoning_trace={use_reasoning_trace})..."
    )
    all_criteria = []
    failures: list[str] = []
    parsed_empty = 0
    consecutive_parse_failures = 0

    for i, battle in enumerate(battles):
        trace = None
        if use_reasoning_trace:
            trace = _generate_reasoning_trace(client, model_id, battle)
        try:
            criteria = extract_criteria_from_battle(
                client,
                model_id,
                battle,
                k_per_pair=k_per_pair,
                reasoning_trace=trace,
                focus_theme=focus_theme,
            )
        except ProviderRequestError as exc:
            battle_id = battle.get("id", "unknown")
            raise CriteriaExtractionError(
                f"provider request failed on battle {battle_id}: {exc}"
            ) from exc
        except CriteriaExtractionError as exc:
            battle_id = battle.get("id", "unknown")
            message = f"{battle_id}: {exc}"
            failures.append(message)
            print(f"  WARN: extraction failed for {message}")
            consecutive_parse_failures += 1
            if consecutive_parse_failures >= 3:
                details = "; ".join(failures[-3:])
                raise CriteriaExtractionError(
                    "criteria extraction stopped after 3 consecutive malformed "
                    f"model responses. Failures: {details}"
                ) from exc
            continue
        consecutive_parse_failures = 0
        if not criteria:
            parsed_empty += 1
        all_criteria.extend(criteria)
        if (i + 1) % 50 == 0:
            print(
                f"  [{i + 1}/{len(battles)}] Extracted {len(all_criteria)} criteria so far"
            )

    print(
        f"  Total extracted: {len(all_criteria)} criteria from {len(battles)} battles"
    )
    if not all_criteria:
        details = "; ".join(failures[:3])
        if len(failures) > 3:
            details += f"; and {len(failures) - 3} more"
        summary = (
            f"{len(failures)} failed, {parsed_empty} returned an empty criteria array"
        )
        if details:
            summary += f". Failures: {details}"
        raise CriteriaExtractionError(
            f"criteria extraction produced no criteria from {len(battles)} "
            f"preference pairs ({summary})"
        )
    if failures:
        print(
            f"  WARN: {len(failures)}/{len(battles)} battles failed extraction; "
            "continuing with successfully parsed criteria"
        )
    return all_criteria


def _generate_reasoning_trace(client, model_id: str, battle: dict) -> str | None:
    """Generate the reasoning trace explaining why the winner won.

    `model_id` is intentionally ignored: this step is pinned to DeepSeek-R1
    so we can capture R1's `reasoningContent` block. Returns the
    reasoning_content (the model's <think> trace) when present, falling back
    to the visible explanation otherwise.
    """
    conv_a = battle.get("conversation_a", [])
    conv_b = battle.get("conversation_b", [])
    prompt_a, response_a = extract_conversation_text(conv_a)
    prompt_b, response_b = extract_conversation_text(conv_b)
    prompt = prompt_a or prompt_b
    if not prompt or not response_a or not response_b:
        return None

    eval_prompt = REASONING_TRACE_PROMPT.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        winner=battle.get("winner", "unknown"),
    )
    text, reasoning = invoke_model(
        client,
        _reasoning_trace_model(),
        eval_prompt,
        temperature=0.6,
        max_tokens=8000,
        return_reasoning=True,
    )
    return reasoning or text


# ============================================================
# Stage 1b: Batch Criteria Extraction (Bedrock Batch Inference)
# ============================================================


def prepare_batch_input(
    battles: list,
    model_id: str,
    output_dir: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    k_per_pair: int = 15,
    reasoning_traces: dict | None = None,
    filename: str = "batch_input.jsonl",
    focus_theme: str = "",
) -> str:
    """Build a JSONL batch input file for Bedrock batch inference. Returns file path.

    If reasoning_traces is provided (battle_id -> trace text), uses the
    with-trace prompt template.
    """
    batch_path = os.path.join(output_dir, filename)
    written = 0
    # Scale max_tokens with k (~200 tokens per criterion + 400 overhead),
    # honor an explicit caller value if it's larger.
    max_tokens = max(max_tokens, 200 * k_per_pair + 400)
    with open(batch_path, "w") as f:
        for battle in battles:
            battle_id = battle.get("id", "unknown")
            conv_a = battle.get("conversation_a", [])
            conv_b = battle.get("conversation_b", [])
            prompt_a, response_a = extract_conversation_text(conv_a)
            prompt_b, response_b = extract_conversation_text(conv_b)
            prompt = prompt_a or prompt_b
            if not prompt or not response_a or not response_b:
                continue

            trace = reasoning_traces.get(battle_id) if reasoning_traces else None
            template = (
                CRITERIA_EXTRACTION_WITH_TRACE_PROMPT
                if trace
                else CRITERIA_EXTRACTION_PROMPT
            )
            fmt_kwargs = {
                "prompt": prompt,
                "response_a": response_a,
                "response_b": response_b,
                "winner": battle.get("winner", "unknown"),
                "k": k_per_pair,
            }
            if trace:
                fmt_kwargs["reasoning_trace"] = trace
            eval_prompt = template.format(**fmt_kwargs) + _focus_cue(focus_theme)

            record = {
                "recordId": battle_id,
                "modelInput": get_batch_model_input(
                    model_id, eval_prompt, temperature, max_tokens
                ),
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"  Wrote {written} records to {batch_path}")
    return batch_path


def prepare_reasoning_trace_batch_input(
    battles: list,
    model_id: str,
    output_dir: str,
    temperature: float = 0.6,
    max_tokens: int = 8000,
    filename: str = "batch_trace_input.jsonl",
) -> str:
    """Build a JSONL batch input file for the reasoning-trace pre-pass.

    `model_id` is intentionally ignored: this step is pinned to DeepSeek-R1
    to capture R1's reasoning_content. Higher max_tokens because R1's
    thinking trace can be 2-4k tokens before the answer.
    """
    del model_id  # pinned to R1
    batch_path = os.path.join(output_dir, filename)
    written = 0
    with open(batch_path, "w") as f:
        for battle in battles:
            battle_id = battle.get("id", "unknown")
            conv_a = battle.get("conversation_a", [])
            conv_b = battle.get("conversation_b", [])
            prompt_a, response_a = extract_conversation_text(conv_a)
            prompt_b, response_b = extract_conversation_text(conv_b)
            prompt = prompt_a or prompt_b
            if not prompt or not response_a or not response_b:
                continue

            eval_prompt = REASONING_TRACE_PROMPT.format(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
                winner=battle.get("winner", "unknown"),
            )
            record = {
                "recordId": battle_id,
                "modelInput": get_batch_model_input(
                    _reasoning_trace_model(),
                    eval_prompt,
                    temperature,
                    max_tokens,
                ),
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(
        f"  Wrote {written} reasoning-trace records to {batch_path} "
        f"(model={_reasoning_trace_model()})"
    )
    return batch_path


def parse_batch_results(result_files: list, model_id: str) -> list[ExtractedCriterion]:
    """Parse downloaded .jsonl.out files into ExtractedCriterion objects."""
    all_criteria = []
    parsed_count = 0
    error_count = 0

    for rf in result_files:
        with open(rf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    error_count += 1
                    continue

                battle_id = rec.get("recordId", "unknown")
                model_output = rec.get("modelOutput", {})
                text = extract_batch_model_response(model_output, model_id)
                if not text:
                    error_count += 1
                    continue

                parsed_count += 1
                try:
                    json_match = re.search(r"\{.*\}", text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        raw_criteria = parsed.get("criteria", [])

                        # Normalize: model may return a dict-of-dicts instead of list-of-dicts
                        if isinstance(raw_criteria, dict):
                            raw_criteria = [
                                {
                                    "name": k,
                                    **(
                                        v
                                        if isinstance(v, dict)
                                        else {"description": str(v)}
                                    ),
                                }
                                for k, v in raw_criteria.items()
                            ]

                        for c in raw_criteria:
                            if not isinstance(c, dict):
                                continue
                            all_criteria.append(
                                ExtractedCriterion(
                                    battle_id=battle_id,
                                    name=c.get("name", "unknown"),
                                    description=c.get("description", ""),
                                    model_a_score=int(c.get("model_a_score", 5)),
                                    model_b_score=int(c.get("model_b_score", 5)),
                                )
                            )
                except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                    pass

    print(
        f"  Parsed {parsed_count} responses, {error_count} errors, {len(all_criteria)} criteria"
    )
    return all_criteria


def _run_batch_job(
    batch_path: str,
    model_id: str,
    output_dir: str,
    job_prefix: str,
    arena_name: str,
    output_subdir: str,
    s3_bucket: str,
    role_arn: str,
    region: str,
) -> list:
    """Upload a batch input JSONL, submit, wait, and download results.

    Returns the list of local .jsonl.out paths. Resumes from disk if
    `output_dir/<output_subdir>/*.jsonl.out` already exist.
    """
    local_results_dir = os.path.join(output_dir, output_subdir)
    existing_results = [
        os.path.join(local_results_dir, f)
        for f in (
            os.listdir(local_results_dir) if os.path.isdir(local_results_dir) else []
        )
        if f.endswith(".jsonl.out")
    ]
    if existing_results:
        print(
            f"  Found {len(existing_results)} existing {output_subdir} result files, skipping submit"
        )
        return existing_results

    file_hash = compute_file_hash(batch_path)[:12]
    job_name = f"{job_prefix}-{arena_name}-{file_hash}"
    job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)[:63]

    s3_key = f"premise/rubric_discovery/{job_name}/input.jsonl"
    s3_output_prefix = f"s3://{s3_bucket}/premise/rubric_discovery/{job_name}/output/"

    print(f"  Uploading batch input to s3://{s3_bucket}/{s3_key}")
    s3_input_uri = upload_to_s3(batch_path, s3_bucket, s3_key, region)

    bedrock = get_bedrock_management_client(region)
    print(f"  Submitting batch job: {job_name}")
    print(f"  Model: {model_id}")
    job_arn = submit_batch_job(
        bedrock, s3_input_uri, s3_output_prefix, job_name, model_id, role_arn
    )
    print(f"  Job ARN: {job_arn}")

    print("  Waiting for batch job to complete (polling every 60s)...")
    result = wait_for_batch_job(bedrock, job_arn, poll_interval=60, timeout=7200)

    output_s3_uri = (
        result.get("outputDataConfig", {})
        .get("s3OutputDataConfig", {})
        .get("s3Uri", s3_output_prefix)
    )
    output_prefix = output_s3_uri.replace(f"s3://{s3_bucket}/", "")
    print(f"  Downloading results from {output_s3_uri}")
    result_files = download_batch_results(
        s3_bucket, output_prefix, local_results_dir, region
    )
    print(f"  Downloaded {len(result_files)} result files")
    return result_files


def run_criteria_extraction_batch(
    model_id: str,
    arena_file: str,
    output_dir: str,
    max_samples: int | None = None,
    s3_bucket: str = DEFAULT_S3_BUCKET,
    role_arn: str = DEFAULT_BATCH_ROLE_ARN,
    region: str = DEFAULT_REGION,
    k_per_pair: int = 15,
    use_reasoning_trace: bool = False,
    focus_theme: str = "",
) -> list[ExtractedCriterion]:
    """Extract criteria using Bedrock batch inference (server-side parallel).

    When use_reasoning_trace=True, runs a trace pre-pass batch first, then
    feeds traces into the criteria batch via the with-trace template.
    """
    print("\n=== Stage 1: Criteria Extraction (Batch Inference) ===")
    print(f"  k_per_pair={k_per_pair}, use_reasoning_trace={use_reasoning_trace}")

    # Fast resume: if criteria batch_output exists, parse and return.
    local_results_dir = os.path.join(output_dir, "batch_output")
    existing_results = [
        os.path.join(local_results_dir, f)
        for f in (
            os.listdir(local_results_dir) if os.path.isdir(local_results_dir) else []
        )
        if f.endswith(".jsonl.out")
    ]
    if existing_results:
        print(
            f"  Found {len(existing_results)} existing batch result files, skipping upload/submit"
        )
        criteria = parse_batch_results(existing_results, model_id)
        print(f"  Total extracted: {len(criteria)} criteria")
        return criteria

    # Load battles
    battles = _load_battles(arena_file, max_samples)
    print(f"  Loaded {len(battles)} battles")

    arena_name = os.path.splitext(os.path.basename(arena_file))[0]

    # Optional Stage 1a: reasoning trace generation (DeepSeek-R1, sync only).
    # R1 doesn't support Bedrock batch inference, so we fan out via threads.
    reasoning_traces = None
    if use_reasoning_trace:
        print(
            f"\n  -- Stage 1a: Reasoning trace via sync R1 ({len(battles)} battles, parallel) --"
        )
        # Resume from cached trace file if available.
        traces_path = os.path.join(output_dir, "reasoning_traces.jsonl")
        cached: dict = {}
        if os.path.exists(traces_path):
            with open(traces_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if rec.get("battle_id") and rec.get("trace"):
                                cached[rec["battle_id"]] = rec["trace"]
                        except json.JSONDecodeError:
                            pass
            print(f"  Resuming: {len(cached)} traces already cached")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        sync_client = get_bedrock_client(region)
        reasoning_traces = dict(cached)
        todo = [b for b in battles if b.get("id") not in reasoning_traces]
        print(f"  Generating {len(todo)} new traces (R1 sync, 30 workers)...")
        if todo:
            with ThreadPoolExecutor(max_workers=30) as ex:
                futures = {
                    ex.submit(
                        _generate_reasoning_trace,
                        sync_client,
                        _reasoning_trace_model(),
                        b,
                    ): b
                    for b in todo
                }
                for done, fut in enumerate(as_completed(futures), 1):
                    b = futures[fut]
                    bid = b.get("id", "unknown")
                    try:
                        trace = fut.result()
                    except Exception as e:
                        print(f"  WARN: trace failed for {bid}: {e}")
                        trace = None
                    if trace:
                        reasoning_traces[bid] = trace
                    if done % 50 == 0 or done == len(todo):
                        print(f"  Traces {done}/{len(todo)}")
        # Persist; downstream consumers + reruns can resume.
        with open(traces_path, "w") as f:
            for bid, t in reasoning_traces.items():
                f.write(json.dumps({"battle_id": bid, "trace": t}) + "\n")
        print(f"  Wrote {len(reasoning_traces)} traces to {traces_path}")

    # Stage 1b: criteria batch.
    print("\n  -- Stage 1b: Criteria batch --")
    batch_path = prepare_batch_input(
        battles,
        model_id,
        output_dir,
        k_per_pair=k_per_pair,
        reasoning_traces=reasoning_traces,
        focus_theme=focus_theme,
    )
    result_files = _run_batch_job(
        batch_path,
        model_id,
        output_dir,
        job_prefix="premise-extract",
        arena_name=arena_name,
        output_subdir="batch_output",
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        region=region,
    )

    # Parse results
    criteria = parse_batch_results(result_files, model_id)
    print(f"  Total extracted: {len(criteria)} criteria from {len(battles)} battles")
    return criteria


# ============================================================
# Stage 1.5: Sample-level Revision
# ============================================================

PREDICT_FROM_CRITERIA_PROMPT = """You are evaluating two AI model responses, A and B, to a user prompt. You will NOT see the responses themselves — only a list of evaluation criteria, each with the score that response A and response B received.

USER PROMPT:
{prompt}

CRITERIA AND SCORES:
{criteria_block}

YOUR TASK:
Based ONLY on the criteria and their scores, predict which response a careful human evaluator would prefer.

OUTPUT FORMAT (strict JSON):
{{"predicted_winner": "A" | "B" | "tie"}}

Return ONLY the JSON object."""


REFINE_CRITERIA_PROMPT = """You previously extracted evaluation criteria from a comparison between two AI responses. The criteria failed to fully explain why the human chose the winner. Revise them so the criteria + scores would let an evaluator predict the true winner.

USER PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

TRUE WINNER: {winner}

CURRENT CRITERIA AND SCORES:
{criteria_block}

YOUR TASK:
Revise the criteria so that, given only their names + descriptions + scores, an evaluator would correctly identify {winner} as the winner. You may:
- Edit names or descriptions to be more discriminative
- Adjust scores (1-10) to better reflect actual quality differences
- Replace weak criteria with sharper ones
You MUST keep exactly {k} criteria.

OUTPUT FORMAT (strict JSON):
{{
  "criteria": [
    {{"name": "...", "description": "...", "model_a_score": <1-10>, "model_b_score": <1-10>}}
  ]
}}

Return ONLY the JSON object with exactly {k} criteria."""


def _format_criteria_block(criteria: list[ExtractedCriterion]) -> str:
    """Render criteria as a numbered prompt block."""
    lines = []
    for i, c in enumerate(criteria, 1):
        lines.append(
            f"{i}. {c.name}: {c.description} (A={c.model_a_score}, B={c.model_b_score})"
        )
    return "\n".join(lines)


def _build_predict_record(
    battle: dict,
    criteria: list[ExtractedCriterion],
    model_id: str,
) -> dict | None:
    battle_id = battle.get("id", "unknown")
    if not criteria:
        return None
    conv_a = battle.get("conversation_a", [])
    conv_b = battle.get("conversation_b", [])
    prompt_a, _ = extract_conversation_text(conv_a)
    prompt_b, _ = extract_conversation_text(conv_b)
    prompt = prompt_a or prompt_b
    if not prompt:
        return None
    eval_prompt = PREDICT_FROM_CRITERIA_PROMPT.format(
        prompt=prompt,
        criteria_block=_format_criteria_block(criteria),
    )
    return {
        "recordId": battle_id,
        "modelInput": get_batch_model_input(model_id, eval_prompt, 0.0, 200),
    }


def _build_refine_record(
    battle: dict,
    criteria: list[ExtractedCriterion],
    model_id: str,
    k: int,
) -> dict | None:
    battle_id = battle.get("id", "unknown")
    if not criteria:
        return None
    conv_a = battle.get("conversation_a", [])
    conv_b = battle.get("conversation_b", [])
    prompt_a, response_a = extract_conversation_text(conv_a)
    prompt_b, response_b = extract_conversation_text(conv_b)
    prompt = prompt_a or prompt_b
    if not prompt or not response_a or not response_b:
        return None
    eval_prompt = REFINE_CRITERIA_PROMPT.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        winner=battle.get("winner", "unknown"),
        criteria_block=_format_criteria_block(criteria),
        k=k,
    )
    max_tokens = max(2000, 200 * k + 400)
    return {
        "recordId": battle_id,
        "modelInput": get_batch_model_input(model_id, eval_prompt, 0.1, max_tokens),
    }


def _parse_predict_results(result_files: list, model_id: str) -> dict:
    """Return {battle_id: predicted_winner_str}."""
    preds = {}
    for rf in result_files:
        with open(rf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bid = rec.get("recordId", "unknown")
                text = extract_batch_model_response(
                    rec.get("modelOutput", {}), model_id
                )
                if not text:
                    continue
                try:
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    if m:
                        parsed = json.loads(m.group(0))
                        pw = str(parsed.get("predicted_winner", "")).strip().lower()
                        if pw in ("a", "model_a", "response_a"):
                            preds[bid] = "model_a"
                        elif pw in ("b", "model_b", "response_b"):
                            preds[bid] = "model_b"
                        elif pw in ("tie", "draw"):
                            preds[bid] = "tie"
                except (json.JSONDecodeError, ValueError):
                    pass
    return preds


def _parse_refine_results(
    result_files: list,
    model_id: str,
) -> dict:
    """Return {battle_id: [ExtractedCriterion, ...]} with revised criteria."""
    revised = {}
    for rf in result_files:
        with open(rf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bid = rec.get("recordId", "unknown")
                text = extract_batch_model_response(
                    rec.get("modelOutput", {}), model_id
                )
                if not text:
                    continue
                try:
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    if not m:
                        continue
                    parsed = json.loads(m.group(0))
                    raw = parsed.get("criteria", [])
                    if isinstance(raw, dict):
                        raw = [
                            {
                                "name": k,
                                **(
                                    v
                                    if isinstance(v, dict)
                                    else {"description": str(v)}
                                ),
                            }
                            for k, v in raw.items()
                        ]
                    out = []
                    for c in raw:
                        if not isinstance(c, dict):
                            continue
                        out.append(
                            ExtractedCriterion(
                                battle_id=bid,
                                name=c.get("name", "unknown"),
                                description=c.get("description", ""),
                                model_a_score=int(c.get("model_a_score", 5)),
                                model_b_score=int(c.get("model_b_score", 5)),
                            )
                        )
                    if out:
                        revised[bid] = out
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
    return revised


def _winrate_against_truth(battles: list, predictions: dict) -> tuple[float, int, int]:
    """Compare {battle_id: predicted_winner} against battle['winner']. Excludes ties on either side."""
    correct = 0
    total = 0
    for b in battles:
        bid = b.get("id", "unknown")
        true_w = b.get("winner", "")
        pred = predictions.get(bid)
        if pred is None or true_w not in ("model_a", "model_b") or pred == "tie":
            continue
        total += 1
        if pred == true_w:
            correct += 1
    return (correct / total if total else 0.0), correct, total


def _group_criteria_by_battle(
    criteria: list[ExtractedCriterion],
) -> dict:
    by_battle = defaultdict(list)
    for c in criteria:
        by_battle[c.battle_id].append(c)
    return by_battle


def run_sample_revision_batch(
    model_id: str,
    arena_file: str,
    output_dir: str,
    criteria: list[ExtractedCriterion],
    max_samples: int | None,
    k_per_pair: int,
    s3_bucket: str,
    role_arn: str,
    region: str,
) -> tuple[list[ExtractedCriterion], dict]:
    """Run pre-predict / refine / post-predict via batch inference.

    Returns (revised_criteria, metrics_dict).
    """
    print("\n=== Stage 1.5: Sample-level Revision (Batch) ===")
    battles = _load_battles(arena_file, max_samples)
    arena_name = os.path.splitext(os.path.basename(arena_file))[0]
    by_battle = _group_criteria_by_battle(criteria)

    def _write_records(records: list, fname: str) -> str:
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        return path

    # --- Pre-predict ---
    print("\n  -- Pre-predict --")
    pre_records = [
        rec
        for b in battles
        if (
            rec := _build_predict_record(
                b, by_battle.get(b.get("id", ""), []), model_id
            )
        )
    ]
    pre_path = _write_records(pre_records, "batch_predict_pre_input.jsonl")
    pre_files = _run_batch_job(
        pre_path,
        model_id,
        output_dir,
        job_prefix="premise-predict-pre",
        arena_name=arena_name,
        output_subdir="batch_predict_pre_output",
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        region=region,
    )
    pre_preds = _parse_predict_results(pre_files, model_id)
    pre_acc, pre_c, pre_t = _winrate_against_truth(battles, pre_preds)
    print(f"  Pre-revision accuracy: {pre_c}/{pre_t} = {pre_acc:.3f}")

    # --- Refine ---
    print("\n  -- Refine --")
    refine_records = [
        rec
        for b in battles
        if (
            rec := _build_refine_record(
                b, by_battle.get(b.get("id", ""), []), model_id, k_per_pair
            )
        )
    ]
    refine_path = _write_records(refine_records, "batch_refine_input.jsonl")
    refine_files = _run_batch_job(
        refine_path,
        model_id,
        output_dir,
        job_prefix="premise-refine",
        arena_name=arena_name,
        output_subdir="batch_refine_output",
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        region=region,
    )
    revised_map = _parse_refine_results(refine_files, model_id)
    # Merge revised in; fall back to original where revision missing.
    revised_criteria: list[ExtractedCriterion] = []
    for b in battles:
        bid = b.get("id", "unknown")
        if bid in revised_map:
            revised_criteria.extend(revised_map[bid])
        else:
            revised_criteria.extend(by_battle.get(bid, []))
    print(
        f"  Revised criteria for {len(revised_map)}/{len(battles)} battles "
        f"(total {len(revised_criteria)})"
    )

    # --- Post-predict ---
    print("\n  -- Post-predict --")
    post_by_battle = _group_criteria_by_battle(revised_criteria)
    post_records = [
        rec
        for b in battles
        if (
            rec := _build_predict_record(
                b, post_by_battle.get(b.get("id", ""), []), model_id
            )
        )
    ]
    post_path = _write_records(post_records, "batch_predict_post_input.jsonl")
    post_files = _run_batch_job(
        post_path,
        model_id,
        output_dir,
        job_prefix="premise-predict-post",
        arena_name=arena_name,
        output_subdir="batch_predict_post_output",
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        region=region,
    )
    post_preds = _parse_predict_results(post_files, model_id)
    post_acc, post_c, post_t = _winrate_against_truth(battles, post_preds)
    print(f"  Post-revision accuracy: {post_c}/{post_t} = {post_acc:.3f}")

    metrics = {
        "sample_revise_pre_winrate": pre_acc,
        "sample_revise_post_winrate": post_acc,
        "sample_revise_pre_correct": pre_c,
        "sample_revise_pre_total": pre_t,
        "sample_revise_post_correct": post_c,
        "sample_revise_post_total": post_t,
        "sample_revise_n_revised": len(revised_map),
    }
    return revised_criteria, metrics


# ============================================================
# Stage 2: Iterative Consolidation
# ============================================================


def _embed_single(
    bedrock, model_id: str, text: str, max_retries: int = 5
) -> list[float]:
    """Embed a single text with exponential backoff on throttling."""
    import time

    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps({"inputText": text[:8000]}),
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except Exception as e:
            err_str = str(e)
            if "ThrottlingException" in err_str or "Too many requests" in err_str:
                wait = min(2**attempt + 0.5, 30)
                time.sleep(wait)
                continue
            print(f"  Embedding error: {e}")
            return [0.0] * 1024
    print(f"  Embedding failed after {max_retries} retries (throttled)")
    return [0.0] * 1024


def generate_embeddings(
    client,
    texts: list[str],
    model_id: str | None = None,
    cache_path: str | None = None,
    max_workers: int = 10,
) -> np.ndarray:
    """Generate embeddings with caching and concurrent API calls.

    Optimizations over naive sequential approach:
    1. Cache: saves/loads .npy file so re-runs skip the API entirely
    2. Concurrency: ThreadPoolExecutor (default 10 workers) for parallel Titan calls
    3. Retry: exponential backoff on ThrottlingException
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from premise.providers import local_hash_embedding, provider_name

    if model_id is None:
        model_id = (
            get_config().model("embedding")
            if provider_name() == "bedrock"
            else "local:hashing"
        )

    # Check cache first
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            print(f"  Loaded {len(texts)} embeddings from cache: {cache_path}")
            return cached
        else:
            print(
                f"  Cache shape mismatch ({cached.shape[0]} vs {len(texts)}), regenerating"
            )

    if model_id == "local:hashing" or provider_name() != "bedrock":
        print(f"  Generating {len(texts)} local hashing embeddings...")
        result = np.array([local_hash_embedding(text) for text in texts])
        if cache_path:
            np.save(cache_path, result)
            print(f"  Cached embeddings to: {cache_path}")
        return result

    print(f"  Generating {len(texts)} embeddings ({max_workers} concurrent workers)...")
    embeddings = [[0.0] * 1024] * len(texts)  # pre-fill with zero vectors
    done_count = 0

    def _embed_task(idx_text):
        idx, text = idx_text
        return idx, _embed_single(client, model_id, text)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_embed_task, (i, t)): i for i, t in enumerate(texts)}
        for future in as_completed(futures):
            idx, emb = future.result()
            embeddings[idx] = emb
            done_count += 1
            if done_count % 100 == 0 or done_count == len(texts):
                print(f"  Embedded {done_count}/{len(texts)}")

    result = np.array(embeddings)

    # Save to cache
    if cache_path:
        np.save(cache_path, result)
        print(f"  Cached embeddings to: {cache_path}")

    return result


def cluster_traits(
    embeddings: np.ndarray, target_cluster_size: int = 20
) -> list[list[int]]:
    """Cluster traits by cosine similarity, capping cluster size.

    Uses AgglomerativeClustering then splits any oversized clusters
    via recursive sub-clustering so no cluster exceeds 2x target size.

    Zero-vector embeddings (from failed embed calls) are routed to
    singleton clusters since cosine distance is undefined for them.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    if len(embeddings) <= target_cluster_size:
        return [list(range(len(embeddings)))]

    # Route zero-vector rows (failed embeddings) to singleton clusters
    # so the remaining set can be safely clustered with cosine distance.
    norms = np.linalg.norm(embeddings, axis=1)
    nonzero_idx = np.where(norms > 0)[0]
    zero_idx = np.where(norms == 0)[0]
    if len(zero_idx) > 0:
        print(
            f"  WARN: {len(zero_idx)}/{len(embeddings)} embeddings are zero (failed); routing to singletons"
        )
    if len(nonzero_idx) <= target_cluster_size:
        return [[i] for i in zero_idx] + (
            [list(nonzero_idx)] if len(nonzero_idx) > 0 else []
        )

    sub = embeddings[nonzero_idx]
    normed = normalize(sub)
    n_clusters = max(1, len(sub) // target_cluster_size)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(normed)

    raw_clusters = defaultdict(list)
    for sub_pos, label in enumerate(labels):
        raw_clusters[label].append(int(nonzero_idx[sub_pos]))

    # Split oversized clusters (> 2x target) via sub-clustering on the original embeddings.
    max_size = target_cluster_size * 2
    final_clusters: list[list[int]] = []
    for indices in raw_clusters.values():
        if len(indices) <= max_size:
            final_clusters.append(indices)
        else:
            sub_emb = embeddings[indices]
            sub_clusters = cluster_traits(sub_emb, target_cluster_size)
            for sc in sub_clusters:
                final_clusters.append([indices[i] for i in sc])

    # Append zero-vector singletons.
    for i in zero_idx:
        final_clusters.append([int(i)])

    return final_clusters


SUBGROUP_PROMPT = """You are an expert at analyzing evaluation criteria for AI model responses. Identify groups of traits that measure the SAME fundamental quality and should be merged.

## Traits to Analyze

{traits_text}

## Instructions

1. Identify traits that are TRUE DUPLICATES — they would produce identical scores on the same response because they measure the exact same thing.
2. Group only those true duplicates for merging.
3. Keep traits SEPARATE if they measure different qualities, even slightly.

## CRITICAL: Operational Distinctness Test

Before proposing a merge, apply this test: "Could a response score HIGH on trait A but LOW on trait B?" If YES → they are DIFFERENT traits. Do NOT merge them.

Examples of traits that SHOULD be merged (true duplicates):
- "avoid fabricated claims" + "do not invent unsupported facts" → same quality, different wording.
- "use bullet points for lists" + "format lists with bullet markers" → same quality.

Examples of traits that must NOT be merged (operationally distinct):
- "factual accuracy about biology" + "factual accuracy about SQL" → a response can be correct on one and wrong on the other. KEEP SEPARATE.
- "completeness of analysis" + "completeness of output format" → one is about depth of reasoning, the other about structural coverage. KEEP SEPARATE.
- "tone matching" + "tone politeness" → a response can match the user's informal tone without being polite. KEEP SEPARATE.
- "conciseness" + "completeness" → directly opposed on some responses. KEEP SEPARATE.

## Size Limit

No subgroup may contain more than 5 traits. If you find a cluster of >5 that seem related, keep only the 2-3 closest duplicates and leave the rest unmerged. Err on the side of smaller groups.

## Default Bias

When in doubt, keep traits SEPARATE. Merging two distinct traits destroys information permanently. Leaving two duplicates unmerged wastes only a small amount of space.

## Output Format

Respond with a JSON object:
```json
{{
  "subgroups": [["trait_id_1", "trait_id_2", ...], ...],
  "unmerged": ["trait_id_x", "trait_id_y", ...]
}}
```

Rules:
- Every trait_id MUST appear exactly once (in a subgroup or unmerged)
- Subgroups must have 2-5 traits (no larger)
- When in doubt, keep traits separate

Respond ONLY with the JSON object."""


DESCRIPTION_PROMPT = """Create a single consolidated evaluation rubric that captures the common quality measured by all traits below.

## Traits to Consolidate

{traits_text}

## Instructions

Create a consolidated trait whose description is a should-statement that can be judged objectively and deterministically — another evaluator should read it and reliably score the same way. The consolidated trait should:
1. Generalize the common quality, abstracting away specifics
2. Be specific enough to be used as an evaluation rubric
3. Describe observable model behaviors
4. Avoid vague terms like "good" or "appropriate" without specifics

## ATOMICITY (critical)

The consolidated trait must measure EXACTLY ONE quality, even when the input traits collectively touch multiple qualities:
- Pick the SINGLE most central/measurable quality across the input traits.
- Do NOT join two qualities with "and", "with", "while", "including", or em-dashes.
- The trait_id must NOT contain "_and_", "_with_", or "_including_".
- A response should be scoreable on this trait independently of any other trait.

If the input traits clearly cover two distinct qualities, do NOT bundle them — pick the dominant one and leave the other for a different consolidated trait.

## Examples of well-formed (atomic) should-statements

- "The response should not invent citations that do not exist."
- "The response should use named, real-world examples to ground its claims."
- "The response should preserve the original numerical units when restating quantities."

## Examples of BAD (bundled) consolidations to avoid

- "The response should be accurate and use clear examples." (joins two qualities)
- "The response should be concise while preserving completeness." (joins two qualities)
- "The response should provide structured reasoning with verified evidence." (joins two qualities)

## Output Format

```json
{{
  "trait_id": "concise_snake_case_name",
  "trait_description": "Atomic should-statement (15-30 words) describing ONE quality"
}}
```

Respond ONLY with the JSON object."""


# Maximum number of criteria to include in a single sub-group prompt.
# Each criterion is ~100-150 tokens; keeping this at 150 ensures the prompt
# stays well within the 163K context window of DeepSeek-V3 (~22K tokens).
MAX_CRITERIA_PER_SUBGROUP_PROMPT = 150


def _identify_subgroups_batch(
    client, model_id: str, traits: list[Trait], cluster_indices: list[int]
) -> tuple[list[list[int]], list[int]]:
    """Identify subgroups for a single batch that fits within the context window."""
    cluster_traits = [traits[i] for i in cluster_indices]
    traits_text = "\n".join(
        f"- **{t.trait_id}**: {t.description}" for t in cluster_traits
    )

    prompt = SUBGROUP_PROMPT.format(traits_text=traits_text)
    response = invoke_model(client, model_id, prompt, temperature=0.0, max_tokens=4096)

    if not response:
        return [], cluster_indices

    # Parse response
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            return [], cluster_indices
    except json.JSONDecodeError:
        return [], cluster_indices

    # Map trait_ids back to indices
    id_to_idx = {
        cluster_traits[i].trait_id: cluster_indices[i]
        for i in range(len(cluster_traits))
    }

    def _to_str(val):
        """Coerce trait ID to string (model may return dicts or other types)."""
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return val.get("trait_id", val.get("id", ""))
        return str(val)

    subgroups = []
    seen = set()
    for group in parsed.get("subgroups", []):
        tids = [_to_str(v) for v in group]
        indices = [
            id_to_idx[tid] for tid in tids if tid in id_to_idx and tid not in seen
        ]
        if len(indices) >= 2:
            subgroups.append(indices)
            seen.update(traits[i].trait_id for i in indices)

    unmerged_raw = [_to_str(v) for v in parsed.get("unmerged", [])]
    unmerged = [
        id_to_idx[tid] for tid in unmerged_raw if tid in id_to_idx and tid not in seen
    ]
    # Add any missing
    for i in cluster_indices:
        if traits[i].trait_id not in seen and i not in unmerged:
            unmerged.append(i)

    return subgroups, unmerged


def identify_subgroups(
    client, model_id: str, traits: list[Trait], cluster_indices: list[int]
) -> tuple[list[list[int]], list[int]]:
    """Use LLM to identify subgroups of semantically equivalent traits within a cluster.

    If the cluster exceeds MAX_CRITERIA_PER_SUBGROUP_PROMPT, it is split into
    smaller batches that are each processed independently, and the results are
    merged.  This prevents context-window overflow errors on large clusters.
    """
    if len(cluster_indices) <= MAX_CRITERIA_PER_SUBGROUP_PROMPT:
        return _identify_subgroups_batch(client, model_id, traits, cluster_indices)

    # Split oversized cluster into batches
    batch_size = MAX_CRITERIA_PER_SUBGROUP_PROMPT
    batches = [
        cluster_indices[i : i + batch_size]
        for i in range(0, len(cluster_indices), batch_size)
    ]
    print(
        f"    Cluster too large ({len(cluster_indices)} traits), splitting into {len(batches)} batches"
    )

    all_subgroups: list[list[int]] = []
    all_unmerged: list[int] = []
    for b_idx, batch in enumerate(batches):
        sg, um = _identify_subgroups_batch(client, model_id, traits, batch)
        all_subgroups.extend(sg)
        all_unmerged.extend(um)

    return all_subgroups, all_unmerged


def generate_consolidated_description(
    client, model_id: str, traits: list[Trait], indices: list[int]
) -> Trait | None:
    """Generate a consolidated trait description for a subgroup."""
    subgroup_traits = [traits[i] for i in indices]
    traits_text = "\n".join(
        f"- **{t.trait_id}**: {t.description}" for t in subgroup_traits
    )

    prompt = DESCRIPTION_PROMPT.format(traits_text=traits_text)
    response = invoke_model(client, model_id, prompt, temperature=0.0, max_tokens=1024)

    if not response:
        return None

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            trait_id = parsed.get("trait_id", "").strip()
            description = parsed.get("trait_description", "").strip()
            if trait_id and description and len(description.split()) >= 10:
                source_ids = []
                source_count = 0
                for t in subgroup_traits:
                    source_ids.extend(t.source_ids or [t.trait_id])
                    source_count += t.source_count
                return Trait(
                    trait_id=trait_id,
                    description=description,
                    source_count=source_count,
                    source_ids=source_ids,
                )
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _load_level(level: int, output_dir: str) -> list[Trait] | None:
    """Load traits from a consolidation level file, or None if it doesn't exist."""
    filepath = os.path.join(output_dir, f"traits_level_{level}.jsonl")
    if not os.path.exists(filepath):
        return None
    traits = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                traits.append(
                    Trait(
                        trait_id=d["trait_id"],
                        description=d["description"],
                        source_count=d.get("source_count", 1),
                        source_ids=d.get("source_ids", []),
                    )
                )
    return traits if traits else None


def run_consolidation(
    client,
    model_id: str,
    criteria: list[ExtractedCriterion],
    output_dir: str,
    target_cluster_size: int = 20,
    max_levels: int = 10,
    embedding_model: str | None = None,
    max_embed_workers: int = 10,
    target_pool_size: int = 100,
) -> list[Trait]:
    """Run iterative consolidation: embed -> cluster -> sub-group -> describe -> repeat.

    Stops when EITHER:
    - trait count <= target_pool_size (reached the desired selection-pool size), OR
    - reduction < 5% (natural convergence), OR
    - max_levels reached.

    The target_pool_size is the trait count we want the LLM top-K selection to
    operate on. Defaults to 100 (enough for a 20-trait final pick with diversity).

    Supports resume: scans for existing level files and resumes from the highest one.
    """
    print("\n=== Stage 2: Iterative Consolidation ===")

    # Check for resume: find highest existing level
    resume_level = -1
    for lvl in range(max_levels, -1, -1):
        loaded = _load_level(lvl, output_dir)
        if loaded is not None:
            resume_level = lvl
            traits = loaded
            break

    if resume_level >= 0:
        print(f"  Resuming from level {resume_level} ({len(traits)} traits)")
        start_level = resume_level + 1
    else:
        # Convert extracted criteria to initial traits (level 0)
        traits = []
        for i, c in enumerate(criteria):
            traits.append(
                Trait(
                    trait_id=f"criterion_{i:05d}",
                    description=f"{c.name}: {c.description}",
                    source_count=1,
                    source_ids=[c.battle_id],
                )
            )
        print(f"  Starting with {len(traits)} criteria")
        _save_level(traits, 0, output_dir)
        start_level = 1

    for level in range(start_level, max_levels + 1):
        print(f"\n  --- Level {level} (input: {len(traits)} traits) ---")

        # Stop when total trait count reaches the target pool range.
        # We want ~100-200 traits for the LLM top-K selection to pick from.
        # But always run at least one consolidation pass (level 1) so that
        # duplicate criteria from different battles get merged and accumulate
        # meaningful source_counts.
        if level >= 2 and len(traits) <= target_pool_size:
            print(
                f"  Converged: {len(traits)} traits <= target_pool_size {target_pool_size}"
            )
            break

        # Step 1: Embed (with caching per level)
        print(f"  Embedding {len(traits)} traits...")
        texts = [t.description for t in traits]
        cache_path = os.path.join(output_dir, f"embeddings_level_{level}.npy")
        embeddings = generate_embeddings(
            client,
            texts,
            embedding_model,
            cache_path=cache_path,
            max_workers=max_embed_workers,
        )

        # Step 2: Cluster
        clusters = cluster_traits(embeddings, target_cluster_size)
        print(f"  Formed {len(clusters)} clusters")

        # Step 3 & 4: Sub-group and Describe (concurrent across clusters)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _process_cluster(cluster_indices, traits=traits):
            """Process a single cluster: identify subgroups, generate descriptions.

            `traits` is bound as a default so each worker sees the list for the
            level it was submitted from, not whatever `traits` was rebound to by
            the time the future runs.
            """
            if len(cluster_indices) == 1:
                return [traits[cluster_indices[0]]]

            subgroups, unmerged = identify_subgroups(
                client, model_id, traits, cluster_indices
            )

            result = []
            for sg_indices in subgroups:
                consolidated = generate_consolidated_description(
                    client, model_id, traits, sg_indices
                )
                if consolidated:
                    result.append(consolidated)
                else:
                    counts = {t: traits[t].source_count for t in sg_indices}
                    best = max(counts, key=counts.__getitem__)
                    result.append(traits[best])

            for idx in unmerged:
                result.append(traits[idx])
            return result

        new_traits = []
        multi_clusters = [(i, c) for i, c in enumerate(clusters) if len(c) > 1]
        single_clusters = [c for c in clusters if len(c) == 1]

        # Pass through singletons
        for c in single_clusters:
            new_traits.append(traits[c[0]])

        # Process multi-trait clusters concurrently (30 workers).
        # Bedrock per-account concurrency easily absorbs this for one
        # discovery run; reduce if running multiple discoveries simultaneously.
        n_workers = min(30, len(multi_clusters)) if multi_clusters else 1
        print(
            f"  Processing {len(multi_clusters)} clusters ({n_workers} concurrent workers)..."
        )
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_cluster, c): i for i, c in multi_clusters
            }
            for done, future in enumerate(as_completed(futures), 1):
                cluster_result = future.result()
                new_traits.extend(cluster_result)
                if done % 50 == 0 or done == len(multi_clusters):
                    print(f"    Consolidated {done}/{len(multi_clusters)} clusters")

        prev_count = len(traits)
        traits = new_traits
        print(f"  Level {level}: {prev_count} -> {len(traits)} traits")

        # Save level output
        _save_level(traits, level, output_dir)

        # Check convergence
        if len(traits) >= prev_count * 0.95:
            print("  Convergence: reduction < 5%, stopping")
            break

    return traits


def _save_level(traits: list[Trait], level: int, output_dir: str):
    """Save traits for a consolidation level."""
    filepath = os.path.join(output_dir, f"traits_level_{level}.jsonl")
    with open(filepath, "w") as f:
        f.writelines(
            json.dumps(
                {
                    "trait_id": t.trait_id,
                    "description": t.description,
                    "source_count": t.source_count,
                    "source_ids": t.source_ids[:10],  # truncate for readability
                }
            )
            + "\n"
            for t in traits
        )


# ============================================================
# Stage 2.5: Trait-level (Rubric) Revision
# ============================================================

SCORE_RUBRIC_PROMPT = """You are scoring two AI model responses, A and B, against a fixed rubric. Score each response on every rubric trait from 1 (poor) to 10 (excellent).

USER PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

RUBRIC:
{rubric_block}

OUTPUT FORMAT (strict JSON):
{{
  "scores": [
    {{"trait_id": "<id>", "model_a_score": <1-10>, "model_b_score": <1-10>}}
  ]
}}

Return ONLY the JSON object. Include every trait in the rubric."""


REFINE_RUBRIC_PROMPT = """You have a rubric that is supposed to discriminate between AI responses well enough that summing per-trait score deltas predicts the human-preferred response. The rubric currently misclassifies many cases.

CURRENT RUBRIC:
{rubric_block}

EXAMPLES THE RUBRIC GETS WRONG (the rubric predicted the wrong winner; true winner is shown):
{wrong_block}

YOUR TASK:
Revise the rubric so it discriminates better. Keep the same trait IDs and the same number of traits. You may rewrite descriptions to be sharper, more behavior-based, more measurable. Do NOT change trait_ids.

OUTPUT FORMAT (strict JSON):
{{
  "rubrics": [
    {{"trait_id": "<id>", "description": "<revised description>"}}
  ]
}}

Return ONLY the JSON object."""


def _format_rubric_block(traits: list[Trait]) -> str:
    return "\n".join(f"- **{t.trait_id}**: {t.description}" for t in traits)


def _build_score_record(
    battle: dict,
    traits: list[Trait],
    model_id: str,
) -> dict | None:
    battle_id = battle.get("id", "unknown")
    conv_a = battle.get("conversation_a", [])
    conv_b = battle.get("conversation_b", [])
    prompt_a, response_a = extract_conversation_text(conv_a)
    prompt_b, response_b = extract_conversation_text(conv_b)
    prompt = prompt_a or prompt_b
    if not prompt or not response_a or not response_b:
        return None
    eval_prompt = SCORE_RUBRIC_PROMPT.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        rubric_block=_format_rubric_block(traits),
    )
    max_tokens = max(2000, 80 * len(traits) + 400)
    return {
        "recordId": battle_id,
        "modelInput": get_batch_model_input(model_id, eval_prompt, 0.0, max_tokens),
    }


def _parse_score_results(result_files: list, model_id: str) -> dict:
    """Return {battle_id: {trait_id: (a_score, b_score)}}."""
    out = {}
    for rf in result_files:
        with open(rf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bid = rec.get("recordId", "unknown")
                text = extract_batch_model_response(
                    rec.get("modelOutput", {}), model_id
                )
                if not text:
                    continue
                try:
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    if not m:
                        continue
                    parsed = json.loads(m.group(0))
                    scores = {}
                    for s in parsed.get("scores", []):
                        if not isinstance(s, dict):
                            continue
                        tid = s.get("trait_id")
                        if not tid:
                            continue
                        try:
                            a = int(s.get("model_a_score", 5))
                            b = int(s.get("model_b_score", 5))
                        except (TypeError, ValueError):
                            continue
                        scores[tid] = (a, b)
                    if scores:
                        out[bid] = scores
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
    return out


def _predict_from_scores(scored: dict) -> dict:
    """{battle_id: {trait_id: (a,b)}} -> {battle_id: predicted_winner}.

    Uses sum-of-deltas across all traits in the rubric. Ties on exact zero map to "tie".
    """
    preds = {}
    for bid, trait_scores in scored.items():
        delta = sum(a - b for (a, b) in trait_scores.values())
        if delta > 0:
            preds[bid] = "model_a"
        elif delta < 0:
            preds[bid] = "model_b"
        else:
            preds[bid] = "tie"
    return preds


def run_trait_revision_batch(
    judge_model: str,
    arena_file: str,
    output_dir: str,
    traits: list[Trait],
    max_samples: int | None,
    s3_bucket: str,
    role_arn: str,
    region: str,
    n_wrong_examples: int = 20,
) -> tuple[list[Trait], dict]:
    """Score rubric on each battle, refine rubric on wrong cases, re-score.

    Returns (revised_traits, metrics_dict). The rubric keeps the same trait_ids.
    """
    print("\n=== Stage 2.5: Trait-level (Rubric) Revision (Batch) ===")
    if not traits:
        print("  No traits to revise; skipping.")
        return traits, {}

    battles = _load_battles(arena_file, max_samples)
    arena_name = os.path.splitext(os.path.basename(arena_file))[0]
    bid_to_battle = {b.get("id", "unknown"): b for b in battles}

    def _write_records(records: list, fname: str) -> str:
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    # --- Pre-score ---
    print("\n  -- Pre-score rubric --")
    pre_records = [
        rec for b in battles if (rec := _build_score_record(b, traits, judge_model))
    ]
    pre_path = _write_records(pre_records, "batch_score_pre_input.jsonl")
    pre_files = _run_batch_job(
        pre_path,
        judge_model,
        output_dir,
        job_prefix="premise-score-pre",
        arena_name=arena_name,
        output_subdir="batch_score_pre_output",
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        region=region,
    )
    pre_scored = _parse_score_results(pre_files, judge_model)
    pre_preds = _predict_from_scores(pre_scored)
    pre_acc, pre_c, pre_t = _winrate_against_truth(battles, pre_preds)
    print(f"  Pre-revision rubric accuracy: {pre_c}/{pre_t} = {pre_acc:.3f}")

    # --- Refine rubric on wrong cases (single call) ---
    print("\n  -- Refine rubric on wrong cases --")
    wrong_bids = [
        bid
        for bid, pred in pre_preds.items()
        if pred != "tie"
        and bid_to_battle.get(bid, {}).get("winner") in ("model_a", "model_b")
        and pred != bid_to_battle[bid]["winner"]
    ]
    sampled = wrong_bids[:n_wrong_examples]
    print(f"  Found {len(wrong_bids)} wrong predictions; sampling {len(sampled)}")

    wrong_blocks = []
    for bid in sampled:
        b = bid_to_battle[bid]
        scores = pre_scored.get(bid, {})
        score_lines = "\n".join(
            f"    - {tid}: A={a}, B={b_}" for tid, (a, b_) in scores.items()
        )
        conv_a = b.get("conversation_a", [])
        conv_b = b.get("conversation_b", [])
        prompt_a, _ = extract_conversation_text(conv_a)
        prompt_b, _ = extract_conversation_text(conv_b)
        wrong_blocks.append(
            f"Example battle {bid} — true winner: {b.get('winner')}\n"
            f"  User prompt: {(prompt_a or prompt_b)[:500]}\n"
            f"  Per-trait scores:\n{score_lines}"
        )
    wrong_block = "\n\n".join(wrong_blocks) if wrong_blocks else "(none)"

    refine_prompt = REFINE_RUBRIC_PROMPT.format(
        rubric_block=_format_rubric_block(traits),
        wrong_block=wrong_block,
    )
    client = get_bedrock_client(region)
    refined_text = invoke_model(
        client,
        judge_model,
        refine_prompt,
        temperature=0.1,
        max_tokens=max(4000, 200 * len(traits) + 800),
    )

    revised_traits = list(traits)
    if refined_text:
        try:
            m = re.search(r"\{.*\}", refined_text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                desc_map = {}
                for r in parsed.get("rubrics", []):
                    if isinstance(r, dict):
                        tid = r.get("trait_id")
                        d = r.get("description")
                        if tid and d:
                            desc_map[tid] = d
                revised_traits = [
                    Trait(
                        trait_id=t.trait_id,
                        description=desc_map.get(t.trait_id, t.description),
                        source_count=t.source_count,
                        source_ids=list(t.source_ids),
                    )
                    for t in traits
                ]
                n_changed = sum(
                    1
                    for t, rt in zip(traits, revised_traits)
                    if t.description != rt.description
                )
                print(f"  Revised descriptions for {n_changed}/{len(traits)} traits")
        except (json.JSONDecodeError, ValueError):
            print("  WARN: failed to parse rubric refinement; keeping original")

    # --- Post-score ---
    print("\n  -- Post-score rubric --")
    post_records = [
        rec
        for b in battles
        if (rec := _build_score_record(b, revised_traits, judge_model))
    ]
    post_path = _write_records(post_records, "batch_score_post_input.jsonl")
    post_files = _run_batch_job(
        post_path,
        judge_model,
        output_dir,
        job_prefix="premise-score-post",
        arena_name=arena_name,
        output_subdir="batch_score_post_output",
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        region=region,
    )
    post_scored = _parse_score_results(post_files, judge_model)
    post_preds = _predict_from_scores(post_scored)
    post_acc, post_c, post_t = _winrate_against_truth(battles, post_preds)
    print(f"  Post-revision rubric accuracy: {post_c}/{post_t} = {post_acc:.3f}")

    metrics = {
        "trait_revise_pre_winrate": pre_acc,
        "trait_revise_post_winrate": post_acc,
        "trait_revise_pre_correct": pre_c,
        "trait_revise_pre_total": pre_t,
        "trait_revise_post_correct": post_c,
        "trait_revise_post_total": post_t,
        "trait_revise_n_wrong_used": len(sampled),
    }
    return revised_traits, metrics


# ============================================================
# Main Pipeline
# ============================================================


def _load_criteria_from_file(criteria_file: str) -> list[ExtractedCriterion]:
    """Load previously saved criteria from JSONL."""
    criteria = []
    with open(criteria_file) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                criteria.append(
                    ExtractedCriterion(
                        battle_id=d.get("battle_id", "unknown"),
                        name=d.get("name", "unknown"),
                        description=d.get("description", ""),
                        model_a_score=d.get("model_a_score", 5),
                        model_b_score=d.get("model_b_score", 5),
                    )
                )
    return criteria


def _save_criteria_to_file(criteria: list[ExtractedCriterion], criteria_file: str):
    """Save criteria to JSONL."""
    with open(criteria_file, "w") as f:
        f.writelines(
            json.dumps(
                {
                    "battle_id": c.battle_id,
                    "name": c.name,
                    "description": c.description,
                    "model_a_score": c.model_a_score,
                    "model_b_score": c.model_b_score,
                }
            )
            + "\n"
            for c in criteria
        )


def run_trait_discovery(
    arena_file: str,
    output_dir: str,
    judge_model: str | None = None,
    embedding_model: str | None = None,
    max_samples: int | None = None,
    target_traits: int = 20,
    cluster_size: int | None = None,
    max_levels: int = 10,
    min_source_count: int = 10,
    skip_extraction: bool = False,
    max_embed_workers: int = 10,
    region: str = DEFAULT_REGION,
    no_batch: bool = False,
    s3_bucket: str = DEFAULT_S3_BUCKET,
    role_arn: str = DEFAULT_BATCH_ROLE_ARN,
    k_per_pair: int = 15,
    use_reasoning_trace: bool = False,
    focus_theme: str = "",
    revise: str = "none",
    final_topk_selection: str = "llm",
    selection_pool_size: int | None = None,
    logdet_eps: float = 0.5,
    logdet_min_increment: float = 0.001,
    logdet_patience: int = 3,
    mmr_lambda: float = 0.75,
    mmr_skip_genericity_filter: bool = False,
    mmr_disable_abstraction_rescue: bool = False,
):
    """Run the full trait discovery pipeline."""
    from premise.providers import model_for_role, provider_name

    config = get_config()
    active_provider = provider_name()
    if judge_model is None:
        judge_model = model_for_role("judge")
    if active_provider != "bedrock":
        no_batch = True
        if embedding_model is None:
            embedding_model = "local:hashing"
    elif embedding_model is None:
        embedding_model = model_for_role("embedding")
    if not no_batch:
        from premise.utils import resolve_batch_settings

        s3_bucket, role_arn = resolve_batch_settings(s3_bucket, role_arn)
    if cluster_size is None:
        cluster_size = target_traits
    if selection_pool_size is None:
        selection_pool_size = max(150, 3 * target_traits)
    assert revise in ("none", "sample", "trait", "both"), f"bad --revise: {revise}"
    from premise.rubric_discovery.selection import SUPPORTED_STRATEGIES, run_selection

    assert final_topk_selection in SUPPORTED_STRATEGIES, (
        f"bad --final-topk-selection: {final_topk_selection!r}; "
        f"supported: {SUPPORTED_STRATEGIES}"
    )

    arena_path = Path(arena_file).expanduser().resolve()
    batch_settings = {
        "s3_bucket": s3_bucket if not no_batch else None,
        "role_arn": role_arn if not no_batch else None,
    }
    identity = {
        "workflow_revision": DISCOVERY_WORKFLOW_REVISION,
        "input_preference_data": str(arena_path),
        "input_preference_data_sha256": file_sha256(arena_path),
        "judge_model": judge_model,
        "embedding_model": embedding_model,
        "max_samples": max_samples,
        "target_traits": target_traits,
        "cluster_size": cluster_size,
        "max_levels": max_levels,
        "min_source_count": min_source_count,
        "max_embed_workers": max_embed_workers,
        "region": region,
        "no_batch": no_batch,
        "batch_settings_sha256": (
            None if no_batch else canonical_sha256(batch_settings)
        ),
        "k_per_pair": k_per_pair,
        "use_reasoning_trace": use_reasoning_trace,
        "focus_theme": focus_theme,
        "revise": revise,
        "final_topk_selection": final_topk_selection,
        "selection_pool_size": selection_pool_size,
        "logdet_eps": logdet_eps,
        "logdet_min_increment": logdet_min_increment,
        "logdet_patience": logdet_patience,
        "mmr_lambda": mmr_lambda,
        "mmr_skip_genericity_filter": mmr_skip_genericity_filter,
        "mmr_disable_abstraction_rescue": mmr_disable_abstraction_rescue,
        **active_config_identity(config),
    }
    state_path, state = _prepare_discovery_state(
        output_dir,
        identity,
        skip_extraction=skip_extraction,
    )
    client = get_bedrock_client(region)

    print("PReMISE Trait Discovery Pipeline")
    print(f"  Arena data: {arena_file}")
    print(f"  Judge model: {judge_model}")
    print(f"  Provider: {active_provider}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Max samples: {max_samples or 'all'}")
    print(f"  Target traits: {target_traits}")
    print(f"  Cluster size: {cluster_size}")
    print(f"  K per pair: {k_per_pair}")
    print(f"  Reasoning trace: {use_reasoning_trace}")
    print(f"  Revise mode: {revise}")
    print(f"  Final top-K selection: {final_topk_selection}")
    print(f"  Selection pool size: {selection_pool_size}")
    print(f"  Batch inference: {'disabled' if no_batch else 'enabled'}")

    revision_metrics: dict = {}

    # Stage 1: Criteria Extraction (with resume support)
    criteria_file = os.path.join(output_dir, "extracted_criteria.jsonl")

    try:
        if os.path.exists(criteria_file) and (
            skip_extraction or os.path.getsize(criteria_file) > 0
        ):
            # Resume: reuse existing extracted criteria
            print(f"\n=== Stage 1: Resuming (reusing {criteria_file}) ===")
            criteria = _load_criteria_from_file(criteria_file)
            print(f"  Loaded {len(criteria)} criteria")
        elif no_batch:
            criteria = run_criteria_extraction(
                client,
                judge_model,
                arena_file,
                max_samples,
                k_per_pair=k_per_pair,
                use_reasoning_trace=use_reasoning_trace,
                focus_theme=focus_theme,
            )
            _save_criteria_to_file(criteria, criteria_file)
            print(f"  Saved {len(criteria)} criteria to {criteria_file}")
        else:
            criteria = run_criteria_extraction_batch(
                judge_model,
                arena_file,
                output_dir,
                max_samples=max_samples,
                s3_bucket=s3_bucket,
                role_arn=role_arn,
                region=region,
                k_per_pair=k_per_pair,
                use_reasoning_trace=use_reasoning_trace,
                focus_theme=focus_theme,
            )
            if not criteria:
                raise CriteriaExtractionError(
                    "batch criteria extraction produced no criteria"
                )
            _save_criteria_to_file(criteria, criteria_file)
            print(f"  Saved {len(criteria)} criteria to {criteria_file}")
        if not criteria:
            raise CriteriaExtractionError(
                f"{criteria_file} contains no criteria; choose a new output "
                "directory and rerun extraction"
            )
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        state["updated_at"] = _utc_now()
        _write_discovery_state(state_path, state)
        raise

    # Stage 1.5: Sample-level revision (optional)
    if revise in ("sample", "both"):
        revised_criteria_file = os.path.join(output_dir, "revised_criteria.jsonl")
        if (
            os.path.exists(revised_criteria_file)
            and os.path.getsize(revised_criteria_file) > 0
        ):
            print(f"\n=== Stage 1.5: Resuming (reusing {revised_criteria_file}) ===")
            criteria = _load_criteria_from_file(revised_criteria_file)
            print(f"  Loaded {len(criteria)} revised criteria")
        else:
            if no_batch:
                print(
                    "  WARN: --revise requires batch inference; skipping sample revision under --no-batch"
                )
            else:
                criteria, sample_metrics = run_sample_revision_batch(
                    judge_model,
                    arena_file,
                    output_dir,
                    criteria,
                    max_samples=max_samples,
                    k_per_pair=k_per_pair,
                    s3_bucket=s3_bucket,
                    role_arn=role_arn,
                    region=region,
                )
                _save_criteria_to_file(criteria, revised_criteria_file)
                revision_metrics.update(sample_metrics)
                with open(
                    os.path.join(output_dir, "sample_revision_metrics.json"), "w"
                ) as f:
                    json.dump(sample_metrics, f, indent=2)

    # Stage 2: Consolidation
    final_traits = run_consolidation(
        client,
        judge_model,
        criteria,
        output_dir,
        target_cluster_size=cluster_size,
        max_levels=max_levels,
        embedding_model=embedding_model,
        max_embed_workers=max_embed_workers,
        target_pool_size=selection_pool_size,
    )

    # When the LLM strategy handles selection, skip the hard min_source_count
    # cutoff — pass all consolidated traits sorted by prevalence. The LLM
    # prompt already prefers higher source_count and can merge low-count
    # paraphrases into high-count ones. The hard filter only removes candidates
    # the LLM could otherwise leverage.
    if final_topk_selection in ("llm", "mmr"):
        high_impact = sorted(final_traits, key=lambda t: -t.source_count)
    else:
        high_impact = [t for t in final_traits if t.source_count >= min_source_count]
        high_impact.sort(key=lambda t: -t.source_count)

    # Stage 2a: Final top-K selection (source_count | logdet | llm | mmr).
    # Caps the input pool, runs the chosen strategy, and produces per-trait
    # annotations (selected_by_strategy + strategy-specific keys).
    print(f"\n=== Stage 2a: Top-K selection ({final_topk_selection}) ===")
    selected_ids, selection_annotations, selection_extras = run_selection(
        final_topk_selection,
        high_impact,
        target_traits,
        selection_pool_size=selection_pool_size,
        region=region,
        embedding_model=embedding_model,
        logdet_eps=logdet_eps,
        logdet_min_increment=logdet_min_increment,
        logdet_patience=logdet_patience,
        mmr_lambda=mmr_lambda,
        mmr_skip_genericity_filter=mmr_skip_genericity_filter,
        mmr_disable_abstraction_rescue=mmr_disable_abstraction_rescue,
    )
    print(
        f"  Strategy={final_topk_selection} selected {len(selected_ids)} "
        f"of {min(len(high_impact), selection_pool_size)} pool candidates "
        f"(target K={target_traits})."
    )
    if selection_extras.get("strategy") == "logdet":
        with open(os.path.join(output_dir, "logdet_curve.json"), "w") as f:
            json.dump(selection_extras.get("curve", []), f, indent=2)
    elif selection_extras.get("strategy") == "llm":
        with open(os.path.join(output_dir, "llm_topk_merges.json"), "w") as f:
            json.dump(
                {
                    "merges": selection_extras.get("merges", []),
                    "notes": selection_extras.get("notes", ""),
                    "underflow": selection_extras.get("underflow", False),
                },
                f,
                indent=2,
            )
    elif selection_extras.get("strategy") == "mmr":
        with open(os.path.join(output_dir, "mmr_selection.json"), "w") as f:
            json.dump(
                {
                    "lambda": selection_extras.get("lambda"),
                    "n_pool_input": selection_extras.get("n_pool_input"),
                    "n_after_genericity": selection_extras.get("n_after_genericity"),
                    "n_dropped_prompt_specific": selection_extras.get(
                        "n_dropped_prompt_specific"
                    ),
                    "n_abstraction_rescued": selection_extras.get(
                        "n_abstraction_rescued", 0
                    ),
                    "abstraction_rescue_enabled": selection_extras.get(
                        "abstraction_rescue_enabled", False
                    ),
                    "rescued_traits": selection_extras.get("rescued_traits", []),
                    "mean_pairwise_cosine_sim": selection_extras.get(
                        "mean_pairwise_cosine_sim"
                    ),
                    "max_pairwise_cosine_sim": selection_extras.get(
                        "max_pairwise_cosine_sim"
                    ),
                    "selected_ids": selected_ids,
                },
                f,
                indent=2,
            )

    # The active rubric = traits selected by the active strategy, in
    # selection rank order (LLM order / log-volume rank / source_count desc).
    by_id = {t.trait_id: t for t in high_impact}
    # MMR can rescue synthetic abstracted traits that aren't in high_impact;
    # materialize them as Trait objects so downstream code can use them.
    for r in selection_extras.get("rescued_traits", []) or []:
        if r["trait_id"] not in by_id:
            by_id[r["trait_id"]] = Trait(
                trait_id=r["trait_id"],
                description=r["description"],
                source_count=r["source_count"],
                source_ids=list(r.get("source_ids", [])),
            )
    top_n = [by_id[tid] for tid in selected_ids if tid in by_id]

    # Stage 2.5: Trait-level (rubric) revision (optional). Operates on the
    # selected active subset only.
    if revise in ("trait", "both") and top_n:
        if no_batch:
            print(
                "  WARN: --revise requires batch inference; skipping trait revision under --no-batch"
            )
        else:
            revised_top_n, trait_metrics = run_trait_revision_batch(
                judge_model,
                arena_file,
                output_dir,
                top_n,
                max_samples=max_samples,
                s3_bucket=s3_bucket,
                role_arn=role_arn,
                region=region,
            )
            top_n = revised_top_n
            # Also update high_impact in-place where ids match.
            desc_map = {t.trait_id: t.description for t in revised_top_n}
            high_impact = [
                Trait(
                    trait_id=t.trait_id,
                    description=desc_map.get(t.trait_id, t.description),
                    source_count=t.source_count,
                    source_ids=t.source_ids,
                )
                for t in high_impact
            ]
            revision_metrics.update(trait_metrics)
            with open(
                os.path.join(output_dir, "trait_revision_metrics.json"), "w"
            ) as f:
                json.dump(trait_metrics, f, indent=2)

    print("\n=== Results ===")
    print(f"  Final traits after consolidation: {len(final_traits)}")
    print(f"  High-impact (source_count >= {min_source_count}): {len(high_impact)}")
    print(f"  Active subset ({final_topk_selection}): {len(top_n)}")

    # Helper that merges per-trait selection annotations into a rubric dict.
    # Backward-compat: with strategy=source_count, only `selected_by_strategy`
    # is present (which is True for top_n, False otherwise) and we omit it
    # entirely from the saved files to keep the legacy schema.
    def _trait_dict(t: "Trait") -> dict:
        d = {
            "id": t.trait_id,
            "category": "discovered",
            "description": t.description,
            "source_count": t.source_count,
        }
        if final_topk_selection != "source_count":
            ann = selection_annotations.get(t.trait_id, {})
            for k, v in ann.items():
                d[k] = v
        return d

    # Save ALL high-impact traits (full discovered set), with annotations
    # when a non-default strategy ran.
    all_traits_data = {
        "schema_version": "1.0" if final_topk_selection == "source_count" else "1.1",
        "rubrics": [_trait_dict(t) for t in high_impact],
    }

    all_traits_file = os.path.join(output_dir, "discovered_traits.json")
    with open(all_traits_file, "w") as f:
        json.dump(all_traits_data, f, indent=2)
    print(f"  All high-impact traits: {all_traits_file}")

    # Save top-N (= active subset) — the usable rubric set.
    top_n_data = {
        "schema_version": "1.0" if final_topk_selection == "source_count" else "1.1",
        "rubrics": [_trait_dict(t) for t in top_n],
    }

    top_n_file = os.path.join(output_dir, f"traits_top_{target_traits}.json")
    with open(top_n_file, "w") as f:
        json.dump(top_n_data, f, indent=2)
    print(f"  Top-{target_traits} selection: {top_n_file}")

    # Save consolidation summary
    summary = {
        "n_battles": max_samples or "all",
        "n_extracted_criteria": len(criteria),
        "n_final_traits": len(final_traits),
        "n_high_impact_traits": len(high_impact),
        "n_top_traits": len(top_n),
        "target_traits": target_traits,
        "min_source_count": min_source_count,
        "judge_model": judge_model,
        "k_per_pair": k_per_pair,
        "use_reasoning_trace": use_reasoning_trace,
        "revise": revise,
        "revision_metrics": revision_metrics,
        "final_topk_selection": final_topk_selection,
        "selection_pool_size": selection_pool_size,
        "selection_actual_k": len(top_n),
        "selection_extras": {
            k: v
            for k, v in selection_extras.items()
            if k
            in ("strategy", "underflow", "notes", "pool_size", "target_k", "actual_k")
        },
        "top_traits": [
            {
                "id": t.trait_id,
                "source_count": t.source_count,
                "description": t.description[:100],
            }
            for t in top_n
        ],
    }

    summary_file = os.path.join(output_dir, "consolidation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    state["status"] = "completed"
    state["completed_at"] = _utc_now()
    state["updated_at"] = state["completed_at"]
    _write_discovery_state(state_path, state)

    # Print top traits
    print("\n  Top discovered traits:")
    for t in high_impact[:10]:
        print(f"    {t.trait_id} ({t.source_count} sources): {t.description[:80]}...")

    return high_impact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PReMISE trait discovery pipeline")
    parser.add_argument(
        "--input-preference-data",
        dest="arena_data",
        help="Path to pairwise preference JSONL.",
    )
    parser.add_argument(
        "--arena-data",
        dest="arena_data",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        default=os.path.join(get_config().runtime("output_root"), "discover", "run"),
        help="Output directory (default: runtime.output_root/discover/run)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for extraction and consolidation (default: config role 'judge')",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Model for embeddings (default: config role 'embedding')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of battles to process",
    )
    parser.add_argument(
        "--target-traits",
        type=int,
        default=20,
        help="Target number of consolidated traits",
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=None,
        help="Traits per cluster (default: same as --target-traits). "
        "Larger values create bigger clusters with more merge candidates.",
    )
    parser.add_argument(
        "--max-levels", type=int, default=10, help="Max consolidation iterations"
    )
    parser.add_argument(
        "--min-source-count",
        type=int,
        default=10,
        help="Min source criteria count for final output",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip Stage 1 and reuse existing extracted_criteria.jsonl",
    )
    parser.add_argument(
        "--max-embed-workers",
        type=int,
        default=20,
        help="Concurrent workers for Titan embedding calls (default: 10)",
    )
    parser.add_argument(
        "--k-per-pair",
        type=int,
        default=15,
        help="Number of criteria to extract per preference pair (default: 15).",
    )
    parser.add_argument(
        "--focus-theme",
        type=str,
        default="",
        help="Optional thematic steer for criteria extraction (default: blank = "
        "no steer). When set (e.g. 'instruction following'), PRIORITIZE — not "
        "mandate — criteria related to the theme when genuinely present; "
        "anti-fabrication guard still applies.",
    )
    parser.add_argument(
        "--use-reasoning-trace",
        action="store_true",
        help="Two-stage Stage 1: generate a reasoning trace per battle "
        "explaining the winner, then condition criteria extraction on it.",
    )
    parser.add_argument(
        "--revise",
        choices=["none", "sample", "trait", "both"],
        default="none",
        help="Add a single revision round. 'sample': revise per-battle "
        "criteria so they predict the true winner; 'trait': revise the "
        "final consolidated rubric on its wrong-prediction cases; "
        "'both': run both. Pre/post win-rate is logged. "
        "Requires batch inference — silently skipped under --no-batch.",
    )
    parser.add_argument(
        "--final-topk-selection",
        choices=["source_count", "logdet", "llm", "mmr"],
        default="llm",
        help="Strategy used to pick the final top-K active rubric from "
        "the high-impact pool. 'llm' (default): configured reasoning-model "
        "selection with paraphrastic merge. 'mmr': genericity filter + "
        "greedy MMR over log-scaled source_count. "
        "'logdet': greedy log-determinant volume maximization over trait "
        "embeddings. 'source_count': top-K by source_count slice (legacy).",
    )
    parser.add_argument(
        "--selection-pool-size",
        type=int,
        default=None,
        help="Cap input pool to top-N high-impact traits by source_count "
        "before running the selection strategy "
        "(default: max(100, 2 * --target-traits)).",
    )
    parser.add_argument(
        "--logdet-eps",
        type=float,
        default=0.5,
        help="Log-determinant regularizer epsilon (default: 0.5).",
    )
    parser.add_argument(
        "--logdet-min-increment",
        type=float,
        default=0.001,
        help="Log-determinant stopping: min log-volume increment per pick "
        "(default 0.001).",
    )
    parser.add_argument(
        "--logdet-patience",
        type=int,
        default=3,
        help="Log-determinant stopping: patience over consecutive "
        "sub-threshold picks (default 3).",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.75,
        help="MMR relevance/diversity trade-off "
        "(0=pure diversity, 1=pure source_count). Default 0.75.",
    )
    parser.add_argument(
        "--mmr-skip-genericity-filter",
        action="store_true",
        help="Skip the LLM genericity filter that drops prompt-specific traits "
        "(only applies to --final-topk-selection mmr).",
    )
    parser.add_argument(
        "--mmr-disable-abstraction-rescue",
        action="store_true",
        help="Disable the abstraction-rescue pass that tries to salvage "
        "generalizable axes from prompt-specific traits the genericity "
        "filter drops (only applies to --final-topk-selection mmr).",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch inference and use sequential Converse API calls",
    )
    parser.add_argument(
        "--s3-bucket",
        default=DEFAULT_S3_BUCKET,
        help="S3 bucket for batch inference (default: config aws.s3_bucket)",
    )
    parser.add_argument(
        "--role-arn",
        default=DEFAULT_BATCH_ROLE_ARN,
        help="IAM role ARN for Bedrock batch jobs",
    )

    args = parser.parse_args()
    if not args.arena_data:
        parser.error("--input-preference-data is required")

    # Say this before the run rather than only at the two skip sites, so a
    # --no-batch --revise invocation isn't discovered to be a no-op an hour in.
    if args.revise != "none" and args.no_batch:
        print(
            f"WARN: --revise {args.revise} needs batch inference and will be "
            "skipped under --no-batch; drop --no-batch to run the revision round."
        )

    try:
        run_trait_discovery(
            arena_file=args.arena_data,
            output_dir=args.output,
            judge_model=args.judge_model,
            embedding_model=args.embedding_model,
            max_samples=args.max_samples,
            target_traits=args.target_traits,
            cluster_size=args.cluster_size,
            max_levels=args.max_levels,
            min_source_count=args.min_source_count,
            skip_extraction=args.skip_extraction,
            max_embed_workers=args.max_embed_workers,
            region=args.region,
            no_batch=args.no_batch,
            s3_bucket=args.s3_bucket,
            role_arn=args.role_arn,
            k_per_pair=args.k_per_pair,
            use_reasoning_trace=args.use_reasoning_trace,
            focus_theme=args.focus_theme,
            revise=args.revise,
            final_topk_selection=args.final_topk_selection,
            selection_pool_size=args.selection_pool_size,
            logdet_eps=args.logdet_eps,
            logdet_min_increment=args.logdet_min_increment,
            logdet_patience=args.logdet_patience,
            mmr_lambda=args.mmr_lambda,
            mmr_skip_genericity_filter=args.mmr_skip_genericity_filter,
            mmr_disable_abstraction_rescue=args.mmr_disable_abstraction_rescue,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"premise discover run: error: {exc}\n")
