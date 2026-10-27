# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Trait and sample loaders for the experiment scripts."""

import json


def load_traits(traits_file: str) -> list:
    """Load rubric definitions from a traits JSON file.

    Supports multiple schemas: {"rubrics": [...]}, {"traits": [...]},
    {"rules": [...]}, or a bare list.

    Normalizes each item to have at least "id" and "description" so
    downstream code can access trait["id"] / trait["description"]
    uniformly.
    """
    with open(traits_file) as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    else:
        items = data.get("rubrics") or data.get("traits") or data.get("rules") or []
    out = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            out.append({"id": f"rule_{i}", "description": item})
        elif isinstance(item, dict):
            norm = dict(item)
            if "id" not in norm:
                norm["id"] = norm.get("name") or norm.get("category") or f"rule_{i}"
            if "description" not in norm:
                norm["description"] = norm.get("definition") or norm.get("name") or ""
            out.append(norm)
        else:
            out.append({"id": f"rule_{i}", "description": str(item)})
    return out


def _extract_conversation_text(conversation: list) -> tuple[str, str]:
    """Extract (prompt, response) from a conversation message list."""
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


def load_samples(samples_file: str, max_samples=None) -> list:
    """Load pairwise comparison samples from a JSONL file.

    Handles three formats:
      - Battle:  {conversation_a, conversation_b}     -> normalized to prompt/response_a/response_b
      - Direct:  {prompt, response_a, response_b}     -> used as-is
      - Single:  {prompt, response}                   -> response_a populated, response_b empty
    """
    samples = []
    with open(samples_file) as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            if "conversation_a" in raw or "conversation_b" in raw:
                conv_a = raw.get("conversation_a", [])
                conv_b = raw.get("conversation_b", [])
                prompt_a, resp_a = _extract_conversation_text(conv_a)
                prompt_b, resp_b = _extract_conversation_text(conv_b)
                sample = {
                    "id": raw.get("id", ""),
                    "prompt": prompt_a or prompt_b,
                    "response_a": resp_a,
                    "response_b": resp_b,
                    "winner": raw.get("winner", ""),
                }
            else:
                sample = raw
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                break
    return samples
