# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Small, stable Python API over the research implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import Any

from premise.config import RuntimeConfig, config_context

ConfigInput = str | Path | Mapping[str, Any] | RuntimeConfig | None


def load_paper_rubrics(
    *,
    dataset: str | None = None,
    method: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Load the 32 machine-readable rubric sets released with the paper.

    Optional filters use exact, case-insensitive matches and preserve the
    release metadata around the selected ``rubric_sets``.
    """
    resource = files("premise").joinpath("data", "policy_rubrics.json")
    if resource.is_file():
        text = resource.read_text(encoding="utf-8")
    else:
        # Editable source checkout: hatch's force-include is applied only when
        # building the wheel.
        text = (
            Path(__file__).parents[1] / "examples" / "policy_rubrics.json"
        ).read_text()
    payload = json.loads(text)
    filters = {
        "dataset": dataset,
        "method": method,
        "variant": variant,
    }
    selected = payload["rubric_sets"]
    for field, requested in filters.items():
        if requested is not None:
            wanted = requested.casefold()
            selected = [
                rubric_set
                for rubric_set in selected
                if str(rubric_set.get(field, "")).casefold() == wanted
            ]
    result = dict(payload)
    result["rubric_sets"] = selected
    return result


@contextmanager
def _provider_configuration(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    config: ConfigInput = None,
) -> Iterator[RuntimeConfig]:
    path = config if isinstance(config, (str, Path)) else None
    config_object = None if path is not None else config
    with config_context(
        path,
        config=config_object,
        provider=provider,
        model=model,
        base_url=base_url,
    ) as active:
        yield active


def discover_rubric(
    preferences: str | Path,
    output_dir: str | Path,
    *,
    target_traits: int = 20,
    max_samples: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    no_batch: bool = True,
    config: ConfigInput = None,
) -> Path:
    """Discover a rubric and return the selected rubric file."""
    from premise.rubric_discovery.run import run_trait_discovery

    output = Path(output_dir)
    with _provider_configuration(provider, model, config=config):
        run_trait_discovery(
            arena_file=str(preferences),
            output_dir=str(output),
            target_traits=target_traits,
            max_samples=max_samples,
            no_batch=no_batch,
        )
    return output / f"traits_top_{target_traits}.json"


def audit_structural(rubric: str | Path) -> dict:
    """Run the local structural-budget audit for every rubric item."""
    from premise.rubric_evaluation.structural_budget import validate

    data = json.loads(Path(rubric).read_text())
    rubrics = data if isinstance(data, list) else data.get("rubrics", [data])
    reports = {}
    for index, item in enumerate(rubrics):
        rubric_id = item.get("id", f"rubric_{index}")
        reports[rubric_id] = asdict(validate(item.get("description", "")))
    return {
        "passes": all(report["passes"] for report in reports.values()),
        "rubrics": reports,
    }


def audit_preference_fit(
    rubric: str | Path,
    preferences: str | Path,
    output_dir: str | Path,
    *,
    max_samples: int = 500,
    provider: str | None = None,
    model: str | None = None,
    no_batch: bool = True,
    config: ConfigInput = None,
) -> dict:
    """Measure agreement with held-out pairwise preferences."""
    from premise.rubric_evaluation.pref_fit import run_pref_fit_experiment

    with _provider_configuration(provider, model, config=config):
        return run_pref_fit_experiment(
            traits_file=str(rubric),
            samples_file=str(preferences),
            output_dir=str(output_dir),
            max_samples=max_samples,
            judge_model=model,
            no_batch=no_batch,
        )


def audit_consistency(
    rubric: str | Path,
    preferences: str | Path,
    output_dir: str | Path,
    *,
    n_runs: int = 5,
    max_samples: int = 200,
    provider: str | None = None,
    model: str | None = None,
    no_batch: bool = True,
    config: ConfigInput = None,
) -> dict:
    """Measure repeated-judgment consistency."""
    from premise.rubric_evaluation.consistency import run_consistency_experiment

    with _provider_configuration(provider, model, config=config):
        result = run_consistency_experiment(
            traits_file=str(rubric),
            samples_file=str(preferences),
            output_dir=str(output_dir),
            n_runs=n_runs,
            max_samples=max_samples,
            judge_model=model,
            no_batch=no_batch,
        )
    return result or {}


def audit_adversarial(
    rubric: str | Path,
    preferences: str | Path,
    output_dir: str | Path,
    *,
    n_attempts: int = 50,
    provider: str | None = None,
    judge_model: str | None = None,
    attacker_model: str | None = None,
    verifier_model: str | None = None,
    config: ConfigInput = None,
) -> dict:
    """Measure Verified Fool Rate with explicit attacker/judge/verifier roles."""
    from premise.rubric_evaluation.adversarial import run_adversarial_experiment

    with _provider_configuration(provider, judge_model, config=config):
        result = run_adversarial_experiment(
            traits_file=str(rubric),
            samples_file=str(preferences),
            output_dir=str(output_dir),
            n_attempts=n_attempts,
            attacker_model=attacker_model,
            judge_model=judge_model,
            verifier_model=verifier_model,
        )
    return result or {}


def repair_by_preference(
    traits: str | Path,
    metrics: str | Path,
    output: str | Path,
    *,
    top_k: int = 20,
    min_accuracy: float | None = None,
) -> dict:
    """Keep the top-K traits ranked by measured preference fit."""
    from premise.rubric_refinement.preference.repair import run

    return run(
        traits_file=str(traits),
        metrics_file=str(metrics),
        output=str(output),
        top_k=top_k,
        min_accuracy=min_accuracy,
    )
