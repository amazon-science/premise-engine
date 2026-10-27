# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Single click-based entry point: `premise <subcommand>`.

Subcommands expose both guided workflows and individual research stages. Each
technique module under premise/{rubric_discovery,rubric_evaluation,
rubric_refinement} exposes plain Python `run(...)` functions; this file wires
them as click commands so the import surface stays small and discoverable.

The command groups keep the short verbs — `discover`, `audit`, `refine` —
which read better on a command line than the package names do in an import.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import click

from premise.providers import SUPPORTED_PROVIDERS

try:
    _VERSION = version("premise-engine")
except PackageNotFoundError:
    from premise import __version__ as _VERSION

# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=_VERSION, prog_name="premise")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    envvar="PREMISE_CONFIG",
    help="Configuration JSON. Precedence: explicit path, PREMISE_CONFIG, "
    "./config.json, packaged defaults.",
)
@click.option(
    "--output-root",
    "output_root",
    type=click.Path(file_okay=False, path_type=str),
    help="Root directory for command outputs. Each subcommand that takes "
    "--output defaults to <root>/<group>/<cmd>/ when not explicitly set. "
    "Has no effect on commands that use --root (refine adversarial stages) or "
    "fixed output paths.",
)
@click.option(
    "--provider",
    type=click.Choice(SUPPORTED_PROVIDERS, case_sensitive=False),
    help="Explicitly override the provider selected by the configuration.",
)
@click.option(
    "--model",
    help="Explicitly override every model role for this invocation.",
)
@click.option(
    "--base-url",
    help="Explicitly override the selected provider's API base URL.",
)
@click.pass_context
def main(
    ctx: click.Context,
    config_path: str | None,
    output_root: str | None,
    provider: str | None,
    model: str | None,
    base_url: str | None,
) -> None:
    """Discover, audit, and repair LLM evaluation rubrics.

    Non-secret settings load from --config, PREMISE_CONFIG, ./config.json, or
    the packaged default. Provider keys and AWS credentials stay in their
    standard credential sources.
    """
    from premise.config import ConfigError, configure

    try:
        config = configure(
            config_path,
            provider=provider,
            model=model,
            base_url=base_url,
            output_root=output_root,
        )
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from None
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["config_path"] = config.source
    ctx.obj["output_root"] = config.runtime("output_root")


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------


@main.group()
def discover() -> None:
    """Trait discovery from preference data."""


def _forward(argv: tuple[str, ...], module_name: str, prog: str) -> None:
    """Forward unknown args to a module's __main__ block.

    Uses runpy so modules whose argparse driver lives under
    `if __name__ == "__main__":` (rather than in a `main()` function) work too.
    """
    import runpy
    import sys

    sequential_modules = {
        "premise.rubric_discovery.run",
        "premise.rubric_discovery.rubric_scan",
        "premise.rubric_evaluation.workflow",
        "premise.rubric_evaluation.adversarial",
        "premise.rubric_evaluation.applicability_eval",
        "premise.rubric_evaluation.consistency",
        "premise.rubric_evaluation.ira",
        "premise.rubric_evaluation.paraphrase",
        "premise.rubric_evaluation.pref_fit",
        "premise.rubric_evaluation.trait_scoring",
    }
    from premise.providers import provider_name

    active_provider = provider_name()
    if (
        active_provider != "bedrock"
        and module_name in sequential_modules
        and "--no-batch" not in argv
        and "--help" not in argv
        and "-h" not in argv
    ):
        argv = ("--no-batch", *argv)

    saved = sys.argv
    try:
        sys.argv = [prog, *argv]
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = saved


@main.command("providers")
def providers_cmd() -> None:
    """Show provider credentials and the active model without printing secrets."""
    from premise.config import get_config
    from premise.providers import load_provider_config, provider_status

    config = get_config()
    click.echo(f"Config source: {config.source}")
    click.echo(f"Active provider: {config.provider}")
    model = load_provider_config(require_key=False).model or "(command default)"
    click.echo(f"Active model: {model}")
    for row in provider_status():
        marker = "*" if row["active"] else " "
        state = "configured" if row["configured"] else "missing credentials"
        click.echo(f"{marker} {row['provider']:<10} {state:<19} {row['credential']}")


def _inject_output(
    argv: tuple[str, ...], group: str, cmd: str, flag: str = "--output"
) -> tuple[str, ...]:
    """If --output-root is set on the click context AND the user did not pass
    --output / --outdir, inject `--output <root>/<group>/<cmd>/` into argv.

    Returns argv unchanged when there's nothing to inject.
    """
    import os

    ctx = click.get_current_context(silent=True)
    root = ctx and ctx.find_root().obj and ctx.find_root().obj.get("output_root")
    if not root:
        return argv
    # Don't override an explicit user choice.
    if any(a == flag or a.startswith(flag + "=") for a in argv):
        return argv
    # Show --help unmodified.
    if any(a in ("-h", "--help") for a in argv):
        return argv
    out = os.path.join(root, group, cmd)
    os.makedirs(out, exist_ok=True)
    return (flag, out, *argv)


@discover.command(
    "run",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def discover_run(argv: tuple[str, ...]) -> None:
    """Full discovery pipeline. (--help for flags)"""
    argv = _inject_output(argv, "discover", "run")
    _forward(argv, "premise.rubric_discovery.run", "premise discover run")


@discover.command(
    "scan",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def discover_scan(argv: tuple[str, ...]) -> None:
    """Score an authored rubric corpus across battles. (--help for flags)"""
    argv = _inject_output(argv, "discover", "scan")
    _forward(argv, "premise.rubric_discovery.rubric_scan", "premise discover scan")


@discover.command(
    "ablation",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def discover_ablation(argv: tuple[str, ...]) -> None:
    """Single-pass Ward-linkage consolidation ablation. (--help for flags)"""
    argv = _inject_output(argv, "discover", "ablation")
    _forward(argv, "premise.rubric_discovery.ablation", "premise discover ablation")


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------


@main.group()
def audit() -> None:
    """Audit a rubric: structure, reliability, preference fit, and robustness."""


@main.group(hidden=True)
def quality() -> None:
    """Deprecated alias for `audit`."""


def _passthrough(
    group,
    name: str,
    module: str,
    doc: str,
    inject_output: bool = True,
    hidden: bool = False,
) -> None:
    """Register a click subcommand that forwards argv to module.__main__.

    inject_output=True (default) prepends `--output <output-root>/<group>/<cmd>/`
    when the user didn't specify --output and --output-root is set on the
    root command. Set False for subcommands that don't take --output
    (e.g. SB-PLR validator, summary commands).
    """
    short_help = doc + " (--help for flags)"

    @group.command(
        name,
        help=short_help,
        hidden=hidden,
        context_settings={
            "ignore_unknown_options": True,
            "allow_extra_args": True,
            "help_option_names": [],
        },
    )
    @click.argument("argv", nargs=-1, type=click.UNPROCESSED)
    def _cmd(
        argv: tuple[str, ...],
        _module=module,
        _name=name,
        _gname=group.name,
        _inject=inject_output,
    ) -> None:
        if _inject:
            argv = _inject_output(argv, _gname, _name)
        _forward(argv, _module, f"premise {_gname} {_name}")


# (name, module, doc, inject_output) — inject_output False for commands that
# either don't take --output, or take it as a file path (not a directory).
_AUDIT_COMMANDS = [
    (
        "run",
        "premise.rubric_evaluation.workflow",
        "Resumable quick audit or full paper protocol.",
        True,
    ),
    (
        "meta-trait",
        "premise.rubric_evaluation.meta_trait",
        "Structural / meta-trait analysis.",
        True,
    ),
    (
        "ira",
        "premise.rubric_evaluation.ira",
        "Inter-rater agreement (Krippendorff alpha).",
        True,
    ),
    (
        "consistency",
        "premise.rubric_evaluation.consistency",
        "Intra-rater consistency / flip rate.",
        True,
    ),
    (
        "paraphrase",
        "premise.rubric_evaluation.paraphrase",
        "Paraphrase stability.",
        True,
    ),
    (
        "adversarial",
        "premise.rubric_evaluation.adversarial",
        "Adversarial gameability / VFR.",
        True,
    ),
    (
        "pref-fit",
        "premise.rubric_evaluation.pref_fit",
        "Rubric-conditioned preference accuracy.",
        True,
    ),
    (
        "pref-fit-analysis",
        "premise.rubric_evaluation.pref_fit_analysis",
        "Extended pref-fit: per-trait accuracy, effective dimensionality, PCA.",
        True,
    ),
    (
        "bt-score",
        "premise.rubric_evaluation.bt_scoring",
        "Bradley-Terry per-trait scoring.",
        True,
    ),
    (
        "bt-regress",
        "premise.rubric_evaluation.bt_regression",
        "Bradley-Terry composite regression.",
        True,
    ),
    (
        "trait-score",
        "premise.rubric_evaluation.trait_scoring",
        "Per-trait benchmark scoring.",
        True,
    ),
    (
        "target-correlation",
        "premise.rubric_evaluation.target_correlation",
        "Correlation vs target metric.",
        True,
    ),
    (
        "structural",
        "premise.rubric_evaluation.structural_budget",
        "Validate rubric structure and complexity.",
        False,
    ),
    (
        "applicability",
        "premise.rubric_evaluation.applicability_eval",
        "In-domain vs out-of-domain trait applicability (generality gap).",
        True,
    ),
    (
        "annotate",
        "premise.rubric_evaluation.annotate_verifier",
        "Verifier annotation tool.",
        False,
    ),
]

for _audit_group in (audit, quality):
    for _name, _module, _doc, _inject in _AUDIT_COMMANDS:
        _passthrough(_audit_group, _name, _module, _doc, inject_output=_inject)
    _passthrough(
        _audit_group,
        "budget",
        "premise.rubric_evaluation.structural_budget",
        "Deprecated alias for `audit structural`.",
        inject_output=False,
        hidden=True,
    )


# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------


@main.group()
def refine() -> None:
    """Rubric repair: preference-fit top-K, or adversarial operators + Pareto."""


@refine.command(
    "pref-repair",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def refine_pref_repair(argv: tuple[str, ...]) -> None:
    """Preference repair: rank traits by preference fit, keep top K. (--help for flags)"""
    import sys

    from premise.rubric_refinement.preference import repair as pref_mod

    saved = sys.argv
    try:
        sys.argv = ["premise refine pref-repair", *argv]
        pref_mod.main()
    finally:
        sys.argv = saved


# Prior name for `pref-repair`, kept so documented invocations keep working.
@refine.command(
    "preference",
    hidden=True,
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def refine_preference(argv: tuple[str, ...]) -> None:
    """Deprecated alias for `refine pref-repair`."""
    import sys

    from premise.rubric_refinement.preference import repair as pref_mod

    saved = sys.argv
    try:
        sys.argv = ["premise refine pref-repair", *argv]
        pref_mod.main()
    finally:
        sys.argv = saved


@refine.command("atomicity")
@click.argument("traits_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output path for the rewritten rubric JSON.",
)
@click.option("--region", default=None)
def refine_atomicity(traits_file: str, output: str, region: str | None) -> None:
    """Rewrite bundled traits as atomic should-statements.

    Operates on a traits JSON file (schema: {rubrics: [{id, description, ...}]}).
    Each trait that looks bundled (contains 'and'/'with'/'while'/'including'
    or '_and_' in name) gets rewritten by Qwen3-235B into a single-axis
    should-statement. Atomic-looking traits are left untouched.
    """
    import json
    from pathlib import Path

    from .rubric_discovery.run import Trait
    from .rubric_discovery.selection import atomic_rewrite_traits
    from .utils.bedrock import DEFAULT_REGION

    src = json.loads(Path(traits_file).read_text())
    traits = [
        Trait(
            trait_id=r["id"],
            description=r["description"],
            source_count=r.get("source_count", 0),
            source_ids=[],
        )
        for r in src.get("rubrics", [])
    ]
    rewritten, annotations = atomic_rewrite_traits(
        traits, region=region or DEFAULT_REGION
    )

    out = dict(src)
    new_rubrics = []
    for t in rewritten:
        r = {
            "id": t.trait_id,
            "description": t.description,
            "category": "discovered",
            "source_count": t.source_count,
        }
        ann = annotations.get(t.trait_id, {})
        if ann.get("atomic_rewrite_applied"):
            r["atomic_rewrite_applied"] = True
            r["atomic_rewrite_old_id"] = ann.get("atomic_rewrite_old_id")
            r["atomic_rewrite_old_description"] = ann.get(
                "atomic_rewrite_old_description"
            )
        new_rubrics.append(r)
    out["rubrics"] = new_rubrics
    out["n_rubrics"] = len(new_rubrics)

    Path(output).write_text(json.dumps(out, indent=2))
    n_changed = sum(1 for a in annotations.values() if a.get("atomic_rewrite_applied"))
    click.echo(f"Wrote {output}: {n_changed}/{len(traits)} traits rewritten.")


@refine.group("adversarial")
def refine_adversarial() -> None:
    """Adversarial repair: rewrite traits under the operators, then Pareto-select.

    Use `run` for the guided resumable path. Advanced stages remain available
    individually: seed -> predict -> merge -> prune -> eval -> pareto.
    """


@refine_adversarial.command(
    "run",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def refine_adversarial_run(argv: tuple[str, ...]) -> None:
    """Guided resumable adversarial repair. (--help for flags)"""
    _forward(
        argv,
        "premise.rubric_refinement.adversarial.workflow",
        "premise refine adversarial run",
    )


# (name, module, stage function, help) — one row per stage, in run order.
# `predict` and `merge` are the repair itself; the rest filter and measure.
_STAGES = [
    (
        "seed",
        "premise.rubric_refinement.adversarial.seed",
        "main",
        "Stage 0: scaffold the corpus root from a v0 traits file.",
    ),
    (
        "predict",
        "premise.rubric_refinement.adversarial.repair",
        "predict_prep_main",
        "Stage 1: apply repair operators to each trait and predict the result.",
    ),
    (
        "merge",
        "premise.rubric_refinement.adversarial.repair",
        "merge_bundles_main",
        "Stage 2: harvest the repaired rubric text into tree.json.",
    ),
    (
        "prune",
        "premise.rubric_refinement.adversarial.filters",
        "prune_topn_main",
        "Stage 3: hard filters + top-N per trait.",
    ),
    (
        "eval",
        "premise.rubric_refinement.adversarial.filters",
        "batch_eval_main",
        "Stage 4: held-out IRA, consistency, and adversarial evaluation.",
    ),
    (
        "pareto",
        "premise.rubric_refinement.adversarial.pareto",
        "main",
        "Stage 5: select Pareto winners, write a pareto.json rubric file.",
    ),
    (
        "pareto-eval",
        "premise.rubric_refinement.adversarial.pareto",
        "pareto_eval_main",
        "Stage 5: iter-2 paper-settings eval on Pareto winners.",
    ),
    (
        "plot",
        "premise.rubric_refinement.adversarial.plots",
        "plot_main",
        "Pareto frontier plots (alpha/flip vs VFR).",
    ),
    (
        "resume",
        "premise.rubric_refinement.adversarial.filters",
        "resume_main",
        "Resume Stage 4's consistency aggregation after a subprocess crash.",
    ),
]


def _stage_command(group, name: str, module: str, fn: str, doc: str, prog: str) -> None:
    """Register a stage subcommand calling <module>.<fn>()."""

    @group.command(
        name,
        help=doc + " (--help for flags)",
        context_settings={
            "ignore_unknown_options": True,
            "allow_extra_args": True,
            "help_option_names": [],
        },
    )
    @click.argument("argv", nargs=-1, type=click.UNPROCESSED)
    def _cmd(argv: tuple[str, ...], _fn=fn, _mod=module, _prog=prog) -> None:
        import importlib
        import sys

        stage_mod = importlib.import_module(_mod)

        saved = sys.argv
        try:
            sys.argv = [_prog, *argv]
            getattr(stage_mod, _fn)()
        finally:
            sys.argv = saved


for _name, _mod, _fn, _doc in _STAGES:
    _stage_command(
        refine_adversarial,
        _name,
        _mod,
        _fn,
        _doc,
        f"premise refine adversarial {_name}",
    )


# Prior flat names, kept hidden so existing invocations keep working. The stages
# are documented under `refine adversarial` now; these forward to the same
# functions rather than duplicating them.
def _flat_alias(name: str, module: str, fn: str, prog: str) -> None:
    """Register a hidden top-level `refine <name>` forwarding to a stage."""

    @refine.command(
        name,
        hidden=True,
        help=f"Deprecated alias for `refine adversarial {name}`.",
        context_settings={
            "ignore_unknown_options": True,
            "allow_extra_args": True,
            "help_option_names": [],
        },
    )
    @click.argument("argv", nargs=-1, type=click.UNPROCESSED)
    def _cmd(argv: tuple[str, ...], _fn=fn, _mod=module, _prog=prog) -> None:
        import importlib
        import sys

        stage_mod = importlib.import_module(_mod)

        saved = sys.argv
        try:
            sys.argv = [_prog, *argv]
            getattr(stage_mod, _fn)()
        finally:
            sys.argv = saved


for _name, _mod, _fn, _doc in _STAGES:
    _flat_alias(_name, _mod, _fn, f"premise refine adversarial {_name}")


if __name__ == "__main__":
    main()
