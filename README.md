# PReMISE

PReMISE discovers evaluation rubrics from pairwise preferences, audits rubric
quality, and repairs measured weaknesses. It is the reference implementation
for **“PReMISE: Policy Rubrics as Measurement Specifications for LLM
Judges.”**

This code is being released solely for academic and scientific reproducibility
purposes, in support of the methods and findings described in the associated
publication. Pull requests are not being accepted in order to maintain the code
exactly as it was used in the paper.

## Install

Python 3.11 or newer is required.

From a source checkout:

```bash
uv sync
uv run premise --help
```

## Configure a model provider

Non-secret defaults live in [`config.json`](config.json). Credentials remain in
the provider's standard environment or AWS credential chain.

| Provider | Credential |
| --- | --- |
| Amazon Bedrock | Standard AWS credentials |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

Check the active provider and model without printing secrets:

```bash
uv run premise --config config.json providers
```

Override the provider and model for one command:

```bash
uv run premise --config config.json \
  --provider openrouter \
  --model deepseek/deepseek-v3.1 \
  providers
```

The same `--provider` and `--model` options work with discovery, audit, and
repair commands. Non-Bedrock providers run synchronously.

The experiments reported in the paper did not use the OpenAI, Anthropic,
Gemini, or OpenRouter APIs. These integrations are provided only as convenient
portability options for readers who want to reproduce or use the pipeline with
their own model providers.

OpenRouter defaults to `reasoning_effort: "minimal"` in `config.json` so
reasoning models leave completion tokens for the structured answer. Increase
the effort only when the selected model and token budget require it.

Discovery and audit output directories are resumable. PReMISE fingerprints the
provider, model roles, effective configuration, input contents, and workflow
options before reusing cached stages. Reuse an output directory only for an
exact rerun; choose a new directory after changing any of those inputs.

## Preference and rubric formats

Preferences are JSONL, one comparison per line:

```json
{"id":"example-1","prompt":"Explain why the sky is blue.","response_a":"...","response_b":"...","winner":"model_b"}
```

`winner` must be `model_a` or `model_b`. Conversation records using
`conversation_a` and `conversation_b` are also accepted.

Rubrics use this JSON shape:

```json
{
  "schema_version": "1.0",
  "rubrics": [
    {
      "id": "direct_answer",
      "description": "The response should answer the question before elaborating."
    }
  ]
}
```

The examples are illustrative inputs, not benchmarks.

## Discover a rubric

Discovery requires only preference data:

```bash
uv run premise --config config.json discover run \
  --input-preference-data examples/instruction-following/preferences.jsonl \
  --output results/discovery \
  --target-traits 5 \
  --cluster-size 5 \
  --max-levels 3 \
  --min-source-count 1 \
  --no-batch
```

The selected rubric is written to `results/discovery/traits_top_5.json`.

## Audit an existing rubric

The structural-budget audit is deterministic and makes no model calls:

```bash
uv run premise --config config.json audit structural \
  examples/instruction-following/discovered-rubric.json
```

The quick audit measures structural budget, preference fit, and consistency:

```bash
uv run premise --config config.json audit run \
  --policy-rubrics-file examples/instruction-following/discovered-rubric.json \
  --input-preference-data examples/instruction-following/preferences.jsonl \
  --output results/audit \
  --max-samples 20 \
  --no-batch
```

Add `--include-adversarial` to measure Verified Fool Rate.

The paper protocol uses the Bedrock model roles in `config.json` and runs the
three-model structural panel, all reliability checks, nested-cross-validation
Bradley–Terry preference fit, and 500 adversarial attempts:

```bash
uv run premise --config config.json audit run \
  --paper-protocol \
  --policy-rubrics-file examples/instruction-following/discovered-rubric.json \
  --input-preference-data examples/instruction-following/preferences.jsonl \
  --output results/paper-audit
```

The paper protocol is intentionally expensive.

## Repair an existing rubric

Adversarial repair rewrites traits using explicit operators:

```bash
uv run premise --config config.json refine adversarial run \
  --policy-rubrics-file examples/instruction-following/discovered-rubric.json \
  --input-preference-data examples/instruction-following/preferences.jsonl \
  --output results/repair \
  --operator grounded \
  --operator parsimony
```

The repaired rubric is written to `results/repair/final-rubric.json`. Available
operators are `observable`, `scope`, `parsimony`, `grounded`, `coverage`,
`commitment`, and `brevity_neutral`.

Preference repair keeps the traits with the strongest measured preference fit
and does not rewrite rubric text:

```bash
uv run premise --config config.json audit ira \
  --traits examples/instruction-following/discovered-rubric.json \
  --samples examples/instruction-following/preferences.jsonl \
  --output results/ira \
  --no-batch

uv run premise --config config.json audit pref-fit-analysis \
  --ira-dir results/ira \
  --battles examples/instruction-following/preferences.jsonl \
  --output results/preference-metrics.json

uv run premise --config config.json refine pref-repair \
  --traits examples/instruction-following/discovered-rubric.json \
  --metrics results/preference-metrics.json \
  --top-k 5 \
  --output results/preference-repaired-rubric.json
```

## Reproduce the human verifier annotation

The paper's adversarial audit uses a verifier model to distinguish genuine
rubric satisfaction from gaming. This exercise measures that verifier against
blinded human judgments.

Install the optional UI:

```bash
uv sync --extra annotate
```

Create `annotation-corpora.json` describing completed adversarial runs:

```json
[
  {
    "label": "discovered",
    "responses": "results/adversarial/responses",
    "traits": "examples/instruction-following/discovered-rubric.json"
  }
]
```

Generate stratified annotation assignments:

```bash
uv run premise audit annotate \
  --generate \
  --config annotation-corpora.json
```

Start the browser interface:

```bash
uv run streamlit run "$(uv run python -c \
  'import premise.rubric_evaluation.annotate_verifier as m; print(m.__file__)')"
```

After annotation, compute verifier precision and recall:

```bash
uv run premise audit annotate --analyze
```

Annotation files are written to `runtime.annotation_output_dir` from
`config.json`.

## Python API

```python
from premise import (
    audit_preference_fit,
    audit_structural,
    discover_rubric,
    load_paper_rubrics,
)

rubric_path = discover_rubric(
    "preferences.jsonl",
    "results/discovery",
    target_traits=10,
    config="config.json",
)

structure = audit_structural(rubric_path)
fit = audit_preference_fit(
    rubric_path,
    "held-out.jsonl",
    "results/preference-fit",
    config="config.json",
)

paper_rubrics = load_paper_rubrics(
    dataset="UltraFeedback",
    method="PReMISE",
    variant="v0_discovered",
)
```

## Paper artifacts

[`examples/policy_rubrics.json`](examples/policy_rubrics.json) contains the 32
rubric sets released with the paper across four datasets. It is also packaged
inside the wheel and available through `load_paper_rubrics()`.

The default Bedrock configuration records the model roles used by the paper.
Alternative providers improve accessibility but are portability paths, not
exact reproductions of the reported experiments.

## Verify the release artifact

```bash
uv sync
uvx ruff check premise
uvx ruff format --check premise
uv build --clear
uvx twine check dist/*
uv run premise audit structural \
  examples/instruction-following/discovered-rubric.json
```

## Citation

If you use PReMISE, please cite:

```bibtex
@inproceedings{roy-etal-2026-premise,
    title = "{PR}e{MISE}: Policy Rubrics as Measurement Specifications for {LLM} Judges",
    author = "Roy, Swastik  and
      Pujari, Rajkumar  and
      Kumarage, Tharindu  and
      Peris, Charith  and
      Gupta, Rahul  and
      Rumshisky, Anna  and
      Natarajan, Pradeep  and
      Saligrama, Venkatesh",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2653/",
    pages = "60382--60406",
    ISBN = "979-8-89176-459-0"
}
```

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under CC BY-NC 4.0. See [`LICENSE`](LICENSE).
