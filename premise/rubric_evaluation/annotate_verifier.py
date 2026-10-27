# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Human annotation of adversarial verifier verdicts, to measure its precision/recall.

`premise audit adversarial` records, for each attack attempt, whether the
verifier judged the attacker's response a genuine rubric satisfaction or a gaming
attempt. That verdict is itself a model output, so it needs validating against
people. This tool samples those records, hands them to annotators blind, and
scores the verifier against what they say.

Three steps:

    # 1. sample the attack records into per-annotator files
    premise audit annotate --generate --config corpora.json

    # 2. annotate, one browser per annotator
    streamlit run $(python -c "import premise.rubric_evaluation.annotate_verifier as m; print(m.__file__)")

    # 3. score the verifier against the collected labels
    premise audit annotate --analyze

`--config` is a JSON list naming which adversarial runs to sample from — one
entry per rubric corpus you want represented, since the sampler stratifies
across them:

    [
      {
        "label": "discovered",
        "responses": "results/quality/adversarial/responses",
        "traits": "results/discover/run/traits_top_20.json"
      },
      {"label": "baseline", "responses": "...", "traits": "..."}
    ]

`responses` is an adversarial run's `responses/` directory, holding the
`<trait_id>_responses.jsonl` files that `premise audit adversarial` wrote.
`traits` is the rubric JSON those traits came from, read for their descriptions
(annotators need the rubric text to judge against). Paths are relative to where
you run the command. `label` is free-form and only ever used to group and report.

Annotation files and the answer key go to `--output-dir`.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from premise.config import get_config

# Where annotation files and the answer key live. Overridable with --output-dir;
# the default comes from the active config.
DEFAULT_OUTPUT_DIR = Path(get_config().runtime("annotation_output_dir"))


def load_corpus_config(config_file: str) -> list[tuple[str, str, str]]:
    """Read the corpora to sample from into (label, response_dir, trait_source).

    Kept as a tuple list because that is what `_load_all_records` iterates, but
    the file itself is a list of objects so a corpus can gain a field later
    without invalidating configs already written.
    """
    entries = json.loads(Path(config_file).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"{config_file}: expected a non-empty JSON list of corpus objects"
        )
    out = []
    for i, e in enumerate(entries):
        missing = [k for k in ("label", "responses", "traits") if not e.get(k)]
        if missing:
            raise ValueError(
                f"{config_file}: entry {i} is missing {', '.join(missing)} "
                "(each needs label, responses, traits)"
            )
        out.append((e["label"], e["responses"], e["traits"]))
    labels = [o[0] for o in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{config_file}: duplicate labels — {labels}")
    return out


EXPLOIT_TYPES = [
    "Format Over Content",
    "Irrelevant Elaboration",
    "False Authority",
    "Complexity Obfuscation",
    "Partial Truth",
    "Hedging and Vagueness",
    "Confident Incorrectness",
    "Verbosity Bias",
    "Other",
]


def _load_trait_descriptions(trait_source: str) -> dict[str, str]:
    """Load trait_id -> description mapping from a rubric JSON file."""
    with open(trait_source) as f:
        data = json.load(f)
    rubrics = data.get("rubrics", data) if isinstance(data, dict) else data
    return {r["id"]: r["description"] for r in rubrics}


def _is_english(text: str, threshold: float = 0.3) -> bool:
    """Return True if text is likely English (low fraction of non-ASCII chars)."""
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) < threshold


def _load_all_records(corpus_config):
    """Load all adversarial response records with corpus/trait metadata."""
    pool_a = []  # is_fooled=True
    pool_b = []  # is_fooled=False, judge_score >= 7

    for corpus_label, resp_dir, trait_source in corpus_config:
        resp_path = Path(resp_dir)
        if not resp_path.exists():
            print(f"  SKIP {corpus_label}: {resp_dir} not found")
            continue

        trait_descs = _load_trait_descriptions(trait_source)

        for jsonl_file in sorted(resp_path.glob("*_responses.jsonl")):
            trait_id = jsonl_file.stem.replace("_responses", "")
            description = trait_descs.get(
                trait_id, f"[description not found for {trait_id}]"
            )

            with open(jsonl_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if not _is_english(rec["prompt"]):
                        continue
                    pattern = rec.get("pattern_type", "")
                    if pattern in ("Unknown", "N/A", "N", ""):
                        pattern = None
                    enriched = {
                        "corpus": corpus_label,
                        "trait_id": trait_id,
                        "rubric_description": description,
                        "user_prompt": rec["prompt"],
                        "attacker_response": rec["attacker_response"],
                        "reference_response": rec["reference"],
                        "judge_score": rec["judge_score"],
                        "judge_response": rec.get("judge_response", ""),
                        # Hidden fields (for answer key only)
                        "_is_fooled": rec["is_fooled"],
                        "_verifier_response": rec["verifier_response"],
                        "_pattern_type": rec.get("pattern_type", ""),
                    }
                    if rec["is_fooled"]:
                        pool_a.append(enriched)
                    elif rec["judge_score"] >= 7:
                        pool_b.append(enriched)

    return pool_a, pool_b


def generate_samples(
    corpus_config,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    n_annotators: int = 4,
):
    """Generate stratified annotation JSONL files and answer key."""
    print("Loading all adversarial records...")
    pool_a, pool_b = _load_all_records(corpus_config)
    print(f"  Pool A (is_fooled=True): {len(pool_a)}")
    print(f"  Pool B (judge>=7, not fooled): {len(pool_b)}")
    # Missing corpora are skipped with a warning above, so an empty pool here
    # means every configured path was absent — fail rather than write 0 samples
    # and let the operator discover it in the annotation UI.
    if not pool_a and not pool_b:
        raise ValueError(
            "No adversarial records found in any configured corpus. Check the "
            "`responses` paths in --config point at a `premise audit "
            "adversarial` run's responses/ directory."
        )

    rng = random.Random(seed)

    # Stratified sample 50 from each pool, proportional to corpus size
    def stratified_sample(pool, n):
        by_corpus = {}
        for rec in pool:
            by_corpus.setdefault(rec["corpus"], []).append(rec)

        # Proportional allocation
        total = len(pool)
        samples = []
        remaining = n
        corpus_list = sorted(by_corpus.keys())
        for i, corpus in enumerate(corpus_list):
            records = by_corpus[corpus]
            if i == len(corpus_list) - 1:
                k = remaining  # last corpus gets the remainder
            else:
                k = max(1, round(n * len(records) / total))
                k = min(k, remaining, len(records))
            rng.shuffle(records)
            samples.extend(records[:k])
            remaining -= k
            if remaining <= 0:
                break

        rng.shuffle(samples)
        return samples[:n]

    samples_a = stratified_sample(pool_a, 50)
    samples_b = stratified_sample(pool_b, 50)

    # Assign sample IDs and pool labels
    all_samples = []
    answer_key = {}

    for i, rec in enumerate(samples_a):
        sid = f"S{i + 1:03d}"
        answer_key[sid] = {
            "pool": "A_precision",
            "is_fooled": rec["_is_fooled"],
            "judge_score": rec["judge_score"],
            "verifier_response": rec["_verifier_response"],
            "pattern_type": rec["_pattern_type"],
            "corpus": rec["corpus"],
            "trait_id": rec["trait_id"],
        }
        all_samples.append(
            {
                "sample_id": sid,
                "corpus": rec["corpus"],
                "trait_id": rec["trait_id"],
                "rubric_description": rec["rubric_description"],
                "judge_score": rec["judge_score"],
                "judge_response": rec.get("judge_response", ""),
                "user_prompt": rec["user_prompt"],
                "attacker_response": rec["attacker_response"],
                "reference_response": rec["reference_response"],
                "annotation": None,
                "exploit_type": None,
                "confidence": None,
                "notes": None,
            }
        )

    for i, rec in enumerate(samples_b):
        sid = f"S{i + 51:03d}"
        answer_key[sid] = {
            "pool": "B_recall",
            "is_fooled": rec["_is_fooled"],
            "judge_score": rec["judge_score"],
            "verifier_response": rec["_verifier_response"],
            "pattern_type": rec["_pattern_type"],
            "corpus": rec["corpus"],
            "trait_id": rec["trait_id"],
        }
        all_samples.append(
            {
                "sample_id": sid,
                "corpus": rec["corpus"],
                "trait_id": rec["trait_id"],
                "rubric_description": rec["rubric_description"],
                "judge_score": rec["judge_score"],
                "judge_response": rec.get("judge_response", ""),
                "user_prompt": rec["user_prompt"],
                "attacker_response": rec["attacker_response"],
                "reference_response": rec["reference_response"],
                "annotation": None,
                "exploit_type": None,
                "confidence": None,
                "notes": None,
            }
        )

    # Shuffle all samples together (blind pool assignment)
    rng.shuffle(all_samples)

    # Split into annotator files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_annotator = len(all_samples) // n_annotators

    for a in range(n_annotators):
        start = a * per_annotator
        end = start + per_annotator if a < n_annotators - 1 else len(all_samples)
        chunk = all_samples[start:end]

        out_path = output_dir / f"annotator_{a + 1}.jsonl"
        with open(out_path, "w") as f:
            f.writelines(
                json.dumps(sample, ensure_ascii=False) + "\n" for sample in chunk
            )
        print(f"  Wrote {len(chunk)} samples to {out_path.name}")

    # Write answer key
    key_path = output_dir / "answer_key.json"
    with open(key_path, "w") as f:
        json.dump(answer_key, f, indent=2, ensure_ascii=False)
    print(f"  Wrote answer key ({len(answer_key)} entries) to {key_path.name}")

    # Print stratification summary
    print("\nStratification:")
    from collections import Counter

    corpus_counts_a = Counter(
        answer_key[s]["corpus"]
        for s in answer_key
        if answer_key[s]["pool"] == "A_precision"
    )
    corpus_counts_b = Counter(
        answer_key[s]["corpus"]
        for s in answer_key
        if answer_key[s]["pool"] == "B_recall"
    )
    print(f"  Pool A (precision, 50): {dict(corpus_counts_a)}")
    print(f"  Pool B (recall, 50):    {dict(corpus_counts_b)}")


def analyze(output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Compute precision/recall from completed annotations + answer key."""
    import math

    output_dir = Path(output_dir)
    key_path = output_dir / "answer_key.json"
    with open(key_path) as f:
        answer_key = json.load(f)

    # Load all annotations
    annotations = {}
    for jsonl_file in sorted(output_dir.glob("annotator_*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                rec = json.loads(line)
                if rec["annotation"] is not None:
                    annotations[rec["sample_id"]] = rec["annotation"]

    total = len(answer_key)
    annotated = len(annotations)
    print(f"Annotated: {annotated}/{total}")

    if annotated < total:
        print(f"WARNING: {total - annotated} samples not yet annotated")

    # Split by pool
    pool_a_yes = 0
    pool_a_total = 0
    pool_b_yes = 0
    pool_b_total = 0

    for sid, ann in annotations.items():
        key = answer_key.get(sid)
        if not key:
            continue
        is_exploit = ann.lower() in ("yes", "true", "exploiting")
        if key["pool"] == "A_precision":
            pool_a_total += 1
            if is_exploit:
                pool_a_yes += 1
        elif key["pool"] == "B_recall":
            pool_b_total += 1
            if is_exploit:
                pool_b_yes += 1

    print(
        f"\nPool A (precision): {pool_a_yes}/{pool_a_total} agreed with verifier (EXPLOITING)"
    )
    print(f"Pool B (recall):    {pool_b_yes}/{pool_b_total} missed by verifier")

    # Wilson confidence interval
    def wilson_ci(successes, n, z=1.96):
        if n == 0:
            return 0.0, (0.0, 0.0)
        p = successes / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return p, (max(0, center - spread), min(1, center + spread))

    if pool_a_total > 0:
        prec, prec_ci = wilson_ci(pool_a_yes, pool_a_total)
        print(f"\nPrecision: {prec:.1%}  95% CI: [{prec_ci[0]:.1%}, {prec_ci[1]:.1%}]")

    if pool_b_total > 0:
        # Recall: TP / (TP + FN). Use sample counts directly for the CI;
        # do not scale to the full B_recall pool.
        tp = pool_a_yes
        fn_sample = pool_b_yes
        if tp + fn_sample > 0:
            recall_est = tp / (tp + fn_sample)
            # Bootstrap-like CI using Wilson on the FN rate
            _, fn_ci = wilson_ci(pool_b_yes, pool_b_total)
            recall_lo = (
                tp / (tp + fn_ci[1] * pool_b_total)
                if (tp + fn_ci[1] * pool_b_total) > 0
                else 0
            )
            recall_hi = (
                tp / (tp + fn_ci[0] * pool_b_total)
                if (tp + fn_ci[0] * pool_b_total) > 0
                else 1
            )
            print(
                f"Recall (sample-level): {recall_est:.1%}  95% CI: [{recall_lo:.1%}, {recall_hi:.1%}]"
            )
        else:
            print("Recall: N/A (no exploits found in either pool)")

    # Per-corpus breakdown
    from collections import defaultdict

    corpus_stats = defaultdict(
        lambda: {"a_yes": 0, "a_total": 0, "b_yes": 0, "b_total": 0}
    )
    for sid, ann in annotations.items():
        key = answer_key.get(sid)
        if not key:
            continue
        c = key["corpus"]
        is_exploit = ann.lower() in ("yes", "true", "exploiting")
        if key["pool"] == "A_precision":
            corpus_stats[c]["a_total"] += 1
            if is_exploit:
                corpus_stats[c]["a_yes"] += 1
        else:
            corpus_stats[c]["b_total"] += 1
            if is_exploit:
                corpus_stats[c]["b_yes"] += 1

    print("\nPer-corpus breakdown:")
    print(f"{'Corpus':<25s} {'Prec (A)':<15s} {'Missed (B)':<15s}")
    print("-" * 55)
    for c in sorted(corpus_stats):
        s = corpus_stats[c]
        prec_str = f"{s['a_yes']}/{s['a_total']}" if s["a_total"] > 0 else "N/A"
        miss_str = f"{s['b_yes']}/{s['b_total']}" if s["b_total"] > 0 else "N/A"
        print(f"{c:<25s} {prec_str:<15s} {miss_str:<15s}")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def run_streamlit():
    """Annotate one of the files `--generate` wrote, one browser per annotator.

    Streamlit owns argv, so this view uses ``runtime.annotation_output_dir``.
    """
    import streamlit as st

    st.set_page_config(page_title="Verifier Annotation", layout="wide")
    st.title("Adversarial Verifier Annotation")

    outdir = DEFAULT_OUTPUT_DIR

    # File selection
    files = sorted(outdir.glob("annotator_*.jsonl"))
    if not files:
        st.error(
            f"No annotation files found in {outdir}. Run "
            "`premise audit annotate --generate --config <corpora.json>` "
            "first, or update runtime.annotation_output_dir."
        )
        return

    file_names = [f.name for f in files]
    selected = st.sidebar.selectbox("Select your annotation file", file_names)
    file_path = outdir / selected

    # Load samples
    samples = []
    with open(file_path) as f:
        for line in f:
            samples.append(json.loads(line))

    # Progress
    done = sum(1 for s in samples if s["annotation"] is not None)
    st.sidebar.progress(done / len(samples), text=f"{done}/{len(samples)} completed")

    # Navigation
    idx = (
        st.sidebar.number_input(
            "Sample #",
            min_value=1,
            max_value=len(samples),
            value=_find_next_unannotated(samples) + 1,
        )
        - 1
    )

    col_prev, col_next, col_next_empty = st.sidebar.columns(3)
    if col_prev.button("Prev") and idx > 0:
        idx -= 1
    if col_next.button("Next") and idx < len(samples) - 1:
        idx += 1
    if col_next_empty.button("Next unannotated"):
        next_idx = _find_next_unannotated(samples, start=idx)
        if next_idx is not None:
            idx = next_idx

    sample = samples[idx]

    # Display sample
    st.subheader(f"Sample {idx + 1}/{len(samples)} — {sample['sample_id']}")
    st.caption(f"Corpus: {sample['corpus']} | Trait: {sample['trait_id']}")

    st.markdown("**Rubric Definition:**")
    st.info(sample["rubric_description"])

    judge_score = sample.get("judge_score", "?")
    st.warning(
        f"An AI judge scored this response **{judge_score}/10** on the rubric above. "
        "Your task: determine whether the response genuinely satisfies the rubric or is gaming it."
    )

    col1, col2 = st.columns(2)
    with col1, st.expander("**User Prompt**", expanded=True):
        st.markdown(sample["user_prompt"])
    with col2, st.expander("**Reference Response**", expanded=True):
        st.markdown(sample["reference_response"])

    with st.expander("**Attacker Response** (the response to evaluate)", expanded=True):
        st.markdown(sample["attacker_response"])

    judge_resp = sample.get("judge_response", "")
    if judge_resp:
        with st.expander("**Judge Reasoning** (why the judge gave this score)"):
            st.markdown(judge_resp)

    st.divider()

    # Annotation form
    st.subheader("Your Annotation")

    current_ann = sample["annotation"]
    ann_options = ["(not annotated)", "Yes", "No"]
    default_idx = 0
    if current_ann == "Yes":
        default_idx = 1
    elif current_ann == "No":
        default_idx = 2

    annotation = st.radio(
        "Is this response genuinely exploiting the rubric?",
        ann_options,
        index=default_idx,
        horizontal=True,
        help="Does the response appear to satisfy the rubric on the surface "
        "but actually violate its intent?",
    )

    exploit_type = None
    if annotation == "Yes":
        current_type = sample.get("exploit_type") or EXPLOIT_TYPES[0]
        type_idx = (
            EXPLOIT_TYPES.index(current_type) if current_type in EXPLOIT_TYPES else 0
        )
        exploit_type = st.selectbox("Exploit type", EXPLOIT_TYPES, index=type_idx)

    conf_options = [1, 2, 3]
    current_conf = sample.get("confidence") or 2
    confidence = st.radio(
        "Confidence",
        conf_options,
        index=conf_options.index(current_conf) if current_conf in conf_options else 1,
        horizontal=True,
        format_func=lambda x: {
            1: "1 (guessing)",
            2: "2 (somewhat)",
            3: "3 (very confident)",
        }[x],
    )

    notes = st.text_input("Notes (optional)", value=sample.get("notes") or "")

    if st.button("Save", type="primary"):
        if annotation == "(not annotated)":
            st.warning("Please select Yes or No before saving.")
        else:
            samples[idx]["annotation"] = annotation
            samples[idx]["exploit_type"] = exploit_type if annotation == "Yes" else None
            samples[idx]["confidence"] = confidence
            samples[idx]["notes"] = notes if notes else None

            # Write back
            with open(file_path, "w") as f:
                f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)

            st.success(f"Saved {sample['sample_id']}!")
            st.rerun()


def _find_next_unannotated(samples, start=0):
    """Find next unannotated sample index starting from `start`."""
    for i in range(start, len(samples)):
        if samples[i]["annotation"] is None:
            return i
    for i in range(start):
        if samples[i]["annotation"] is None:
            return i
    return start


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human annotation of adversarial verifier verdicts."
    )
    parser.add_argument(
        "--generate", action="store_true", help="Generate annotation JSONL files"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Compute precision/recall from completed annotations",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON list of corpora to sample from (required with --generate); "
        "each entry needs label, responses, traits — see the module docstring",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Where annotation files and the answer key live "
        f"(default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-annotators",
        type=int,
        default=4,
        help="How many annotator files to split the sample across",
    )
    args = parser.parse_args()

    if args.generate:
        if not args.config:
            parser.error("--generate requires --config")
        generate_samples(
            load_corpus_config(args.config),
            output_dir=Path(args.output_dir),
            seed=args.seed,
            n_annotators=args.n_annotators,
        )
        print(
            "\nNext: set runtime.annotation_output_dir in config.json to "
            f"{args.output_dir!r}, then run\n"
            "  streamlit run "
            "$(python -c 'import premise.rubric_evaluation.annotate_verifier as m; "
            "print(m.__file__)')"
        )
    elif args.analyze:
        analyze(output_dir=Path(args.output_dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    # Under `streamlit run`, streamlit owns argv — serve the UI instead of parsing it.
    if "streamlit" in sys.modules and hasattr(sys.modules["streamlit"], "runtime"):
        run_streamlit()
    else:
        main()
