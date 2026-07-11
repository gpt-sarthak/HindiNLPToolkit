"""
webapp.pipeline
===============
Thin adapter between the web layer and the logic packages.  Runs the full
parse → filter → variants(+scorers) pipeline for one job, writing each
artifact to disk the moment its stage completes so downloads become
available progressively.
"""

from __future__ import annotations

import json
from pathlib import Path

from webapp import jobs

# Below this many (reference, variant) pairs, ranking accuracy is too noisy to
# show — evaluation.json then carries status "insufficient_pairs" instead.
MIN_EVAL_PAIRS = 200


def _evaluation_payload(pairs_df) -> dict:
    """Ranking-accuracy result for evaluation.json, or a status stub when the
    pairs table cannot support one (no scorer deltas / too few pairs)."""
    delta_cols = [c for c in pairs_df.columns if c.startswith("Delta_")]
    base = {"n_pairs": int(len(pairs_df)), "min_pairs": MIN_EVAL_PAIRS}
    if not delta_cols:
        return {"status": "no_predictors", **base}
    if len(pairs_df) < MIN_EVAL_PAIRS:
        return {"status": "insufficient_pairs", **base}

    from evaluation import evaluate_pairs
    from scoring import get_scorers

    # Present each delta under the name of the scorer that declared it.
    labels = {
        col: scorer.name
        for scorer in get_scorers().values()
        for (col, _ref_fn, _var_fn) in scorer.deltas()
    }
    result = evaluate_pairs(pairs_df, deltas=delta_cols, labels=labels)
    return {"status": "ok", "min_pairs": MIN_EVAL_PAIRS, **result}


def run_job(job: "jobs.Job", input_path: Path, options: dict) -> None:
    """
    Execute the pipeline for *job*.

    options keys
    ------------
    allowed_root_pos : list[str] or None (None = Paninian defaults)
    min_phrases      : int
    max_variants     : int
    grammar_filter   : bool — when False, permutations are not restricted to
                       deprel bigrams observed in the corpus (useful for small
                       uploads where the observed set is too sparse to allow
                       any reordering)
    scorers          : list[str] of scorer names to apply to the pairs table
    evaluate         : bool — when True, run the pairwise ranking-accuracy
                       evaluation over the scorers' Delta_* columns and write
                       evaluation.json (needs >= MIN_EVAL_PAIRS pairs)
    context_text     : optional preceding sentence (Sentence mode) parsed and
                       prepended to the scoring corpus so context-aware scorers
                       can see a predecessor; never filtered or permuted
    """
    from filtering import filter_sentences, summarize
    from scoring import apply_scorers, build_corpus_context
    from stanza_parser import load_input
    from variants import generate_variants

    out = jobs.job_dir(job.job_id)

    # ── Stage 1: parse ────────────────────────────────────────────────────
    job.stage = "parse"
    sentences = load_input(str(input_path))          # target sentence(s)
    if input_path.suffix.lower() == ".txt":
        # .txt input went through Stanza — offer the resulting CoNLL-U (the
        # target only; the tree view reads this single-sentence parse).
        conllu_text = "".join(sent.serialize() for sent in sentences)
        (out / "parsed.conllu").write_text(conllu_text, encoding="utf-8")
        job.artifacts.append("parsed.conllu")

    # Optional context sentence: parse it separately and place it *before* the
    # target in the scoring corpus so context-aware scorers (adaptive_lstm,
    # information_status) can see a preceding sentence. The context is NEVER
    # filtered or permuted — only the target is.
    corpus_sentences = list(sentences)
    context_text = (options.get("context_text") or "").strip()
    if context_text:
        ctx_sentences = load_input(context_text)
        for i, sent in enumerate(ctx_sentences):
            sent.metadata["sent_id"] = f"context_s{i + 1}"
        corpus_sentences = list(ctx_sentences) + list(sentences)

    # ── Stage 2: filter ───────────────────────────────────────────────────
    job.stage = "filter"
    passed, rejected_df, _passed_df = filter_sentences(
        sentences,
        allowed_root_pos=options.get("allowed_root_pos"),
        min_phrases=options.get("min_phrases", 2),
        output_dir=str(out),
    )
    job.artifacts.extend(["passed_sentences.csv", "rejected_sentences.csv"])

    summary = summarize(passed, rejected_df)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=int),
        encoding="utf-8",
    )
    job.artifacts.append("summary.json")
    job.summary = summary

    # ── Stage 3: variants + scorers ───────────────────────────────────────
    job.stage = "variants"
    valid_pairs = None  # None → generate_variants builds it from the corpus
    if not options.get("grammar_filter", True):
        # Cross product of all observed deprel labels: every adjacency is
        # licit, so permutations are limited only by max_variants.
        labels = set()
        for item in passed:
            root_id = item["root_id"]
            for const in item["constituents"]:
                labels.add(
                    next(
                        (t["deprel"] for t in const if t["head"] == root_id),
                        "UNKNOWN",
                    )
                )
        valid_pairs = {(a, b) for a in labels for b in labels}
    pairs_df = generate_variants(
        passed,
        valid_deprel_pairs=valid_pairs,
        max_variants=options.get("max_variants", 99),
    )
    scorer_names = options.get("scorers") or []
    if scorer_names and not pairs_df.empty:
        # Read-only corpus context for scheme-aware scorers (e.g. Information
        # Status, which needs the parse and the preceding sentence).  Built
        # from the full pre-filter sentence list so textual predecessors are
        # preserved even when filtered out.
        context = {
            "corpus": build_corpus_context(corpus_sentences),
            "passed": passed,
            "scheme": options.get("scheme"),
        }
        pairs_df = apply_scorers(pairs_df, scorer_names, context=context)
    pairs_df.to_csv(out / "variants.csv", index=False, encoding="utf-8")
    job.artifacts.append("variants.csv")

    # Optional ranking-accuracy evaluation over the scorers' Delta_* columns
    # (File mode opt-in; still part of the "variants" stage for the UI).
    if options.get("evaluate"):
        payload = _evaluation_payload(pairs_df)
        (out / "evaluation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )
        job.artifacts.append("evaluation.json")

    job.stage = "complete"
