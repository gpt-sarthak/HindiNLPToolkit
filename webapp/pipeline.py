"""
webapp.pipeline
===============
Thin adapter between the web layer and the logic packages.  Runs the full
parse → filter → variants(+scorers) pipeline for one job, writing each
artifact to disk the moment its stage completes so downloads become
available progressively.

``heartbeat()`` is called at every checkpoint along the way: a cancelled job
raises ``JobCancelled`` there and unwinds, leaving whatever was already written
on disk (see webapp/jobs.py and helpers/cancellation.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from helpers import JobCancelled, heartbeat

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

    from evaluation import evaluate_by_construction, evaluate_pairs
    from scoring import get_scorers
    from variants import WORD_ORDER_COLUMNS

    # Present each delta under the title of the scorer that declared it.
    labels = {
        col: (getattr(scorer, "display_name", "") or scorer.name)
        for scorer in get_scorers().values()
        for (col, _ref_fn, _var_fn) in scorer.deltas()
    }
    result = evaluate_pairs(pairs_df, deltas=delta_cols, labels=labels)
    payload = {"status": "ok", "min_pairs": MIN_EVAL_PAIRS, **result}
    # The same measurement per word-order construction (SOV references vs any
    # variant, then paper Table 4: OSV / DOSV / IOSV references vs canonical
    # variants), each under the same 200-pair policy.
    if all(c in pairs_df.columns for c in WORD_ORDER_COLUMNS):
        payload["constructions"] = evaluate_by_construction(
            pairs_df, deltas=delta_cols, labels=labels, min_pairs=MIN_EVAL_PAIRS
        )
    return payload


def _recommendation_payload(pairs_df, scorer_names) -> dict:
    """Most-natural-order pick for recommendation.json, or a status stub when
    the pairs table cannot support one (no rankable features / no variants)."""
    from scoring import recommend_order

    rec = recommend_order(pairs_df, scorer_names)
    if not rec["features"]:
        return {"status": "no_scorers", **rec}
    if not rec["sentences"]:
        return {"status": "no_pairs", **rec}
    return {"status": "ok", **rec}


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
    recommend        : bool — when True (the default), rank each sentence's
                       reference and variants with the ticked scorers' features
                       and append the Recommended_* columns to variants.csv
    recommend_json   : bool (Sentence mode) — when True, also write
                       recommendation.json (the most natural word order with
                       its full candidate list)
    context_text     : optional preceding sentence (Sentence mode) parsed and
                       prepended to the scoring corpus so context-aware scorers
                       can see a predecessor; never filtered or permuted
    """
    from filtering import filter_sentences, summarize
    from scoring import apply_scorers, build_corpus_context, recommend_column
    from stanza_parser import load_input
    from variants import generate_variants, label_word_orders

    out = jobs.job_dir(job.job_id)

    # ── Stage 1: parse ────────────────────────────────────────────────────
    job.stage = "parse"
    heartbeat()
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
    heartbeat()          # before the stage marker, so a cancelled job reports
    job.stage = "filter"  # the stage it actually died in
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
    heartbeat()
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
    # Word-order construction labels (SOV / DOSV / IOSV) for both sides of
    # every pair — surface metadata, joined before any scorer so they are in
    # variants.csv in both modes, with or without scorers, and in the
    # cancel-flush below.
    if not pairs_df.empty:
        pairs_df = pairs_df.join(label_word_orders(pairs_df, passed))
    heartbeat()
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
        try:
            pairs_df = apply_scorers(pairs_df, scorer_names, context=context)
            heartbeat()
            # Recommended word order per source sentence, broadcast across its
            # rows (one column set, constant within each Sent_ID).  None when no
            # ticked scorer contributes a rankable feature.
            if options.get("recommend", True):
                rec_cols = recommend_column(pairs_df, scorer_names)
                if rec_cols is not None:
                    pairs_df = pairs_df.join(rec_cols)
        except JobCancelled:
            # Cancelled mid-scoring: flush what we have rather than nothing,
            # then unwind.  `pairs_df` is only rebound on a *successful* call,
            # so this writes either the plain surface pairs or a fully scored
            # table — never a half-filled one.
            pairs_df.to_csv(out / "variants.csv", index=False, encoding="utf-8")
            job.artifacts.append("variants.csv")
            raise
    pairs_df.to_csv(out / "variants.csv", index=False, encoding="utf-8")
    job.artifacts.append("variants.csv")
    heartbeat()

    # Optional ranking-accuracy evaluation over the scorers' Delta_* columns
    # (File mode opt-in; still part of the "variants" stage for the UI).
    if options.get("evaluate"):
        payload = _evaluation_payload(pairs_df)
        (out / "evaluation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )
        job.artifacts.append("evaluation.json")

    # Sentence-mode recommendation: pick the most natural word order among the
    # reference and its variants (still part of the "variants" stage).
    if options.get("recommend_json"):
        payload = _recommendation_payload(pairs_df, scorer_names)
        (out / "recommendation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )
        job.artifacts.append("recommendation.json")

    job.stage = "complete"
