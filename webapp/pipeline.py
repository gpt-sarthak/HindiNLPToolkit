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
    """
    from filtering import filter_sentences, summarize
    from scoring import apply_scorers, build_corpus_context
    from stanza_parser import load_input
    from variants import generate_variants

    out = jobs.job_dir(job.job_id)

    # ── Stage 1: parse ────────────────────────────────────────────────────
    job.stage = "parse"
    sentences = load_input(str(input_path))
    if input_path.suffix.lower() == ".txt":
        # .txt input went through Stanza — offer the resulting CoNLL-U.
        conllu_text = "".join(sent.serialize() for sent in sentences)
        (out / "parsed.conllu").write_text(conllu_text, encoding="utf-8")
        job.artifacts.append("parsed.conllu")

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
            "corpus": build_corpus_context(sentences),
            "passed": passed,
            "scheme": options.get("scheme"),
        }
        pairs_df = apply_scorers(pairs_df, scorer_names, context=context)
    pairs_df.to_csv(out / "variants.csv", index=False, encoding="utf-8")
    job.artifacts.append("variants.csv")

    job.stage = "complete"
