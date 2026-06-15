"""
filtering
=========
Seven individual filter functions plus a combined pipeline and summarizer.

Individual filters (call any subset independently)
---------------------------------------------------
filter_questions(sentences, check_punct, check_pos, check_features)
filter_negatives(sentences, check_pos, check_features)
filter_ghost_ids(sentences)
filter_non_projective(sentences)
filter_bad_root(sentences, allowed_pos)
filter_punct_constituents(sentences_or_items, allowed_pos)
filter_min_phrases(sentences_or_items, min_phrases, allowed_pos)

Combined pipeline
-----------------
filter_sentences(sentences, allowed_root_pos, min_phrases, output_dir)

Summary
-------
summarize(passed_list, rejected_df)
"""

from .filters import (
    filter_bad_root,
    filter_ghost_ids,
    filter_min_phrases,
    filter_negatives,
    filter_non_projective,
    filter_punct_constituents,
    filter_questions,
    filter_sentences,
    summarize,
)

__all__ = [
    "filter_questions",
    "filter_negatives",
    "filter_ghost_ids",
    "filter_non_projective",
    "filter_bad_root",
    "filter_punct_constituents",
    "filter_min_phrases",
    "filter_sentences",
    "summarize",
]
