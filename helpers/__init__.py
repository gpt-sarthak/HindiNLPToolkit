"""
helpers
=======
Shared, scorer-agnostic primitives used across the toolkit.

These exist so contributors can reuse the recurring building blocks rather than
re-implementing them inside each scorer:

- **Preceding-sentence lookup** (``corpus_context``): "what sentence(s) came
  before this one?" — needed by givenness / Information Status and friends.
- **Variant reconstruction** (``variant_tree``): rebuild a variant's reordered,
  re-indexed dependency tree from the reference parse + the variant surface
  string — needed by any scorer that measures something positional on a variant
  (dependency length, etc.).

Depends only on ``conllu``; safe to import from any logic package.

    from helpers import (
        CorpusContext, build_corpus_context,
        get_previous_sentences, get_previous_text,
        VariantTree, rebuild_variant_tree,
        recover_permutation, reindex_tokens, block_start_index,
    )
"""

from __future__ import annotations

from .corpus_context import (
    CorpusContext,
    build_corpus_context,
    get_previous_sentences,
    get_previous_text,
)
from .variant_tree import (
    VariantTree,
    block_start_index,
    rebuild_variant_tree,
    recover_permutation,
    reindex_tokens,
)

__all__ = [
    "CorpusContext",
    "build_corpus_context",
    "get_previous_sentences",
    "get_previous_text",
    "VariantTree",
    "rebuild_variant_tree",
    "recover_permutation",
    "reindex_tokens",
    "block_start_index",
]
