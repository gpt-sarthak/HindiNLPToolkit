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
- **Cooperative cancellation** (``cancellation``): ``heartbeat()`` checkpoints
  that let the web app stop a long run, and ``register_process()`` so external
  child processes are killed with it. Both are no-ops outside the web app.

Depends only on ``conllu`` and the standard library; safe to import from any
logic package.

    from helpers import (
        CorpusContext, build_corpus_context,
        get_previous_sentences, get_previous_text,
        VariantTree, rebuild_variant_tree,
        recover_permutation, reindex_tokens, block_start_index,
        heartbeat, register_process, CancelToken, JobCancelled,
    )
"""

from __future__ import annotations

from .cancellation import (
    CancelToken,
    JobCancelled,
    bind,
    current,
    heartbeat,
    register_process,
    unbind,
)
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
    "CancelToken",
    "JobCancelled",
    "heartbeat",
    "register_process",
    "bind",
    "unbind",
    "current",
]
