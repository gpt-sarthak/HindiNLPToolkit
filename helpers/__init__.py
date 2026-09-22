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
- **Word-order classification** (``word_order``): label a preverbal constituent
  order ``SOV`` / ``DOSV`` / ``IOSV`` (Ranjan & van Schijndel 2024; ``OSV`` =
  ``DOSV`` or ``IOSV``) from its deprel sequence — the same rule for a
  reference parse and a rebuilt variant.

Depends only on ``conllu`` and the standard library; safe to import from any
logic package.

    from helpers import (
        CorpusContext, build_corpus_context,
        get_previous_sentences, get_previous_text,
        VariantTree, rebuild_variant_tree,
        recover_permutation, reindex_tokens, block_start_index,
        heartbeat, register_process, CancelToken, JobCancelled,
        classify_word_order, constituent_deprel, is_osv,
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
from .word_order import (
    DIRECT_OBJECT_DEPRELS,
    INDIRECT_OBJECT_DEPRELS,
    OSV_LABELS,
    SUBJECT_DEPRELS,
    WORD_ORDER_LABELS,
    classify_word_order,
    constituent_deprel,
    is_osv,
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
    "classify_word_order",
    "constituent_deprel",
    "is_osv",
    "SUBJECT_DEPRELS",
    "DIRECT_OBJECT_DEPRELS",
    "INDIRECT_OBJECT_DEPRELS",
    "WORD_ORDER_LABELS",
    "OSV_LABELS",
]
