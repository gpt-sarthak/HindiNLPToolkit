"""
helpers.grouping
================
Stable integer keys for a submitted file, and the per-reference grouping built
on them.

Every sentence is stamped with its 1-based position in the submitted file
(``metadata["hnt_index"]``) as soon as it is loaded — *before* filtering, so a
passed sentence keeps its original position and rejected ones leave gaps.
``generate_variants`` carries that position into the pairs table as
``Ref_Index``, next to ``Variant_No`` (1..n within the reference).  Sorting by
``(Ref_Index, Variant_No)`` therefore reproduces the submitted file's order.

``Sent_ID`` cannot serve as that key: it is free text (``s10`` sorts before
``s2``), it falls back to ``"Unknown_ID"`` when the metadata is missing, and an
uploaded ``.conllu`` may repeat it — any of which would merge two different
references into one group.

    from helpers import reference_groups, reference_lookup, stamp_ref_index
    stamp_ref_index(sentences)                  # done by load_input already
    parses = reference_lookup(pairs_df, passed) # row -> filter-output item
    for key, positions in reference_groups(pairs_df):
        ...                                     # one reference + its variants

Depends on the standard library only (the DataFrame is read through plain
column access, so pandas is not imported here).
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence, Tuple

#: Metadata key holding a sentence's 1-based position in the submitted file.
REF_INDEX_KEY = "hnt_index"

#: Pairs-table columns: the reference's file position and the variant number.
REF_INDEX_COLUMN = "Ref_Index"
VARIANT_NO_COLUMN = "Variant_No"


def stamp_ref_index(sentences: Sequence, start: int = 1) -> None:
    """Write each sentence's position (``start``, ``start+1``, ...) into its
    ``metadata[REF_INDEX_KEY]``, in place.  Stored as a string, like every other
    CoNLL-U metadata value."""
    for offset, sent in enumerate(sentences):
        sent.metadata[REF_INDEX_KEY] = str(start + offset)


def ref_index_of(sentence) -> Optional[int]:
    """A sentence's stamped position, or ``None`` when it was never stamped."""
    raw = sentence.metadata.get(REF_INDEX_KEY)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def passed_ref_indices(passed: Sequence[dict]) -> List[int]:
    """
    The ``Ref_Index`` of each filter-output item, in order.

    The stamped file positions when every item carries a distinct one (the
    normal case — ``load_input`` stamps them); otherwise 1..n by position in
    *passed*, for library callers who built their TokenLists by hand.  Both
    ``generate_variants`` and :func:`reference_lookup` use this one rule, so the
    pairs table and the parse lookup always agree.
    """
    stamped = [ref_index_of(item["sentence"]) for item in passed]
    if None not in stamped and len(set(stamped)) == len(stamped):
        return stamped  # type: ignore[return-value]
    return list(range(1, len(passed) + 1))


def reference_lookup(pairs_df, passed: Sequence[dict]) -> List[Optional[dict]]:
    """
    The filter-output item (``sentence`` / ``root_id`` / ``constituents``) for
    every row of *pairs_df*, or ``None`` where no parse matches.

    Matched on ``Ref_Index`` when the table has it; otherwise on ``Sent_ID``
    (first occurrence wins — the pre-``Ref_Index`` behaviour).
    """
    if REF_INDEX_COLUMN in pairs_df.columns:
        by_index = dict(zip(passed_ref_indices(passed), passed))
        return [by_index.get(_as_int(k)) for k in pairs_df[REF_INDEX_COLUMN]]
    by_id: Dict[str, dict] = {}
    for item in passed:
        by_id.setdefault(item["sentence"].metadata.get("sent_id", "Unknown_ID"), item)
    return [by_id.get(sid) for sid in pairs_df["Sent_ID"]]


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def reference_groups(pairs_df) -> List[Tuple[Hashable, List[int]]]:
    """
    Group a pairs table's rows by their reference sentence.

    Returns ``[(key, positions), ...]`` in first-appearance order, where
    *positions* are 0-based row positions (``iloc``) — so a scorer can compute
    in any order it likes and still write results back to the right rows.

    The key is ``Ref_Index`` when the table has it; otherwise (a table built
    before that column existed) ``(Sent_ID, Reference_Sentence)``, which still
    keeps two different references with the same ``Sent_ID`` apart.
    """
    if REF_INDEX_COLUMN in pairs_df.columns:
        keys = list(pairs_df[REF_INDEX_COLUMN])
    else:
        keys = list(zip(pairs_df["Sent_ID"], pairs_df["Reference_Sentence"]))

    groups: Dict[Hashable, List[int]] = {}
    for pos, key in enumerate(keys):
        groups.setdefault(key, []).append(pos)
    return list(groups.items())
