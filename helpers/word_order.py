"""
helpers.word_order
==================
Classify a clause's preverbal constituent order as canonical or object-fronted,
following Ranjan & van Schijndel (2024), Table 1:

- ``SOV``  — canonical: the subject precedes every object.
- ``DOSV`` — a direct object precedes the subject.
- ``IOSV`` — an indirect object precedes the subject (and no direct object does).

There is deliberately **no stored ``OSV`` label**: the paper's "OSV"
(object-fronted) class is the *union* of ``DOSV`` and ``IOSV`` — its own counts
say so (233 OSV ≈ 133 DOSV + 101 IOSV) — so ``OSV`` is answered by
:func:`is_osv` rather than by a fourth value.  When both a direct and an
indirect object precede the subject the sentence is ``DOSV``: the direct-object
test runs first, exactly as in the research reproduction's
``get_construction_type``.

The input is the deprel sequence of the preverbal constituents in surface
order (``["obj", "nsubj", "obl"]``), which is what both the reference parse and
a rebuilt variant expose — the same function therefore labels either side of a
(reference, variant) pair.  Relation sets union both annotation schemes, like
filter 7 does (the schemes share no label, so no scheme flag is needed):
subjects ``nsubj`` / ``nsubj:pass`` / ``k1``, direct objects ``obj`` / ``k2``,
indirect objects ``iobj`` / ``k4``.

    from helpers import classify_word_order, constituent_deprel, is_osv
    classify_word_order(["nsubj", "obj", "obl"])   # 'SOV'
    classify_word_order(["obj", "nsubj"])          # 'DOSV'
    classify_word_order(["k4", "k1", "k2"])        # 'IOSV'

Depends on the standard library only.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence

# Same core-argument relations filter 7 requires (filtering/filters.py), so a
# sentence is classified with the relations that admitted it.
SUBJECT_DEPRELS: FrozenSet[str] = frozenset({"nsubj", "nsubj:pass", "k1"})
DIRECT_OBJECT_DEPRELS: FrozenSet[str] = frozenset({"obj", "k2"})
INDIRECT_OBJECT_DEPRELS: FrozenSet[str] = frozenset({"iobj", "k4"})

WORD_ORDER_LABELS: tuple = ("SOV", "DOSV", "IOSV")
OSV_LABELS: FrozenSet[str] = frozenset({"DOSV", "IOSV"})


def constituent_deprel(constituent: List[Dict], root_id: int) -> str:
    """Deprel of the token inside *constituent* that attaches directly to
    *root_id* — i.e. the constituent's own relation to the verb.  ``'UNKNOWN'``
    if no token does."""
    for tok in constituent:
        if tok["head"] == root_id:
            return tok["deprel"]
    return "UNKNOWN"


def _first_index(deprels: Sequence[str], rels: FrozenSet[str]) -> Optional[int]:
    for i, d in enumerate(deprels):
        if d in rels:
            return i
    return None


def classify_word_order(deprels: Sequence[str]) -> str:
    """
    Label a preverbal constituent order as ``'SOV'``, ``'DOSV'`` or ``'IOSV'``.

    *deprels* is the sequence of root-attached relations of the preverbal
    constituents in surface order.  Positions are those of the *first*
    constituent bearing each role; constituents are contiguous, non-overlapping
    subtrees, so head order and span order agree.

    - ``DOSV`` if a direct object exists and precedes the subject
    - else ``IOSV`` if an indirect object exists and precedes the subject
    - else ``SOV``

    A clause with an object but no preverbal subject counts as object-fronted
    (only reachable with ``require_core_args=False``, since filter 7 otherwise
    guarantees both); a clause with no object at all is ``SOV``.
    """
    subj = _first_index(deprels, SUBJECT_DEPRELS)
    dobj = _first_index(deprels, DIRECT_OBJECT_DEPRELS)
    iobj = _first_index(deprels, INDIRECT_OBJECT_DEPRELS)
    if dobj is not None and (subj is None or dobj < subj):
        return "DOSV"
    if iobj is not None and (subj is None or iobj < subj):
        return "IOSV"
    return "SOV"


def is_osv(label: str) -> bool:
    """True for the paper's object-fronted class: ``DOSV`` or ``IOSV``."""
    return label in OSV_LABELS


__all__ = [
    "SUBJECT_DEPRELS",
    "DIRECT_OBJECT_DEPRELS",
    "INDIRECT_OBJECT_DEPRELS",
    "WORD_ORDER_LABELS",
    "OSV_LABELS",
    "constituent_deprel",
    "classify_word_order",
    "is_osv",
]
