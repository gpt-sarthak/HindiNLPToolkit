"""
scoring.dl_scorer
=================
Dependency-length (DL) feature scorer.

Computes the 5-element dependency-length feature vector for both the reference
and the variant order of every pair, following the dependency-length-
minimization convention of Gildea & Jaeger (2015):

    [total_DL, last_DL, second_last_DL, last_len, second_last_len]

with per-arc length measured as ``(arc_length - 1)``.

Why this is a context-aware scorer
-----------------------------------
``pairs_df`` carries only surface strings — the variant generator no longer
computes features.  To score a variant this scorer needs the reference parse
(from ``context["passed"]``) and the variant's constituent order, which it
recovers from the variant surface string by matching each reference
constituent's token block (the same technique the IS scorer uses).  It then
rebuilds the reordered, re-indexed dependency tree and extracts features.

Output
------
Adds ``Ref_Features`` and ``Var_Features`` (the two 5-vectors, in true roles).
The headline advantage ``Delta_DL`` (the ML_Label-oriented difference of
``total_DL``) is produced by the central delta step in ``scoring.apply_scorers``
via :meth:`deltas`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from helpers import rebuild_variant_tree
from .base import Scorer

_ZERO_VECTOR = [0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# DL metric helpers (the generic variant reconstruction lives in helpers/)
# ---------------------------------------------------------------------------

def _calculate_sentence_total_dl(sentence) -> int:
    """Sum of (arc_length - 1) over all non-root dependency arcs."""
    total = 0
    for tok in sentence:
        if isinstance(tok["id"], int) and tok["head"] != 0:
            dist = abs(tok["id"] - tok["head"])
            if dist > 0:
                total += dist - 1
    return total


def _get_constituent_head_distance(constituent: List[dict], root_id: int) -> int:
    """Dependency length between a constituent's attachment point and the root,
    measured as (|position_gap| - 1).  0 if no direct attachment is found."""
    for tok in constituent:
        if isinstance(tok["id"], int) and tok["head"] == root_id:
            return max(0, abs(tok["id"] - root_id) - 1)
    return 0


def _extract_features(sentence, constituents: List[List[dict]], root_id: int) -> List[int]:
    """Build the 5-element DL feature vector
    [total_DL, last_DL, second_last_DL, last_len, second_last_len]."""
    total_dl = _calculate_sentence_total_dl(sentence)
    last_dl = _get_constituent_head_distance(constituents[-1], root_id) if constituents else 0
    second_last_dl = (
        _get_constituent_head_distance(constituents[-2], root_id)
        if len(constituents) >= 2 else 0
    )
    last_len = len(constituents[-1]) if constituents else 0
    second_last_len = len(constituents[-2]) if len(constituents) >= 2 else 0
    return [total_dl, last_dl, second_last_dl, last_len, second_last_len]


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------

class DependencyLengthScorer(Scorer):
    name = "dependency_length"
    description = (
        "Dependency-length feature vectors [total_DL, last_DL, second_last_DL, "
        "last_len, second_last_len] for the reference and variant orders "
        "(per-arc length = arc_length - 1; Gildea & Jaeger 2015). Advantage: Delta_DL."
    )

    def score(self, pairs_df: pd.DataFrame, context: Optional[dict] = None) -> pd.DataFrame:
        df = pairs_df.copy()
        passed = (context or {}).get("passed")

        if df.empty or not passed:
            df["Ref_Features"] = [list(_ZERO_VECTOR) for _ in range(len(df))]
            df["Var_Features"] = [list(_ZERO_VECTOR) for _ in range(len(df))]
            return df

        # sent_id -> (sentence TokenList, root_id, constituents)
        parse_by_id: Dict[str, Tuple[object, int, List[List[dict]]]] = {}
        for item in passed:
            sid = item["sentence"].metadata.get("sent_id", "Unknown_ID")
            parse_by_id.setdefault(sid, (item["sentence"], item["root_id"], item["constituents"]))

        ref_feats: Dict[str, List[int]] = {}   # cache per sentence
        ref_col: List[List[int]] = []
        var_col: List[List[int]] = []

        for sent_id, variant_sentence in zip(df["Sent_ID"], df["Variant_Sentence"]):
            parse = parse_by_id.get(sent_id)
            if parse is None:
                ref_col.append(list(_ZERO_VECTOR))
                var_col.append(list(_ZERO_VECTOR))
                continue

            sentence, root_id, constituents = parse

            if sent_id not in ref_feats:
                ref_feats[sent_id] = _extract_features(sentence, constituents, root_id)
            ref_col.append(list(ref_feats[sent_id]))
            var_col.append(self._variant_features(sentence, root_id, constituents, str(variant_sentence)))

        df["Ref_Features"] = ref_col
        df["Var_Features"] = var_col
        return df

    @staticmethod
    def _variant_features(sentence, root_id, constituents, variant_sentence) -> List[int]:
        """Rebuild the variant's reordered, re-indexed tree (via the shared
        helper) and extract DL features from it."""
        vt = rebuild_variant_tree(sentence, constituents, root_id, variant_sentence)
        if not vt.tokens:
            return list(_ZERO_VECTOR)
        return _extract_features(vt.tokens, vt.constituents, vt.root_id)

    def deltas(self):
        """Delta_DL = ML_Label-oriented difference of total_DL (feature[0])."""
        return [(
            "Delta_DL",
            lambda row: row["Ref_Features"][0],
            lambda row: row["Var_Features"][0],
        )]
