"""
scoring.dl_scorer
=================
Dependency-length (DL) feature scorer.

Computes the total dependency length of both the reference and the variant order
of every pair, following the dependency-length-minimization convention of
Gildea & Jaeger (2015): the sum of ``(arc_length - 1)`` over every non-root
dependency arc.

Why this is a context-aware scorer
-----------------------------------
``pairs_df`` carries only surface strings — the variant generator no longer
computes features.  To score a variant this scorer needs the reference parse
(from ``context["passed"]``) and the variant's constituent order, which it
recovers from the variant surface string by matching each reference
constituent's token block (the same technique the IS scorer uses).  It then
rebuilds the reordered, re-indexed dependency tree and measures it.

Output
------
Adds ``DL_Reference`` and ``DL_Variant`` (the two total dependency lengths, in
true roles).  The advantage ``Delta_DL`` (their ML_Label-oriented difference) is
produced by the central delta step in ``scoring.apply_scorers`` via
:meth:`deltas`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from helpers import rebuild_variant_tree
from .base import Scorer

_OUTPUT_COLUMNS = ("DL_Reference", "DL_Variant")


# ---------------------------------------------------------------------------
# DL metric helper (the generic variant reconstruction lives in helpers/)
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


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------

class DependencyLengthScorer(Scorer):
    name = "dependency_length"
    display_name = "Dependency Length"
    description = (
        "Total dependency length of each word order — the summed distance "
        "between every word and its syntactic head.\n"
        "Feature in CSV: Delta_DL."
    )
    trained_on = "Not trained (deterministic)"
    built_with = "Dependency-length minimization, per-arc length = arc_length - 1"
    higher_is_more_natural = {"Delta_DL": False}  # shorter dependencies = more natural

    def score(self, pairs_df: pd.DataFrame, context: Optional[dict] = None) -> pd.DataFrame:
        df = pairs_df.copy()
        passed = (context or {}).get("passed")

        if df.empty or not passed:
            for col in _OUTPUT_COLUMNS:
                df[col] = [0] * len(df)
            return df

        # sent_id -> (sentence TokenList, root_id, constituents)
        parse_by_id: Dict[str, Tuple[object, int, List[List[dict]]]] = {}
        for item in passed:
            sid = item["sentence"].metadata.get("sent_id", "Unknown_ID")
            parse_by_id.setdefault(sid, (item["sentence"], item["root_id"], item["constituents"]))

        ref_dl: Dict[str, int] = {}   # cache per sentence
        ref_col: List[int] = []
        var_col: List[int] = []

        for sent_id, variant_sentence in zip(df["Sent_ID"], df["Variant_Sentence"]):
            parse = parse_by_id.get(sent_id)
            if parse is None:
                ref_col.append(0)
                var_col.append(0)
                continue

            sentence, root_id, constituents = parse

            if sent_id not in ref_dl:
                ref_dl[sent_id] = _calculate_sentence_total_dl(sentence)
            ref_col.append(ref_dl[sent_id])
            var_col.append(self._variant_dl(sentence, root_id, constituents, str(variant_sentence)))

        df["DL_Reference"] = ref_col
        df["DL_Variant"] = var_col
        return df

    @staticmethod
    def _variant_dl(sentence, root_id, constituents, variant_sentence) -> int:
        """Rebuild the variant's reordered, re-indexed tree (via the shared
        helper) and measure its total dependency length."""
        vt = rebuild_variant_tree(sentence, constituents, root_id, variant_sentence)
        if not vt.tokens:
            return 0
        return _calculate_sentence_total_dl(vt.tokens)

    def deltas(self):
        """Delta_DL = ML_Label-oriented difference of total dependency length."""
        return [("Delta_DL",
                 lambda row: row["DL_Reference"],
                 lambda row: row["DL_Variant"])]
