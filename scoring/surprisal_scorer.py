"""
scoring/surprisal_scorer.py
===========================
Constituency (PCFG) surprisal scorer for the Word-Order pipeline, backed by the
local Taru / SyntacticTreeSurprisal engine (HDTB Berkeley grammar + synproc).

Because Taru runs in the SAME process (mounted via webapp/taru_routes.py), this
scorer calls it directly — no HTTP, no Hugging Face. For each (reference,
variant) pair it computes total constituency surprisal of each word order and
declares Delta_Surprisal, oriented centrally by ML_Label.

Discovered automatically by the scoring package — appears as the `surprisal`
checkbox in the UI (matching CONTRIBUTING.md).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from .base import Scorer

# Make the Taru backend importable (same path the router uses).
_TARU_WS = Path(__file__).resolve().parent.parent / "taru" / "workspace"
if str(_TARU_WS) not in sys.path:
    sys.path.insert(0, str(_TARU_WS))


def _backend():
    import taru_backend as tb
    return tb


@lru_cache(maxsize=4096)
def _sentence_surprisal(sentence: str) -> float:
    """Total HDTB constituency surprisal (bits) for one word order. Cached."""
    tb = _backend()
    try:
        obj = tb.parse_one(sentence, model_id="hdtb", want_surprisal=True)
        surp = obj.get("surprisal") or {}
        return float(sum(float(v) for v in surp.values()))
    except Exception:
        return float("nan")


class SurprisalScorer(Scorer):
    name = "surprisal"
    description = ("Constituency (PCFG) incremental surprisal (bits) of each "
                  "word order. Advantage: Delta_Surprisal.")
    trained_on = "HDTB (Hindi Dependency Treebank)"
    built_with = "Berkeley 'hdtb_fresh' grammar + Taru synproc incremental parser"

    def score(self, pairs_df):
        df = pairs_df.copy()
        df["Surprisal_Reference"] = [
            _sentence_surprisal(s) for s in df["Reference_Sentence"]
        ]
        df["Surprisal_Variant"] = [
            _sentence_surprisal(s) for s in df["Variant_Sentence"]
        ]
        return df

    def deltas(self):
        return [("Delta_Surprisal",
                 lambda row: row["Surprisal_Reference"],
                 lambda row: row["Surprisal_Variant"])]
