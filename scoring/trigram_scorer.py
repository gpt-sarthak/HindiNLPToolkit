"""
scoring.trigram_scorer
======================
Trigram language-model surprisal scorer.

Backed by a Katz-backoff model over Good-Turing discounted counts, trained on
1M Hindi Wikipedia sentences with a 30k vocabulary and stored as ten
memory-mapped files in ``scoring/models/trigram_katz/``.  For each
(reference, variant) pair it computes the total sentence surprisal of each word
order and declares ``Delta_Trigram``, oriented centrally by ``ML_Label``.

Scoring
-------
The sentence is padded with ``<s> <s>`` and scored through ``</s>``, so every
word contributes — including the first two.  The input first goes through the
model's training-time cleaning (``scoring/_katz_trigram.py:normalize``); that
step is not optional, because the training corpus was split on ``[।!?]`` and a
sentence-final danda therefore never appeared in training.

Sentence surprisal = sum of per-position ``-log2 P`` over the padded sequence,
i.e. the total is reported in **bits**.

Smoothing is Katz backoff with data-derived backoff weights over Good-Turing
discounted counts — a genuine probability distribution, so an unseen trigram
falls back to the bigram and then the unigram rather than hitting a floor.

The heavy lifting lives in ``scoring/_katz_trigram.py`` (private, no Scorer
subclass, so plugin discovery skips it).

Discovered automatically by the scoring package — appears as the ``trigram``
checkbox in the UI.
"""

from __future__ import annotations

import pandas as pd

from helpers import heartbeat

from .base import Scorer


class TrigramScorer(Scorer):
    name = "trigram"
    display_name = "Trigram"
    description = (
        "Trigram language-model surprisal, scoring every word including the "
        "sentence boundaries.\n"
        "Feature in CSV: Delta_Trigram."
    )
    trained_on = "1M Hindi Wikipedia sentences, 30k vocabulary"
    built_with = "Katz backoff over Good-Turing discounted counts"
    notes = "surprisal in bits; memory-mapped, so it loads instantly"
    higher_is_more_natural = {"Delta_Trigram": False}  # lower surprisal = more natural

    def score(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        df = pairs_df.copy()
        if df.empty:
            df["Trigram_Reference"] = []
            df["Trigram_Variant"] = []
            return df

        try:
            from scoring._katz_trigram import get_lm

            lm = get_lm()
        except Exception:
            # Missing or unreadable model degrades to NaN rather than killing
            # the run, matching the other model scorers.
            df["Trigram_Reference"] = [float("nan")] * len(df)
            df["Trigram_Variant"] = [float("nan")] * len(df)
            return df

        cache: dict = {}

        def surprisal(text) -> float:
            heartbeat()          # cancellation checkpoint (no-op outside a job)
            key = str(text)
            if key not in cache:
                cache[key] = lm.surprisal(key)
            return cache[key]

        df["Trigram_Reference"] = [surprisal(s) for s in df["Reference_Sentence"]]
        df["Trigram_Variant"] = [surprisal(s) for s in df["Variant_Sentence"]]
        return df

    def deltas(self):
        return [("Delta_Trigram",
                 lambda row: row["Trigram_Reference"],
                 lambda row: row["Trigram_Variant"])]
