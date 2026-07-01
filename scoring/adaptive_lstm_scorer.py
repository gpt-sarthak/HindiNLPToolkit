"""
scoring.adaptive_lstm_scorer
============================
Adaptive LSTM surprisal scorer (van Schijndel & Linzen 2018, as applied in
Ranjan & van Schijndel 2024).

The base LSTM is updated by one gradient step on the *preceding* (context)
sentence, then used to score the reference and its variants under the adapted
weights — modelling how discourse context shapes a reader's expectations.

Procedure (per unique reference sentence):
    1. ``deepcopy`` the base model (so the shared base is never mutated).
    2. If a preceding sentence exists, run one SGD step on it.
    3. Score the reference once, then every variant of it, in eval mode.
    4. Discard the adapted copy.

When no preceding sentence is available (first sentence in a document, or no
corpus context), no adaptation runs and the score equals the plain LSTM
surprisal.

This is a *context-aware* scorer: it declares a ``context`` parameter on
``score`` and reads ``context["corpus"]`` (a :class:`helpers.CorpusContext`) for
the preceding-sentence lookup.  ``Delta_Adaptive`` is oriented centrally by
``ML_Label``.

Discovered automatically by the scoring package — appears as the
``adaptive_lstm`` checkbox in the UI.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Optional

from ._lstm_common import adapt_one_step, get_lstm, sentence_lstm_surprisal
from .base import Scorer


class AdaptiveLSTMScorer(Scorer):
    name = "adaptive_lstm"
    description = (
        "Adaptive LSTM surprisal (nats) of each word order, after adapting to "
        "the preceding sentence. Advantage: Delta_Adaptive."
    )
    trained_on = "Hindi Wikipedia (base LSTM)"
    built_with = "base LSTM + one-step online adaptation (van Schijndel & Linzen 2018)"
    needs_previous_sentence = True

    def score(self, pairs_df, context: Optional[dict] = None):
        df = pairs_df.copy()
        n = len(df)
        if df.empty:
            df["Adaptive_Reference"] = []
            df["Adaptive_Variant"] = []
            return df

        try:
            base_model, word2idx, device = get_lstm()
        except Exception:
            df["Adaptive_Reference"] = [float("nan")] * n
            df["Adaptive_Variant"] = [float("nan")] * n
            return df

        corpus = (context or {}).get("corpus")

        # Group rows by source sentence so adaptation runs once per reference,
        # not once per pair.  Positional indices keep us aligned with the df.
        groups: dict = defaultdict(list)
        for pos, sid in enumerate(df["Sent_ID"]):
            groups[sid].append(pos)

        ref_vals = [float("nan")] * n
        var_vals = [float("nan")] * n

        ref_series = df["Reference_Sentence"]
        var_series = df["Variant_Sentence"]

        for sid, positions in groups.items():
            ctx_text = ""
            if corpus is not None:
                try:
                    ctx_text = corpus.previous_text(sid, 1)
                except Exception:
                    ctx_text = ""

            model = copy.deepcopy(base_model)
            if ctx_text and ctx_text.strip():
                adapt_one_step(ctx_text, model, word2idx, device)
            model.eval()

            ref_text = str(ref_series.iloc[positions[0]])
            ref_s = sentence_lstm_surprisal(ref_text, model, word2idx, device)

            for pos in positions:
                ref_vals[pos] = ref_s
                var_vals[pos] = sentence_lstm_surprisal(
                    str(var_series.iloc[pos]), model, word2idx, device
                )

        df["Adaptive_Reference"] = ref_vals
        df["Adaptive_Variant"] = var_vals
        return df

    def deltas(self):
        return [("Delta_Adaptive",
                 lambda row: row["Adaptive_Reference"],
                 lambda row: row["Adaptive_Variant"])]
