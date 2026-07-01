"""
scoring.lstm_scorer
===================
Base LSTM language-model surprisal scorer.

For each (reference, variant) pair it computes the total next-word surprisal of
each word order under a 2-layer LSTM trained on Hindi text
(``scoring/models/base_model.pt`` + ``vocab.pkl``) and declares ``Delta_LSTM``,
oriented centrally by ``ML_Label``.

The model machinery is shared with the adaptive scorer; see
``scoring/_lstm_common.py``.  Loading is deferred to first use so plugin
discovery stays cheap.

Discovered automatically by the scoring package — appears as the ``lstm``
checkbox in the UI.
"""

from __future__ import annotations

from ._lstm_common import get_lstm, sentence_lstm_surprisal
from .base import Scorer


class LSTMScorer(Scorer):
    name = "lstm"
    description = (
        "LSTM language-model surprisal (nats) of each word order. "
        "Advantage: Delta_LSTM."
    )
    trained_on = "Hindi Wikipedia"
    built_with = "2-layer LSTM language model (embed 256 -> LSTM 256, dropout 0.3)"

    def score(self, pairs_df):
        df = pairs_df.copy()
        if df.empty:
            df["LSTM_Reference"] = []
            df["LSTM_Variant"] = []
            return df

        try:
            model, word2idx, device = get_lstm()
        except Exception:
            df["LSTM_Reference"] = [float("nan")] * len(df)
            df["LSTM_Variant"] = [float("nan")] * len(df)
            return df

        cache: dict = {}

        def surprisal(text) -> float:
            key = str(text)
            if key not in cache:
                cache[key] = sentence_lstm_surprisal(key, model, word2idx, device)
            return cache[key]

        df["LSTM_Reference"] = [surprisal(s) for s in df["Reference_Sentence"]]
        df["LSTM_Variant"] = [surprisal(s) for s in df["Variant_Sentence"]]
        return df

    def deltas(self):
        return [("Delta_LSTM",
                 lambda row: row["LSTM_Reference"],
                 lambda row: row["LSTM_Variant"])]
