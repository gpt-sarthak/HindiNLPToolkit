"""
scoring.lstm_wiki50m_scorer
===========================
LSTM language-model surprisal from the Wikipedia-50M checkpoint.

Sits **alongside** the older ``lstm`` scorer rather than replacing it, so
``Delta_LSTM``'s committed weight stays valid and `variants.csv` files produced
before this scorer existed remain comparable.

It differs from ``lstm`` in two independent ways, each measured on the research
corpus's 92,299 pairs (``D:\\audit\\lstm\\FINDINGS.md``):

* **Scoring** (+5.65 pts).  Pads with ``<SOS>``/``<EOS>`` and applies the
  model's training-time cleaning, so every word contributes — including the
  first, which ``lstm`` never predicts, and the sentence end, which it never
  scores.
* **Model** (+9.50 pts).  Trained on 50.8M tokens rather than ~2M, with a
  held-out split, early stopping and gradient clipping.  Same architecture and
  the same 30k vocabulary, so the two differ only in training.

Together: **77.45% → 92.60%** ranking accuracy.

⚠ The checkpoint stopped on a 25-minute training budget, not convergence
(validation loss was still falling), so 92.60% is a floor rather than a ceiling.
"""

from __future__ import annotations

import pandas as pd

from helpers import heartbeat

from ._lstm_wiki50m import batch_surprisal, get_lstm_wiki50m
from .base import Scorer


class LSTMWiki50MScorer(Scorer):
    name = "lstm_wiki50m"
    display_name = "LSTM (Wikipedia 50M)"
    description = (
        "LSTM language-model surprisal, scoring every word including the "
        "sentence boundaries.\n"
        "Feature in CSV: Delta_LSTMWiki50M."
    )
    trained_on = "50.8M tokens of Hindi Wikipedia (dumps.wikimedia.org/hiwiki), 30k vocabulary"
    built_with = "2-layer LSTM language model"
    notes = "surprisal in nats; batched, so a corpus pass is fast"
    higher_is_more_natural = {"Delta_LSTMWiki50M": False}  # lower = more natural

    def score(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        df = pairs_df.copy()
        if df.empty:
            df["LSTMWiki50M_Reference"] = []
            df["LSTMWiki50M_Variant"] = []
            return df

        try:
            model, word2idx, device = get_lstm_wiki50m()
        except Exception:
            # Missing or unreadable checkpoint degrades to NaN rather than
            # killing the run, matching the other model scorers.
            df["LSTMWiki50M_Reference"] = [float("nan")] * len(df)
            df["LSTMWiki50M_Variant"] = [float("nan")] * len(df)
            return df

        # One heartbeat per batch: frequent enough to keep a long run
        # cancellable, without a per-row call the batching exists to avoid.
        df["LSTMWiki50M_Reference"] = batch_surprisal(
            df["Reference_Sentence"].tolist(), model, word2idx, device,
            on_batch=heartbeat)
        df["LSTMWiki50M_Variant"] = batch_surprisal(
            df["Variant_Sentence"].tolist(), model, word2idx, device,
            on_batch=heartbeat)
        return df

    def deltas(self):
        return [("Delta_LSTMWiki50M",
                 lambda row: row["LSTMWiki50M_Reference"],
                 lambda row: row["LSTMWiki50M_Variant"])]
