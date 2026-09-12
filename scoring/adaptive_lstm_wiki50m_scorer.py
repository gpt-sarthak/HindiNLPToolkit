"""
scoring.adaptive_lstm_wiki50m_scorer
====================================
Adaptive LSTM surprisal from the Wikipedia-50M checkpoint.

The adaptive sibling of ``lstm_wiki50m``, mirroring how ``adaptive_lstm``
relates to ``lstm``: for each sentence it starts from pristine base weights,
takes one SGD step on the *preceding* sentence, and scores the reference and its
variants under those adapted weights.  Reset semantics — nothing accumulates
across sentences.

Two honest caveats, both measured on 92,299 pairs
-------------------------------------------------
* **The adaptation is worth ~+0.03 points** over the plain
  ``lstm_wiki50m`` (92.63% vs 92.60%), while deep-copying a 63 MB model for
  every sentence.  It ships for parity with the existing
  ``lstm`` / ``adaptive_lstm`` pair and so the two generations can be compared
  like for like — not because the adaptation earns its cost.
* **Unlike ``adaptive_lstm``, this scorer is deterministic.**  That one runs the
  gradient step with dropout live, so two runs of the same job disagree on
  *every row* by up to 2.11 nats.  Here the dropout probability is zeroed for
  the step (see ``_lstm_wiki50m.adapt_one_step``), which costs nothing
  measurable and makes ``Delta_AdaptiveWiki50M`` reproducible.

See ``D:\\audit\\lstm\\FINDINGS.md``.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import pandas as pd

from helpers import heartbeat

from ._lstm_wiki50m import adapt_one_step, batch_surprisal, get_lstm_wiki50m
from .base import Scorer

_ADAPT_LR = 0.01  # the research default; measured irrelevant across 0.001-0.01


class AdaptiveLSTMWiki50MScorer(Scorer):
    name = "adaptive_lstm_wiki50m"
    display_name = "Adaptive LSTM (Wikipedia 50M)"
    description = (
        "LSTM surprisal after adapting to the preceding sentence, scoring "
        "every word including the sentence boundaries.\n"
        "Feature in CSV: Delta_AdaptiveWiki50M."
    )
    trained_on = "50.8M tokens of Hindi Wikipedia, 30k vocabulary"
    built_with = "2-layer LSTM + one-step online adaptation"
    notes = "surprisal in nats; deterministic, unlike the older adaptive scorer"
    needs_previous_sentence = True
    higher_is_more_natural = {"Delta_AdaptiveWiki50M": False}

    def score(self, pairs_df: pd.DataFrame, context=None) -> pd.DataFrame:
        df = pairs_df.copy()
        if df.empty:
            df["AdaptiveWiki50M_Reference"] = []
            df["AdaptiveWiki50M_Variant"] = []
            return df

        try:
            base_model, word2idx, device = get_lstm_wiki50m()
        except Exception:
            df["AdaptiveWiki50M_Reference"] = [float("nan")] * len(df)
            df["AdaptiveWiki50M_Variant"] = [float("nan")] * len(df)
            return df

        corpus = (context or {}).get("corpus")

        groups = defaultdict(list)
        for pos, sid in enumerate(df["Sent_ID"]):
            groups[sid].append(pos)

        n = len(df)
        ref_vals = [float("nan")] * n
        var_vals = [float("nan")] * n

        for sid, rows in groups.items():
            heartbeat()  # cancellation checkpoint (no-op outside a job)

            ctx_text = ""
            if corpus is not None:
                try:
                    ctx_text = corpus.previous_text(sid, 1)
                except Exception:
                    ctx_text = ""

            # Pristine weights every sentence: the shared singleton is never
            # mutated, so concurrent jobs cannot interfere with each other.
            model = copy.deepcopy(base_model)
            if ctx_text and ctx_text.strip():
                adapt_one_step(ctx_text, model, word2idx, device, lr=_ADAPT_LR)
            model.eval()

            # One reference per group: score once and broadcast.
            ref_score = batch_surprisal(
                [df["Reference_Sentence"].iat[rows[0]]], model, word2idx,
                device, batch_size=1)[0]
            var_scores = batch_surprisal(
                [df["Variant_Sentence"].iat[i] for i in rows],
                model, word2idx, device)
            for i, v in zip(rows, var_scores):
                ref_vals[i] = ref_score
                var_vals[i] = v

        df["AdaptiveWiki50M_Reference"] = ref_vals
        df["AdaptiveWiki50M_Variant"] = var_vals
        return df

    def deltas(self):
        return [("Delta_AdaptiveWiki50M",
                 lambda row: row["AdaptiveWiki50M_Reference"],
                 lambda row: row["AdaptiveWiki50M_Variant"])]
