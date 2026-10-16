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
  ``lstm_wiki50m`` (92.63% vs 92.60%), at the cost of one SGD step (and an
  in-place reset of a 63 MB working copy) per sentence.  It ships for parity with the existing
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

import pandas as pd

from helpers import REF_INDEX_COLUMN, heartbeat, reference_groups

from ._lstm_engine import AdaptationCopy, score_group
from ._lstm_wiki50m import PAD, _ids, adapt_one_step, get_lstm_wiki50m
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
    trained_on = "50.8M tokens of Hindi Wikipedia (dumps.wikimedia.org/hiwiki), 30k vocabulary"
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
        has_index = REF_INDEX_COLUMN in df.columns
        encode = lambda text: _ids(text, word2idx)  # noqa: E731
        pad_id = word2idx.get(PAD, 0)
        # One reusable working copy, reset to pristine base weights before every
        # adaptation: the shared singleton is never mutated, so concurrent jobs
        # cannot interfere with each other.
        work = AdaptationCopy(base_model)

        n = len(df)
        ref_vals = [float("nan")] * n
        var_vals = [float("nan")] * n

        for key, positions in reference_groups(df):
            heartbeat()  # cancellation checkpoint (no-op outside a job)

            ctx_text = ""
            if corpus is not None:
                try:
                    ctx_text = corpus.previous_text(
                        df["Sent_ID"].iat[positions[0]], 1,
                        ref_index=key if has_index else None)
                except Exception:
                    ctx_text = ""

            # No preceding sentence: nothing to adapt to, so score with the
            # base model itself — no copy, no SGD step.
            model = base_model
            if ctx_text and ctx_text.strip():
                model = work.fresh()
                adapt_one_step(ctx_text, model, word2idx, device, lr=_ADAPT_LR)

            # Reference and variants in one engine call; the reference is
            # scored once and broadcast.
            ref_score, var_scores = score_group(
                df, positions, model, device, encode, pad_id=pad_id,
                on_chunk=heartbeat)
            for i, v in zip(positions, var_scores):
                ref_vals[i] = ref_score
                var_vals[i] = v

        df["AdaptiveWiki50M_Reference"] = ref_vals
        df["AdaptiveWiki50M_Variant"] = var_vals
        return df

    def deltas(self):
        return [("Delta_AdaptiveWiki50M",
                 lambda row: row["AdaptiveWiki50M_Reference"],
                 lambda row: row["AdaptiveWiki50M_Variant"])]
