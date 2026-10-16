"""
scoring.adaptive_lstm_scorer
============================
Adaptive LSTM surprisal scorer (van Schijndel & Linzen 2018, as applied in
Ranjan & van Schijndel 2024).

The base LSTM is updated by one gradient step on the *preceding* (context)
sentence, then used to score the reference and its variants under the adapted
weights — modelling how discourse context shapes a reader's expectations.

Procedure (per reference sentence — ``helpers.reference_groups``):
    1. If a preceding sentence exists, reset a working copy of the base model
       to the base weights (one copy per ``score`` call, restored in place —
       the shared base is never mutated) and run one SGD step on it.
    2. Score the reference and all its variants together, in eval mode,
       through the batched engine (``scoring/_lstm_engine.py``).

When no preceding sentence is available (first sentence in a document, or no
corpus context), no copy and no SGD step are made: the group is scored with the
base model directly, so the score equals the plain LSTM surprisal.

The SGD step runs with dropout switched off (since 2026-10-04), so the scores
are deterministic; before that, dropout made every run differ — see
``_lstm_common.adapt_one_step``.

This is a *context-aware* scorer: it declares a ``context`` parameter on
``score`` and reads ``context["corpus"]`` (a :class:`helpers.CorpusContext`) for
the preceding-sentence lookup.  ``Delta_Adaptive`` is oriented centrally by
``ML_Label``.

Discovered automatically by the scoring package — appears as the
``adaptive_lstm`` checkbox in the UI.
"""

from __future__ import annotations

from typing import Optional

from helpers import REF_INDEX_COLUMN, heartbeat, reference_groups

from ._lstm_common import adapt_one_step, encode_ids, get_lstm
from ._lstm_engine import AdaptationCopy, score_group
from .base import Scorer


class AdaptiveLSTMScorer(Scorer):
    name = "adaptive_lstm"
    display_name = "Adaptive LSTM"
    description = (
        "Adaptive LSTM surprisal of each word order, after adapting to "
        "the preceding sentence.\n"
        "Feature in CSV: Delta_Adaptive."
    )
    trained_on = "Hindi Wikipedia (dumps.wikimedia.org/hiwiki), base LSTM"
    built_with = "base LSTM + one-step online adaptation"
    needs_previous_sentence = True
    higher_is_more_natural = {"Delta_Adaptive": False}  # lower surprisal = more natural

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
        has_index = REF_INDEX_COLUMN in df.columns
        encode = lambda text: encode_ids(text, word2idx)  # noqa: E731
        work = AdaptationCopy(base_model)

        ref_vals = [float("nan")] * n
        var_vals = [float("nan")] * n

        for key, positions in reference_groups(df):
            heartbeat()      # cancellation checkpoint (no-op outside a job) —
                             # one SGD step per sentence below
            ctx_text = ""
            if corpus is not None:
                try:
                    ctx_text = corpus.previous_text(
                        df["Sent_ID"].iat[positions[0]], 1,
                        ref_index=key if has_index else None)
                except Exception:
                    ctx_text = ""

            model = base_model
            if ctx_text and ctx_text.strip():
                model = work.fresh()
                adapt_one_step(ctx_text, model, word2idx, device)

            ref_s, var_s = score_group(df, positions, model, device, encode,
                                       on_chunk=heartbeat)
            for pos, v in zip(positions, var_s):
                ref_vals[pos] = ref_s
                var_vals[pos] = v

        df["Adaptive_Reference"] = ref_vals
        df["Adaptive_Variant"] = var_vals
        return df

    def deltas(self):
        return [("Delta_Adaptive",
                 lambda row: row["Adaptive_Reference"],
                 lambda row: row["Adaptive_Variant"])]
