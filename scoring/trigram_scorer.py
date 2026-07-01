"""
scoring.trigram_scorer
======================
Trigram language-model surprisal scorer.

Backed by a pickled NLTK MLE trigram model (``scoring/models/trigram.pkl``)
trained on Hindi text.  For each (reference, variant) pair it computes the total
sentence surprisal of each word order and declares ``Delta_Trigram``, oriented
centrally by ``ML_Label``.

Smoothing
---------
The MLE model returns exactly 0.0 for unseen ngrams (no built-in smoothing), so
per-word probability uses a three-level backoff:

    1. Trigram  P(w3 | w1, w2)
    2. Bigram   P(w3 | w2)        if the trigram is unseen
    3. Unigram  P(w3)             if the bigram is also unseen
    4. Epsilon  1e-12             if the word is fully out-of-vocabulary

Sentence surprisal = sum of per-word ``-ln P`` over words with full trigram
context (Ranjan & van Schijndel 2024).

Discovered automatically by the scoring package — appears as the ``trigram``
checkbox in the UI.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from threading import Lock

from .base import Scorer

_MODEL_PATH = Path(__file__).resolve().parent / "models" / "trigram.pkl"

_model = None
_lock = Lock()


def _get_model():
    """Lazy, thread-safe load of the 226 MB pickled MLE trigram model.
    ``import nltk`` first so the classes the pickle references are registered
    (and so a missing dependency fails with a clear error)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import nltk  # noqa: F401  (registers nltk.lm classes for unpickling)
                with open(_MODEL_PATH, "rb") as fh:
                    _model = pickle.load(fh)
    return _model


def _trigram_prob(model, w1, w2, w3) -> float:
    """P(w3 | w1, w2) with trigram->bigram->unigram backoff; never returns 0
    (minimum is the 1e-12 OOV epsilon)."""
    prob = model.score(w3, [w1, w2])
    if prob > 0:
        return prob
    prob = model.score(w3, [w2])
    if prob > 0:
        return prob
    prob = model.score(w3)
    if prob > 0:
        return prob
    return 1e-12


def _sentence_surprisal(sentence: str, model) -> float:
    """Total trigram surprisal (nats) = sum of -ln P over words with full
    trigram context.  Sentences shorter than three tokens score 0.0."""
    words = sentence.split()
    if len(words) < 3:
        return 0.0
    total = 0.0
    for i in range(2, len(words)):
        total += -math.log(_trigram_prob(model, words[i - 2], words[i - 1], words[i]))
    return total


class TrigramScorer(Scorer):
    name = "trigram"
    description = (
        "Trigram language-model surprisal (nats) of each word order. "
        "Advantage: Delta_Trigram."
    )
    trained_on = "Hindi text corpus"
    built_with = "NLTK MLE trigram model"
    notes = "trigram -> bigram -> unigram backoff smoothing"

    def score(self, pairs_df):
        df = pairs_df.copy()
        if df.empty:
            df["Trigram_Reference"] = []
            df["Trigram_Variant"] = []
            return df

        try:
            model = _get_model()
        except Exception:
            df["Trigram_Reference"] = [float("nan")] * len(df)
            df["Trigram_Variant"] = [float("nan")] * len(df)
            return df

        cache: dict = {}

        def surprisal(text) -> float:
            key = str(text)
            if key not in cache:
                cache[key] = _sentence_surprisal(key, model)
            return cache[key]

        df["Trigram_Reference"] = [surprisal(s) for s in df["Reference_Sentence"]]
        df["Trigram_Variant"] = [surprisal(s) for s in df["Variant_Sentence"]]
        return df

    def deltas(self):
        return [("Delta_Trigram",
                 lambda row: row["Trigram_Reference"],
                 lambda row: row["Trigram_Variant"])]
