"""
scoring._katz_trigram
=====================
Shared machinery for the Katz-backoff trigram model.  Private, like
``_lstm_common``: no ``Scorer`` subclass lives here, so the plugin discovery in
``scoring/__init__.py`` ignores it.  The scorer itself is
``scoring/katz_trigram_scorer.py``.

Ported verbatim from the audit pipeline at
``D:\\audit\\trigram\\run_trigram_pipeline.py`` (``normalize`` :201, ``TrigramLM``
:710).  **Keep it verbatim.**  Two details are load-bearing and easy to "tidy"
into a silent regression:

* ``normalize`` must stay byte-identical to the cleaning applied to the training
  text.  The model was trained on Wikipedia text whose sentences were split on
  ``[।!?]``, so a danda never survives into training data — but a treebank
  sentence still ends in one, and ~91% of them do.  Feeding it through
  unnormalized makes the final token out-of-vocabulary every single time.
* Scoring pads with ``<s> <s>`` and scores ``</s>``.  The model has real
  probabilities for those positions, so skipping them (as the older
  ``trigram_scorer`` does) throws away the boundary statistics the model was
  trained with, and leaves the first two words of every sentence contributing
  nothing at all.

Surprisal here is in **bits** (log2), unlike ``trigram_scorer``'s nats.

The model is 10 files of about 148 MB in ``scoring/models/trigram_katz/``, and
is memory-mapped: opening it is effectively free (well under a second) and costs
almost no resident RAM, in contrast to the 226 MB pickle the older scorer
unpickles into multi-GB dictionaries over roughly a minute and a half.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models" / "trigram_katz"

# Special tokens.  <s> is context-only and is never predicted.
UNK, BOS, EOS = "<UNK>", "<s>", "</s>"

# The danda U+0964 IS inside this Devanagari block, but sentence splitting
# consumes it as a delimiter, so it never survives into training text.
_DEVANAGARI_ONLY = re.compile(r"[^\u0900-\u097F\s,\-]")
_SENTENCE_SPLIT = re.compile(r"[।!?]")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Clean Hindi text down to the token conventions the LM was trained on.

    Applied to Wikipedia text at training time AND to treebank sentences at
    scoring time, so the model is never asked about tokens it could not have
    been trained on.  Returns a whitespace-joined string (possibly empty).
    """
    text = _SENTENCE_SPLIT.sub(" ", text)
    text = _DEVANAGARI_ONLY.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


class TrigramLM:
    """Katz-backoff trigram model over Good-Turing discounted counts.

    Loads by memory-mapping, so opening it is instant and costs almost no RAM.
    """

    def __init__(self, mdl: Path):
        self.meta = json.loads((mdl / "meta.json").read_text(encoding="utf-8"))
        vocab = json.loads((mdl / "vocab.json").read_text(encoding="utf-8"))
        self.word2id = vocab["word2id"]
        self.V = self.meta["V"]
        self.unk = self.word2id[UNK]
        self.bos = self.word2id[BOS]
        self.eos = self.word2id[EOS]

        def m(name):
            return np.load(mdl / name, mmap_mode="r")

        self.uni_logp = m("uni_logp.npy")
        self.bi_key = m("bi_key.npy")
        self.bi_logp = m("bi_logp.npy")
        self.bi_alpha_log = m("bi_alpha_log.npy")
        self.tri_key = m("tri_key.npy")
        self.tri_logp = m("tri_logp.npy")
        self.tri_ctx_key = m("tri_ctx_key.npy")
        self.tri_alpha_log = m("tri_alpha_log.npy")

    # -- lookups -------------------------------------------------------------
    @staticmethod
    def _find(keys, q):
        i = np.searchsorted(keys, q)
        return int(i) if i < keys.size and keys[i] == q else -1

    def logp_level(self, w1, w2, w3):
        """log2 P(w3 | w1, w2) and which level answered ('3gram'/'2gram'/'1gram')."""
        i = self._find(self.tri_key, w1 * self.V * self.V + w2 * self.V + w3)
        if i >= 0:
            return float(self.tri_logp[i]), "3gram"

        # back off to the bigram, paying this context's backoff weight
        c = self._find(self.tri_ctx_key, w1 * self.V + w2)
        a3 = float(self.tri_alpha_log[c]) if c >= 0 else 0.0

        j = self._find(self.bi_key, w2 * self.V + w3)
        if j >= 0:
            return a3 + float(self.bi_logp[j]), "2gram"

        a2 = float(self.bi_alpha_log[w2])
        return a3 + a2 + float(self.uni_logp[w3]), "1gram"

    def logp(self, w1, w2, w3):
        """log2 P(w3 | w1, w2) as a plain number."""
        return self.logp_level(w1, w2, w3)[0]

    def encode(self, text, already_clean=False):
        toks = (text if already_clean else normalize(text)).split()
        return [self.word2id.get(t, self.unk) for t in toks], toks

    # -- sentence scoring ----------------------------------------------------
    def surprisal(self, text, already_clean=False) -> float:
        """Total surprisal in bits.  Every word is scored, including the first
        two, via <s> <s> padding."""
        ids, _ = self.encode(text, already_clean)
        if not ids:
            return 0.0
        seq = [self.bos, self.bos] + ids + [self.eos]
        return -sum(self.logp(seq[i - 2], seq[i - 1], seq[i])
                    for i in range(2, len(seq)))

    def per_word(self, text, already_clean=False):
        """Per-word breakdown: [(word, context, level, log2p, surprisal), ...]"""
        ids, toks = self.encode(text, already_clean)
        if not ids:
            return []
        seq = [self.bos, self.bos] + ids + [self.eos]
        shown = [BOS, BOS] + toks + [EOS]
        rows = []
        for i in range(2, len(seq)):
            lp, level = self.logp_level(seq[i - 2], seq[i - 1], seq[i])
            surface = shown[i]
            if i - 2 < len(toks) and seq[i] == self.unk and toks[i - 2] != UNK:
                surface = f"{toks[i - 2]}[UNK]"
            rows.append((surface, f"{shown[i - 2]} {shown[i - 1]}",
                         level, lp, -lp))
        return rows


_lm = None
_lock = Lock()


def get_lm() -> TrigramLM:
    """Lazy, thread-safe load of the memory-mapped model.

    Double-checked locking, matching the other scorers' loaders, so concurrent
    jobs pay the (already tiny) open cost at most once.
    """
    global _lm
    if _lm is None:
        with _lock:
            if _lm is None:
                _lm = TrigramLM(MODEL_DIR)
    return _lm
