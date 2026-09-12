"""
scoring._lstm_wiki50m
=====================
Machinery for the ``lstm_wiki50m`` / ``adaptive_lstm_wiki50m`` scorers.

Private helper: it defines no ``Scorer`` subclass, so plugin discovery skips it
(``scoring/__init__.py`` only registers subclasses whose ``__module__`` matches
the module it is scanning).  Same arrangement as ``_katz_trigram.py``.

Why this exists alongside ``_lstm_common``
------------------------------------------
Two things differ from the older LSTM scorers, and both are deliberate:

1. **A different checkpoint.**  ``_lstm_common.get_lstm()`` is hard-wired to
   ``base_model.pt`` through module-level constants and a single module-global
   ``_cache``; there is no path keying and no second slot.  So this module keeps
   its own double-checked singleton over ``base_model_wiki50m.pt``.  The
   architecture itself is *not* duplicated — ``_build_model_class`` is imported
   from ``_lstm_common``, so the network is defined in exactly one place.

2. **Correct scoring.**  The model was trained on a stream where every sentence
   is ``["<SOS>"] + tokens + ["<EOS>"]``, and on text cleaned by a
   danda/non-Devanagari regex.  The older scorers reproduce neither: they feed
   bare ``sentence.split()``, so word 1 is never predicted, no sentence-end is
   ever scored, and the danda (absent from the 30k vocab) becomes ``<UNK>`` on
   ~91% of treebank sentences.  Restoring both is worth **+5.65 accuracy
   points** on 92,299 pairs; see ``D:\\audit\\lstm\\FINDINGS.md``.

The vocabulary is the **existing** ``vocab.pkl``, unchanged — the rebuilt model
was trained against it deliberately, so the two model generations differ only in
training, not in symbol set.

Units are nats (``torch.log_softmax``), matching ``lstm`` / ``adaptive_lstm``.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence

from ._lstm_common import _build_model_class

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_MODEL_PATH = _MODELS_DIR / "base_model_wiki50m.pt"
_VOCAB_PATH = _MODELS_DIR / "vocab.pkl"

_cache = None  # (model, word2idx, device) — populated on first get_lstm_wiki50m()
_lock = Lock()

# --- training-time text cleaning ---------------------------------------------
# Byte-identical to the regex pair the training corpus was built with, and to
# scoring/_katz_trigram.py:normalize. Keep the three in step or the "aligned"
# behaviour silently stops matching how the model was trained.
_SENTENCE_SPLIT = re.compile(r"[।!?]")
_DEVANAGARI_ONLY = re.compile(r"[^\u0900-\u097F\s,\-]")
_WHITESPACE = re.compile(r"\s+")

PAD, UNK, SOS, EOS = "<PAD>", "<UNK>", "<SOS>", "<EOS>"


def normalize(text: str) -> str:
    """Apply the cleaning the training corpus went through."""
    text = _SENTENCE_SPLIT.sub(" ", text)
    text = _DEVANAGARI_ONLY.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _load():
    import torch

    LSTMLanguageModel = _build_model_class()
    with open(_VOCAB_PATH, "rb") as fh:
        raw_vocab = pickle.load(fh)
    word2idx = raw_vocab["word2idx"] if "word2idx" in raw_vocab else raw_vocab

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMLanguageModel(len(word2idx))
    model.load_state_dict(torch.load(_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, word2idx, device


def get_lstm_wiki50m():
    """Lazy, thread-safe singleton: ``(model, word2idx, device)``.

    Deliberately separate from ``_lstm_common.get_lstm()`` so the two
    checkpoints coexist; the adaptive scorer ``deepcopy``-s this one before
    mutating, exactly as ``adaptive_lstm`` does with the older model.
    """
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _load()
    return _cache


# --- scoring ------------------------------------------------------------------
def _ids(sentence: str, word2idx) -> List[int]:
    """Token ids with the training-time cleaning and the <SOS>/<EOS> framing."""
    unk = word2idx.get(UNK, 0)
    ids = [word2idx.get(w, unk) for w in normalize(sentence).split()]
    return [word2idx[SOS]] + ids + [word2idx[EOS]]


def sentence_surprisal(sentence: str, model, word2idx, device) -> float:
    """Total surprisal of one sentence, in nats. Batch-of-one convenience."""
    return batch_surprisal([sentence], model, word2idx, device, batch_size=1)[0]


def batch_surprisal(sentences: Sequence[str], model, word2idx, device,
                    batch_size: int = 128,
                    on_batch=None) -> List[float]:
    """Total surprisal per sentence, in nats, in the order given.

    Batched because the older scorers run a batch of ONE with a ``.item()``
    sync per token, which is what makes a full corpus pass slow.  Sentences are
    de-duplicated and length-bucketed so padding waste stays small, and a mask
    keeps padded positions out of the sum — the result is the same as scoring
    one at a time, to float32 rounding.

    ``on_batch`` is called once per batch (used for the cancellation heartbeat).
    """
    import torch

    pad = word2idx.get(PAD, 0)
    out: List[float] = [0.0] * len(sentences)
    pending: Dict[str, List[int]] = {}
    for i, s in enumerate(sentences):
        pending.setdefault(str(s), []).append(i)

    keys = sorted(pending, key=lambda k: len(_ids(k, word2idx)))
    for start in range(0, len(keys), batch_size):
        if on_batch is not None:
            on_batch()
        chunk = keys[start:start + batch_size]
        seqs = [_ids(k, word2idx) for k in chunk]

        live = [(k, s) for k, s in zip(chunk, seqs) if len(s) >= 2]
        for k, s in zip(chunk, seqs):
            if len(s) < 2:                    # normalize() emptied the sentence
                for i in pending[k]:
                    out[i] = 0.0
        if not live:
            continue

        width = max(len(s) for _, s in live) - 1
        inp = torch.full((len(live), width), pad, dtype=torch.long)
        tgt = torch.full((len(live), width), pad, dtype=torch.long)
        mask = torch.zeros((len(live), width), dtype=torch.bool)
        for r, (_, s) in enumerate(live):
            n = len(s) - 1
            inp[r, :n] = torch.tensor(s[:-1], dtype=torch.long)
            tgt[r, :n] = torch.tensor(s[1:], dtype=torch.long)
            mask[r, :n] = True

        inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
        with torch.no_grad():
            logits, _ = model(inp)
            logp = torch.log_softmax(logits, dim=-1)
            picked = logp.gather(2, tgt.unsqueeze(2)).squeeze(2)
            # float64 reduction: the reference implementation sums Python floats
            # one .item() at a time, so a float32 reduction here would differ in
            # the 5th decimal purely by summation order.
            totals = (-picked.double() * mask).sum(dim=1).tolist()
        for (k, _), v in zip(live, totals):
            for i in pending[k]:
                out[i] = v
    return out


def adapt_one_step(sentence: str, model, word2idx, device, lr: float = 0.01,
                   *, deterministic: bool = True,
                   clip: Optional[float] = None) -> None:
    """One SGD step of next-word cross-entropy on *sentence*, in place.

    ``deterministic`` defaults to True, unlike the older ``adaptive_lstm``.
    That scorer leaves dropout (p=0.3) active during the gradient step, so its
    adapted weights — and every score downstream — depend on RNG state:
    measured over the full 92,299 pairs, two seeds changed **every single row**,
    by up to 2.11 nats.  Turning it off costs nothing measurable (83.11% vs
    83.08%), so this scorer is reproducible by default.

    ⚠ Determinism is achieved by zeroing the dropout probability, NOT by
    ``model.eval()``: cuDNN raises "cudnn RNN backward can only be called in
    training mode" if an LSTM backward runs under eval.  A CPU-only test does
    not catch that — CPU RNN backward has no such rule.
    """
    import torch
    import torch.nn as nn

    words = normalize(sentence).split()
    if len(words) < 2:
        return
    unk = word2idx.get(UNK, 0)
    ids = ([word2idx[SOS]] + [word2idx.get(w, unk) for w in words]
           + [word2idx[EOS]])

    inp = torch.tensor(ids[:-1]).unsqueeze(0).to(device)
    tgt = torch.tensor(ids[1:]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    saved_dropout = model.lstm.dropout
    if deterministic:
        model.lstm.dropout = 0.0
    model.train()
    optimizer.zero_grad()
    logits, _ = model(inp)
    loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
    loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    model.lstm.dropout = saved_dropout
    model.eval()
