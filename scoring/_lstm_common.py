"""
scoring._lstm_common
====================
Shared LSTM language-model machinery for the ``lstm`` and ``adaptive_lstm``
scorers.

This is a *private* support module: it defines no :class:`~scoring.base.Scorer`
subclass, so the package auto-discovery in ``scoring/__init__.py`` imports it
harmlessly and registers nothing.  The heavy work — importing ``torch`` and
loading the 65 MB checkpoint — is deferred to first use so plugin discovery
(``get_scorers`` / ``/api/plugins``) stays cheap.

Adapted from the Hindi Word Order Explorer
(``feature_extraction/lstm_features.py``, ``adaptive_features.py``,
``models/lstm/model.py``).

The scorers score through the batched engine (``scoring/_lstm_engine.py``)
with :func:`encode_ids`; :func:`sentence_lstm_surprisal` is the original
one-sentence-at-a-time implementation, kept as the parity reference the engine
is tested against.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from threading import Lock

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_MODEL_PATH = _MODELS_DIR / "base_model.pt"
_VOCAB_PATH = _MODELS_DIR / "vocab.pkl"

_cache = None  # (model, word2idx, device) — populated on first get_lstm()
_lock = Lock()


def _build_model_class():
    """Define the LSTM architecture lazily (so importing this module is cheap).

    Embedding(vocab, 256) -> 2-layer LSTM(256, dropout 0.3) -> Linear(256, vocab).
    Must match the architecture the checkpoint was trained with, or
    ``load_state_dict`` will fail.
    """
    import torch.nn as nn

    class LSTMLanguageModel(nn.Module):
        def __init__(self, vocab_size, embed_size=256, hidden_size=256,
                     num_layers=2, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(
                embed_size, hidden_size, num_layers=num_layers,
                dropout=dropout, batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None):
            x = self.embedding(x)
            out, hidden = self.lstm(x, hidden)
            return self.fc(out), hidden

    return LSTMLanguageModel


def _load():
    import torch

    from ._lstm_engine import configure_torch_threads

    configure_torch_threads()
    LSTMLanguageModel = _build_model_class()
    with open(_VOCAB_PATH, "rb") as fh:
        raw_vocab = pickle.load(fh)
    word2idx = raw_vocab["word2idx"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMLanguageModel(len(word2idx))
    state = torch.load(_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, word2idx, device


def get_lstm():
    """Lazy, thread-safe singleton: ``(base_model, word2idx, device)``.

    The base model is shared and used read-only for plain-LSTM scoring; the
    adaptive scorer adapts a working copy of it
    (``_lstm_engine.AdaptationCopy``), so the shared instance is never modified.
    """
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _load()
    return _cache


def encode_ids(sentence: str, word2idx) -> list:
    """Token ids exactly as :func:`sentence_lstm_surprisal` reads them: bare
    whitespace tokens, no ``<SOS>``/``<EOS>`` framing, unknown words -> ``<UNK>``.
    Kept identical so the batched scores reproduce the committed ``Delta_LSTM``
    weight's inputs."""
    unk = word2idx.get("<UNK>", 0)
    return [word2idx.get(w, unk) for w in str(sentence).split()]


def sentence_lstm_surprisal(sentence: str, model, word2idx, device) -> float:
    """Total next-word surprisal of *sentence* in nats: sum of ``-log_softmax``
    log-probabilities of each word given its left context.  Returns 0.0 for
    sentences shorter than two tokens (no next-word prediction possible)."""
    import torch

    words = sentence.split()
    if len(words) < 2:
        return 0.0

    unk = word2idx.get("<UNK>", 0)
    indices = [word2idx.get(w, unk) for w in words]
    input_tensor = torch.tensor(indices[:-1]).unsqueeze(0).to(device)
    target_tensor = torch.tensor(indices[1:]).to(device)

    with torch.no_grad():
        logits, _ = model(input_tensor)
        log_probs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for i, target in enumerate(target_tensor):
            total += -log_probs[0, i, target].item()
    return total


def adapt_one_step(sentence: str, model, word2idx, device, lr: float = 0.01,
                   *, deterministic: bool = True) -> None:
    """Run one SGD step (cross-entropy next-word loss) on *sentence* to adapt
    *model* in place (van Schijndel & Linzen 2018).  Skipped for sentences with
    fewer than two tokens.  Leaves the model in ``eval`` mode for the caller.

    ``deterministic`` (default True) zeroes the LSTM's dropout for the step.
    Until 2026-10-04 the step ran with dropout p=0.3 live, so the adapted
    weights — and every ``Delta_Adaptive`` value — depended on the RNG state and
    changed run to run.  Dropout is a training-time regulariser with no place in
    a single adaptation step; ``_lstm_wiki50m.adapt_one_step`` measured turning
    it off at no accuracy cost.  It is zeroed rather than switched off with
    ``model.eval()`` because cuDNN refuses an LSTM backward pass in eval mode.
    """
    from ._lstm_engine import sgd_step

    ids = encode_ids(sentence, word2idx)
    if len(ids) < 2:
        return
    # Same update as torch.optim.SGD over model.parameters(), with the
    # embedding written sparsely (only the context's rows) — see sgd_step.
    sgd_step(model, ids, device, lr, deterministic=deterministic)
