"""
webapp.warmup
=============
Background pre-warming of the heavy lazily-loaded models, so the *first* user
job after a fresh start is fast (cold container: ~223 s for the first job vs
~10 s warm — dominated by the 226 MB trigram unpickle and Stanza init).

Runs on a daemon thread started from the FastAPI lifespan hook, so the server
binds and serves immediately while models load behind it.  Targets are warmed
cheap-first (Stanza → LSTM → trigram) and each one just calls the model's own
lazy, thread-safe singleton loader — a job that races the warm-up blocks on
the same lock instead of loading twice.

Set ``HNT_NO_PREWARM=1`` to skip warming entirely (light local dev server).

``berkeley_pcfg`` / ``surprisal`` are not warmed: they shell out to the
Berkeley jar per call, so there is nothing to keep in memory.
"""

from __future__ import annotations

import logging
import os
import threading
import time

# uvicorn's error logger is the one whose output reaches the console/container
# logs; plain module loggers have no handler under uvicorn's default config.
logger = logging.getLogger("uvicorn.error")


def _warm_stanza() -> None:
    from stanza_parser.parser import _get_stanza_pipeline

    _get_stanza_pipeline("hi")


def _warm_lstm() -> None:
    from scoring._lstm_common import get_lstm  # also covers adaptive_lstm

    get_lstm()


def _warm_trigram() -> None:
    from scoring.trigram_scorer import _get_model

    _get_model()


# Cheap first: Stanza serves Sentence-mode parses immediately, the LSTM loads
# in seconds, and the multi-minute trigram unpickle goes last.
_TARGETS = [
    ("stanza", _warm_stanza),
    ("lstm", _warm_lstm),
    ("trigram", _warm_trigram),
]


def warm_models() -> None:
    """Load every warm target in order; a failure (e.g. gitignored model files
    absent locally) is logged and skipped, never raised."""
    for name, load in _TARGETS:
        start = time.perf_counter()
        try:
            load()
            logger.info("warmup: %s ready in %.1fs", name, time.perf_counter() - start)
        except Exception as exc:
            logger.warning("warmup: %s skipped (%s)", name, exc)
    logger.info("warmup: done")


def start_background_warmup() -> threading.Thread | None:
    """Kick off :func:`warm_models` on a daemon thread and return it, or return
    ``None`` without warming when ``HNT_NO_PREWARM`` is set."""
    if os.environ.get("HNT_NO_PREWARM"):
        logger.info("warmup: disabled (HNT_NO_PREWARM set)")
        return None
    thread = threading.Thread(target=warm_models, name="model-warmup", daemon=True)
    thread.start()
    return thread
