"""
helpers.cancellation
====================
Cooperative cancellation for long pipeline runs.

Python threads cannot be killed from the outside, so a run is stopped by
agreement: the worker calls :func:`heartbeat` at frequent checkpoints, and a
cancel request makes the next heartbeat raise :class:`JobCancelled`, unwinding
the call stack.  Work already written to disk is left untouched.

Two moving parts:

- **The killer** — :meth:`CancelToken.cancel`.  Flips the token and terminates
  every external child process registered against it (the Berkeley JVM, and any
  future shell-out), which is the one thing a cooperative flag alone cannot
  interrupt.
- **The heartbeat** — :meth:`CancelToken.beat`, reached from library code via
  the module-level :func:`heartbeat`.  Records liveness and raises once the
  token is cancelled.

The current token is bound to the worker *thread*, so scorers do not need a new
argument threaded through every signature::

    token = CancelToken()
    bind(token)
    try:
        ...                       # heartbeat() anywhere below here honours it
    finally:
        unbind()

Outside the web app nothing binds a token, so :func:`heartbeat` and
:func:`register_process` are no-ops — the library, notebooks and the test suite
behave exactly as before.

Depends only on the standard library; safe to import from any logic package.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional, Set

__all__ = [
    "JobCancelled",
    "CancelToken",
    "bind",
    "unbind",
    "current",
    "heartbeat",
    "register_process",
]

# Seconds to let a child process exit after terminate() before killing it.
_TERM_GRACE = 2.0


class JobCancelled(RuntimeError):
    """Raised by a heartbeat once its token has been cancelled."""


class CancelToken:
    """A cancellation flag plus the child processes that must die with it.

    Thread-safe: ``cancel()`` runs on the request thread while ``beat()`` and
    the process registration run on the worker thread.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._procs: Set[object] = set()
        self._lock = threading.Lock()
        self.last_beat: float = time.time()

    # ── state ────────────────────────────────────────────────────────────
    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    # ── the killer ───────────────────────────────────────────────────────
    def cancel(self) -> None:
        """Flip the token and kill every registered child process.

        Idempotent — calling it twice is harmless, and it never raises, so a
        stubborn child cannot break the cancel endpoint.
        """
        self._event.set()
        self._kill_children()

    def _kill_children(self) -> None:
        with self._lock:
            procs = list(self._procs)
            self._procs.clear()
        for proc in procs:
            try:
                if proc.poll() is not None:      # already exited
                    continue
                proc.terminate()
                try:
                    proc.wait(timeout=_TERM_GRACE)
                except Exception:
                    proc.kill()
            except Exception:
                pass                             # best effort; never propagate

    # ── the heartbeat ────────────────────────────────────────────────────
    def beat(self) -> None:
        """Checkpoint: record liveness, and abort the run if cancelled.

        Cheap enough to call once per DataFrame row.
        """
        self.last_beat = time.time()
        if self._event.is_set():
            # A child may have been registered between cancel() and now.
            self._kill_children()
            raise JobCancelled("Job cancelled by the user.")

    # ── child processes ──────────────────────────────────────────────────
    def register(self, proc: object) -> None:
        """Track a ``subprocess.Popen`` so the killer can terminate it.

        Registering after a cancel kills the process immediately, so a race
        with the cancel endpoint cannot leave an orphan running.
        """
        with self._lock:
            self._procs.add(proc)
        if self._event.is_set():
            self._kill_children()

    def unregister(self, proc: object) -> None:
        with self._lock:
            self._procs.discard(proc)


# ── thread-local binding ─────────────────────────────────────────────────
_local = threading.local()


def bind(token: CancelToken) -> None:
    """Make *token* the cancellation token for the calling thread."""
    _local.token = token


def unbind() -> None:
    """Drop this thread's token (heartbeats become no-ops again)."""
    _local.token = None


def current() -> Optional[CancelToken]:
    """This thread's token, or ``None`` outside a cancellable job."""
    return getattr(_local, "token", None)


def heartbeat() -> None:
    """Checkpoint for long-running library code.

    Raises :class:`JobCancelled` if the current job has been cancelled; a no-op
    when no token is bound, so library callers are unaffected.
    """
    token = current()
    if token is not None:
        token.beat()


@contextmanager
def register_process(proc: object) -> Iterator[object]:
    """Register *proc* with the current token for the duration of the block.

    A no-op (but still yields *proc*) outside a cancellable job::

        with subprocess.Popen(...) as proc, register_process(proc):
            out, err = proc.communicate(timeout=300)
    """
    token = current()
    if token is None:
        yield proc
        return
    token.register(proc)
    try:
        yield proc
    finally:
        token.unregister(proc)
