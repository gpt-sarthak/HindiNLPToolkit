"""
webapp.ratelimit
================
Per-route request limiting for the web front end.

Web-only concern: imported by ``webapp.app`` / ``webapp.taru_routes`` and never
by the logic packages, per the dependency rule in CLAUDE.md.  (Contrast
``helpers/cancellation.py``, which lives a layer down precisely because
``scoring/`` calls it.)

Two independent tiers are checked on every guarded route:

* **session** -- ``sha256(client_ip + session_cookie)``.  The tight, fair-use
  limit a real user feels.  One browser profile = one bucket, so a whole lab
  behind a single NAT address does not share one budget.
* **ip** -- the client address alone.  Looser, and a cookie reset cannot escape
  it: clearing cookies (or opening a private window) buys a fresh *session*
  bucket but still runs into the address ceiling.

A request is admitted only when **both** tiers have budget, and admission is
all-or-nothing (see ``_admit``) so a request rejected by the IP tier does not
still burn a slot of the session tier's budget.

Only expensive endpoints opt in, as an explicit ``Depends`` -- deliberately not
middleware.  ``GET /api/jobs/{job_id}`` is polled every 1.5 s by the SPA for the
whole life of a job (~40 req/min, for 15+ minutes on a big corpus), so a blanket
limiter would break normal use on the very first run.  A per-route dependency
cannot catch the poller by accident.

Known limitation of that choice: FastAPI parses the request body *before* it
solves dependencies, so a throttled ``POST /api/jobs`` has already transferred
its whole upload by the time this module rejects it (measured: 48 MB on a 429).
CPU, workers and disk are protected; bandwidth is not.  Fixing it means moving
the jobs rate check into the ASGI middleware next to ``MaxBodySizeMiddleware``
-- deferred, with the full write-up under "a throttled upload is still
transferred in full" in CLAUDE.md.

Job *concurrency* ("N runs in flight") is not tracked here -- that lives in
``webapp.jobs.try_create_job``, which folds the check into the same lock that
registers the job, so check-then-act cannot race.  This module meters rate only.

In-memory and single-process, which is correct for the one-uvicorn-process
container this app ships as.  More than one worker or instance would need a
shared backend (Redis or similar).
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
import uuid
from collections import deque
from http.cookies import SimpleCookie
from typing import Callable, Deque, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request

# ---- Limits (requests per 60 s, per tier) --------------------------
JOBS_SESSION_PER_MIN = 2
JOBS_IP_PER_MIN = 10
TARU_PARSE_SESSION_PER_MIN = 6
TARU_PARSE_IP_PER_MIN = 20
# Tighter than job creation: /taru/train spawns a raw daemon thread with no
# worker pool behind it, so its rate cap is the only bound on concurrency.
TARU_TRAIN_SESSION_PER_MIN = 2
TARU_TRAIN_IP_PER_MIN = 5

# ---- Concurrent in-flight jobs (enforced in webapp.jobs) -----------
JOBS_SESSION_INFLIGHT = 1
JOBS_IP_INFLIGHT = 3

SESSION_COOKIE = "hnt_sid"
SESSION_MAX_AGE = 86400          # 1 day, comfortably longer than any session
WINDOW_SECONDS = 60.0
_SWEEP_INTERVAL = 300.0          # idle-bucket eviction cadence

# Rendered verbatim to the user by index.html (`data.detail` -> #form-error),
# so these are written for a reader, not a developer.
RATE_MESSAGE = (
    "You are sending requests faster than this server allows. "
    "Please wait about {retry} seconds and try again."
)
BUSY_MESSAGE = (
    "You already have a run in progress. Wait for it to finish, or press "
    "Cancel run, then submit again."
)

_lock = threading.Lock()
_buckets: Dict[Tuple[str, str], Deque[float]] = {}
_last_sweep = time.monotonic()


def enabled() -> bool:
    """Whether limiting is active.

    Read from the environment on *every* call rather than frozen at import, so
    a test (or a local dev session) can flip it without reloading the module.
    This diverges from the ``HNT_NO_PREWARM`` convention in webapp/warmup.py,
    which is consulted once at startup; the difference is deliberate.
    """
    return os.environ.get("HNT_NO_RATELIMIT", "") != "1"


def client_ip(request: Request) -> str:
    """The caller's address.

    ``request.client.host`` is the true client today: the container publishes
    port 80 directly, with no reverse proxy in front.  Set ``HNT_TRUSTED_PROXY=1``
    only once something trusted *is* terminating traffic -- and note we take the
    **last** hop of X-Forwarded-For, never the first.  That header is
    client-supplied, so a caller can prepend anything they like; only the entry
    appended by your own edge is beyond their control.  Trusting the first hop
    would hand every request a fresh IP bucket and void the tier entirely.
    """
    if os.environ.get("HNT_TRUSTED_PROXY", "") == "1":
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


_SCOPE_KEY = "hnt_session_id"


class SessionMiddleware:
    """Give every caller a stable opaque session id, in a first-party cookie.

    This has to be middleware rather than part of the ``guard`` dependency, and
    the reason is easy to get wrong: a dependency *can* set a cookie on the
    injected ``Response``, but FastAPI throws that object away and builds a
    fresh response whenever the handler raises ``HTTPException``.  Every error
    reply — including the limiter's own 429 — would therefore arrive with no
    cookie, the client would mint a new session on its next attempt, and the
    session tier would reset on each retry, silently degrading to IP-only.

    It sets a cookie and nothing else: no counting, no rejection.  So unlike a
    limiter, it is safe to run on every request, and the status poller is
    untouched beyond one header on its first reply.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not enabled():
            await self.app(scope, receive, send)
            return

        sid = _cookie_from_scope(scope, SESSION_COOKIE)
        if sid:
            scope[_SCOPE_KEY] = sid
            await self.app(scope, receive, send)
            return

        sid = uuid.uuid4().hex
        scope[_SCOPE_KEY] = sid
        jar = SimpleCookie()
        jar[SESSION_COOKIE] = sid
        morsel = jar[SESSION_COOKIE]
        morsel["path"] = "/"
        morsel["max-age"] = SESSION_MAX_AGE
        morsel["httponly"] = True
        morsel["samesite"] = "Lax"
        # No Secure flag: the deployment is plain HTTP on port 80 today, and a
        # Secure cookie would simply never be stored.  Set it once HTTPS lands.
        header = morsel.OutputString().encode("latin-1")

        async def send_with_cookie(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"set-cookie", header)
                ]
            await send(message)

        await self.app(scope, receive, send_with_cookie)


def _cookie_from_scope(scope, name: str) -> str:
    """Read one cookie straight off the raw ASGI headers."""
    for key, value in scope.get("headers", []):
        if key == b"cookie":
            jar = SimpleCookie()
            try:
                jar.load(value.decode("latin-1"))
            except Exception:
                return ""
            morsel = jar.get(name)
            return morsel.value if morsel else ""
    return ""


def _session_id(request: Request) -> str:
    """The caller's session id, as placed on the scope by SessionMiddleware."""
    return request.scope.get(_SCOPE_KEY) or request.cookies.get(SESSION_COOKIE, "")


def session_key(request: Request) -> str:
    """Hashed (ip, session) pair.  Hashing keeps raw addresses out of the store."""
    raw = f"{client_ip(request)}|{_session_id(request)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _maybe_sweep(now: float) -> None:
    """Drop idle buckets.

    Session keys are one-shot UUIDs and never recur, so without this the store
    grows by one entry per visitor forever.  O(buckets), run inline on a request
    but throttled to once per _SWEEP_INTERVAL.  Caller must hold ``_lock``.
    """
    global _last_sweep
    if now - _last_sweep < _SWEEP_INTERVAL:
        return
    _last_sweep = now
    cutoff = now - WINDOW_SECONDS
    for key in [k for k, dq in _buckets.items() if not dq or dq[-1] < cutoff]:
        del _buckets[key]


def _admit(entries: List[Tuple[str, str, int]]) -> Optional[float]:
    """Check every ``(bucket, key, limit)`` and record a hit in each, or in none.

    Returns ``None`` when admitted, else the seconds until the tightest bucket
    frees a slot.  One lock acquisition covers check *and* mutate: splitting
    them would let two concurrent requests both observe room and both take the
    last slot.
    """
    now = time.monotonic()
    with _lock:
        _maybe_sweep(now)
        cutoff = now - WINDOW_SECONDS
        queues: List[Deque[float]] = []
        retry_after = 0.0
        for bucket, key, limit in entries:
            dq = _buckets.setdefault((bucket, key), deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            queues.append(dq)
            if len(dq) >= limit:
                retry_after = max(retry_after, dq[0] + WINDOW_SECONDS - now)
        if retry_after > 0:
            return retry_after
        for dq in queues:
            dq.append(now)
        return None


def guard(name: str, *, session_limit: int, ip_limit: int) -> Callable:
    """Build the FastAPI dependency that rate-limits one route.

    Returns the ``(session_key, ip_key)`` pair so a route that also needs
    caller identity (``create_job``, for its in-flight check) reuses it instead
    of re-deriving it.  The session cookie itself is handled by
    SessionMiddleware, not here.
    """

    def _dependency(request: Request) -> Tuple[str, str]:
        if not enabled():
            return "", ""
        ip_key = client_ip(request)
        sess_key = session_key(request)
        retry_after = _admit(
            [
                (f"{name}:session", sess_key, session_limit),
                (f"{name}:ip", ip_key, ip_limit),
            ]
        )
        if retry_after is not None:
            seconds = max(1, math.ceil(retry_after))
            raise HTTPException(
                429,
                RATE_MESSAGE.format(retry=seconds),
                headers={"Retry-After": str(seconds)},
            )
        return sess_key, ip_key

    return _dependency


def reset() -> None:
    """Clear every counter.  For tests -- nothing in the app calls this."""
    global _last_sweep
    with _lock:
        _buckets.clear()
        _last_sweep = time.monotonic()
