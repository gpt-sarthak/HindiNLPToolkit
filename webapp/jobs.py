"""
webapp.jobs
===========
In-memory job store with a small worker pool.

Jobs are kept in a dict and their artifacts on disk under
``webapp/outputs/<job_id>/``.  Heavy pipeline work runs on a
ThreadPoolExecutor so uploads return immediately and at most MAX_WORKERS
pipelines run concurrently; further jobs wait in the executor queue.
Finished jobs older than JOB_TTL_SECONDS are swept on each new submission.

A running job can be cancelled (``Job.cancel``).  Threads cannot be killed, so
cancellation is cooperative: the pipeline and the scorers call
``helpers.heartbeat()`` at frequent checkpoints and unwind on the next one,
while external child processes registered with the token are terminated
outright.  Artifacts already written stay on disk and downloadable.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from helpers import CancelToken, JobCancelled, bind, unbind

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
JOB_TTL_SECONDS = 3600
MAX_WORKERS = 2

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_jobs: Dict[str, "Job"] = {}
_lock = threading.Lock()


@dataclass
class Job:
    job_id: str
    # Who submitted this, for the in-flight caps in try_create_job.  Both are
    # opaque (the session one is already hashed); neither reaches to_dict(), so
    # nothing about the caller is exposed through the status API.
    session_key: str = ""
    ip_key: str = ""
    status: str = "queued"          # queued | running | done | failed | cancelled
    stage: str = ""                 # parse | filter | variants | complete
    error: str = ""
    artifacts: List[str] = field(default_factory=list)
    summary: Optional[dict] = None
    created_at: float = field(default_factory=time.time)
    token: CancelToken = field(default_factory=CancelToken)

    def cancel(self) -> None:
        """Stop this job for good.

        Marks it cancelled straight away — so the next poll reflects it and the
        client is free to submit again — then fires the killer, which kills any
        registered child process.  The worker thread unwinds on its next
        heartbeat, in the background.  There is no resuming a cancelled job.
        """
        self.status = "cancelled"
        self.token.cancel()

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "artifacts": list(self.artifacts),
            "summary": self.summary,
            # Wall-clock time of the worker's last checkpoint — a liveness
            # signal for the UI while a long stage is running.
            "last_beat": self.token.last_beat,
        }


def job_dir(job_id: str) -> Path:
    return OUTPUT_ROOT / job_id


def create_job(session_key: str = "", ip_key: str = "") -> Job:
    _cleanup_expired()
    job = Job(job_id=uuid.uuid4().hex[:12], session_key=session_key, ip_key=ip_key)
    job_dir(job.job_id).mkdir(parents=True, exist_ok=True)
    with _lock:
        _jobs[job.job_id] = job
    return job


def try_create_job(
    session_key: str,
    ip_key: str,
    *,
    session_max: int,
    ip_max: int,
) -> Optional[Job]:
    """Create a job only if the caller is under both in-flight caps.

    Returns ``None`` when either cap is already met, so the caller can answer
    429 without having created anything.

    The count and the insert share **one** ``_lock`` acquisition on purpose.
    Counting under the lock, releasing, then inserting under it again would
    still be a check-then-act race: two simultaneous submissions from one
    session could both observe zero in flight and both register a job.  Since
    the whole point of the cap is to stop one caller occupying both workers,
    that race would defeat it exactly when it matters.

    A cap of ``0`` or less means unlimited (used when limiting is switched off).
    """
    _cleanup_expired()
    active = ("queued", "running")
    with _lock:
        if session_max > 0:
            in_flight = sum(
                1 for j in _jobs.values()
                if j.session_key == session_key and j.status in active
            )
            if in_flight >= session_max:
                return None
        if ip_max > 0:
            in_flight = sum(
                1 for j in _jobs.values()
                if j.ip_key == ip_key and j.status in active
            )
            if in_flight >= ip_max:
                return None
        job = Job(job_id=uuid.uuid4().hex[:12], session_key=session_key, ip_key=ip_key)
        _jobs[job.job_id] = job
    job_dir(job.job_id).mkdir(parents=True, exist_ok=True)
    return job


def discard(job: Job) -> None:
    """Drop a job that was registered but never submitted.

    Validation can still fail after the job exists (an oversized upload, an
    empty file), and such a job would otherwise sit at ``queued`` forever:
    _cleanup_expired only reaps terminal jobs, so it would count against its
    caller's in-flight cap for the life of the process.
    """
    with _lock:
        _jobs.pop(job.job_id, None)
    shutil.rmtree(job_dir(job.job_id), ignore_errors=True)


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def submit(job: Job, fn, *args) -> None:
    """Run fn(*args) on the worker pool, tracking status on the job."""

    def _run():
        # Cancelled while still queued behind the workers — never start it.
        if job.token.cancelled:
            job.status = "cancelled"
            return
        job.status = "running"
        bind(job.token)          # heartbeats anywhere below now honour this job
        try:
            fn(*args)
            job.status = "done"
        except JobCancelled:     # cooperative stop — not a failure
            job.status = "cancelled"
        except Exception as exc:  # surfaced to the user via the status API
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            unbind()

    _executor.submit(_run)


def _cleanup_expired() -> None:
    now = time.time()
    with _lock:
        expired = [
            jid
            for jid, job in _jobs.items()
            if now - job.created_at > JOB_TTL_SECONDS
            and job.status in ("done", "failed", "cancelled")
        ]
        for jid in expired:
            _jobs.pop(jid)
    for jid in expired:
        shutil.rmtree(job_dir(jid), ignore_errors=True)
