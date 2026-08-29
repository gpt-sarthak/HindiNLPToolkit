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


def create_job() -> Job:
    _cleanup_expired()
    job = Job(job_id=uuid.uuid4().hex[:12])
    job_dir(job.job_id).mkdir(parents=True, exist_ok=True)
    with _lock:
        _jobs[job.job_id] = job
    return job


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
