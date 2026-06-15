"""
webapp.jobs
===========
In-memory job store with a small worker pool.

Jobs are kept in a dict and their artifacts on disk under
``webapp/outputs/<job_id>/``.  Heavy pipeline work runs on a
ThreadPoolExecutor so uploads return immediately and at most MAX_WORKERS
pipelines run concurrently; further jobs wait in the executor queue.
Finished jobs older than JOB_TTL_SECONDS are swept on each new submission.
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

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
JOB_TTL_SECONDS = 3600
MAX_WORKERS = 2

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_jobs: Dict[str, "Job"] = {}
_lock = threading.Lock()


@dataclass
class Job:
    job_id: str
    status: str = "queued"          # queued | running | done | failed
    stage: str = ""                 # parse | filter | variants | complete
    error: str = ""
    artifacts: List[str] = field(default_factory=list)
    summary: Optional[dict] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "artifacts": list(self.artifacts),
            "summary": self.summary,
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
        job.status = "running"
        try:
            fn(*args)
            job.status = "done"
        except Exception as exc:  # surfaced to the user via the status API
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"

    _executor.submit(_run)


def _cleanup_expired() -> None:
    now = time.time()
    with _lock:
        expired = [
            jid
            for jid, job in _jobs.items()
            if now - job.created_at > JOB_TTL_SECONDS
            and job.status in ("done", "failed")
        ]
        for jid in expired:
            _jobs.pop(jid)
    for jid in expired:
        shutil.rmtree(job_dir(jid), ignore_errors=True)
