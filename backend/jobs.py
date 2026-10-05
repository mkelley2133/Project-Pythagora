"""In-memory job registry for the analysis pipeline.

Tracks the lifecycle of each analysis job: queued -> running -> done/failed,
with stage labels and 0-100 progress for the dashboard's progress UI.

Note: this registry lives in the API process. It is the right store for the
starter single-process deployment (BackgroundTasks). A multi-worker Celery
deployment should back this with Redis instead.
"""
from __future__ import annotations

import time
from uuid import uuid4

_jobs: dict[str, dict] = {}


def create_job(track_id: str) -> dict:
    job_id = f"job-{uuid4().hex[:12]}"
    job = {
        "job_id": job_id,
        "track_id": track_id,
        "state": "queued",  # queued | running | done | failed
        "stage": "Queued",
        "progress": 0,
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _jobs[job_id] = job
    return job


def update_job(job_id: str, **fields) -> dict | None:
    job = _jobs.get(job_id)
    if job is None:
        return None
    job.update(fields)
    job["updated_at"] = time.time()
    return job


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)
