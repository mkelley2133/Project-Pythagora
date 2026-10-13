"""In-memory job registry + dispatcher for the analysis pipeline.

Tracks the lifecycle of each analysis job: queued -> running -> done/failed,
with named stages and honest progress. Progress is stage-weighted: it only
advances when a stage actually completes (the active stage contributes
nothing until it finishes).

Dispatch: jobs are executed by a thread pool sized to
`settings.max_concurrent_jobs`. Uploads beyond the cap wait in the pool
queue with state "queued" — visible to the client — instead of being
rejected (batch uploads rely on this).

Note: this registry lives in the API process. It is the right store for
the starter single-process deployment. A multi-worker Celery deployment
should back this with Redis instead.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from backend.config import settings

_jobs: dict[str, dict] = {}

_FINISHED_TTL_SECONDS = 3600
_MAX_FINISHED_KEPT = 200

_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_jobs),
            thread_name_prefix="pythagora-pipeline",
        )
    return _executor


def _prune_finished() -> None:
    """Drop stale finished jobs so the registry cannot grow without bound."""
    now = time.time()
    finished = [
        job_id
        for job_id, job in _jobs.items()
        if job["state"] in ("done", "failed") and now - job["updated_at"] > _FINISHED_TTL_SECONDS
    ]
    for job_id in finished:
        _jobs.pop(job_id, None)
    finished = [j for j in _jobs.values() if j["state"] in ("done", "failed")]
    if len(finished) > _MAX_FINISHED_KEPT:
        finished.sort(key=lambda j: j["updated_at"])
        for job in finished[: len(finished) - _MAX_FINISHED_KEPT]:
            _jobs.pop(job["job_id"], None)


def active_count() -> int:
    """Jobs currently queued or running — used for the concurrency cap."""
    return sum(1 for job in _jobs.values() if job["state"] in ("queued", "running"))


def create_job(track_id: str) -> dict:
    _prune_finished()
    job_id = f"job-{uuid4().hex[:12]}"
    job = {
        "job_id": job_id,
        "track_id": track_id,
        "state": "queued",  # queued | running | done | failed
        "stage": "Queued",
        "progress": 0,
        "error": None,
        "stage_index": 0,
        "stage_total": 0,
        "stages": [],  # [{"name": str, "status": pending|active|done|skipped}]
        "started_at": None,
        "finished_at": None,
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
    if job["state"] in ("done", "failed") and job.get("finished_at") is None:
        job["finished_at"] = time.time()
    return job


def get_job(job_id: str) -> dict | None:
    job = _jobs.get(job_id)
    if job is None:
        return None
    out = dict(job)
    started = out.get("started_at")
    if started is not None:
        end = out.get("finished_at") or time.time()
        out["elapsed_seconds"] = round(max(0.0, end - started), 1)
    else:
        out["elapsed_seconds"] = None
    return out


# ------------------------------------------------------------- stage flow

def set_stages(job_id: str, names: list[str]) -> dict | None:
    """Register the pipeline's stage list (all pending) at job start."""
    job = _jobs.get(job_id)
    if job is None:
        return None
    job["stages"] = [{"name": n, "status": "pending"} for n in names]
    job["stage_total"] = len(names)
    job["stage_index"] = 0
    job["progress"] = 0
    job["updated_at"] = time.time()
    return job


def _recompute_progress(job: dict) -> None:
    """Percent = share of stages fully completed (done or skipped)."""
    stages = job.get("stages") or []
    if not stages:
        return
    completed = sum(1 for s in stages if s["status"] in ("done", "skipped"))
    job["progress"] = int(100 * completed / len(stages))


def advance_stage(job_id: str, name: str) -> dict | None:
    """Mark every stage before `name` done and `name` active."""
    job = _jobs.get(job_id)
    if job is None:
        return None
    stages = job.get("stages") or []
    names = [s["name"] for s in stages]
    if name not in names:
        # Unknown stage (defensive): still surface the label.
        job["stage"] = name
        job["updated_at"] = time.time()
        return job
    idx = names.index(name)
    for i, s in enumerate(stages):
        if i < idx and s["status"] != "skipped":
            s["status"] = "done"
        elif i == idx:
            s["status"] = "active"
    job["stage_index"] = idx
    job["stage"] = name
    if job.get("started_at") is None:
        job["started_at"] = time.time()
    _recompute_progress(job)
    job["updated_at"] = time.time()
    return job


def skip_stage(job_id: str, name: str) -> dict | None:
    """Mark a stage skipped (it counts as completed for progress)."""
    job = _jobs.get(job_id)
    if job is None:
        return None
    for s in job.get("stages") or []:
        if s["name"] == name:
            s["status"] = "skipped"
    _recompute_progress(job)
    job["updated_at"] = time.time()
    return job


def finish_stages(job_id: str) -> dict | None:
    """Mark every remaining stage done (job completed successfully)."""
    job = _jobs.get(job_id)
    if job is None:
        return None
    for s in job.get("stages") or []:
        if s["status"] != "skipped":
            s["status"] = "done"
    _recompute_progress(job)
    job["updated_at"] = time.time()
    return job


# -------------------------------------------------------------- dispatch

def enqueue(job_id: str, track_id: str, audio_path: str) -> None:
    """Queue a pipeline run on the capped executor.

    The job stays visibly "queued" until a worker slot frees up.
    run_pipeline itself records done/failed on the job record; the wrapper
    only guards against the import/call raising before that happens.
    """
    from workers.pipeline import run_pipeline  # deferred: pipeline imports jobs

    def _run() -> None:
        update_job(job_id, state="running")
        try:
            run_pipeline(job_id, track_id, audio_path)
        except Exception:  # noqa: BLE001 - run_pipeline already flagged the job
            pass

    _get_executor().submit(_run)
