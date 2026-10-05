from celery import Celery

from backend.config import settings

celery_app = Celery(
    "pythagoras",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.tasks"],
)


@celery_app.task(name="workers.tasks.healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "workers"}


@celery_app.task(name="workers.tasks.analyze_chords")
def analyze_chords(audio_path: str, key: str = "C", mode: str = "major") -> list[dict]:
    """Run deterministic chord recognition on an audio file."""
    from workers.analysis.chords import recognize_chords

    return recognize_chords(audio_path, key=key, mode=mode)


@celery_app.task(name="workers.tasks.analyze_vocals")
def analyze_vocals(vocal_stem_path: str) -> dict:
    """Run vocal production forensics on a vocal stem (or full mix)."""
    from workers.analysis.vocals import analyze_vocals as _analyze

    return _analyze(vocal_stem_path)


@celery_app.task(name="workers.tasks.compute_waveform")
def compute_waveform(audio_path: str, n_peaks: int = 600) -> list[float]:
    """Compute the dashboard waveform peak envelope for an audio file."""
    from workers.analysis.waveform import waveform_peaks

    return waveform_peaks(audio_path, n_peaks=n_peaks)


@celery_app.task(name="workers.tasks.analyze_track", bind=True)
def analyze_track(self, job_id: str, track_id: str, audio_path: str) -> dict:
    """Full analysis pipeline as a Celery task (production path).

    Runs the same run_pipeline() the API uses via BackgroundTasks. Note: in
    a multi-process Celery deployment the in-memory job registry in
    backend/jobs.py is per-process — back it with Redis before relying on
    cross-process job status.
    """
    from workers.pipeline import run_pipeline

    return run_pipeline(job_id, track_id, audio_path)
