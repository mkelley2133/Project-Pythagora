from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from backend import jobs, store
from backend.models.schemas import (
    JobStatus,
    TrackCreate,
    TrackRecord,
)
from workers.pipeline import run_pipeline

router = APIRouter(prefix="/tracks", tags=["tracks"])

STORAGE = Path("storage")
AUDIO_DIR = STORAGE / "audio"
ARTWORK_DIR = STORAGE / "artwork"
for _d in (AUDIO_DIR, ARTWORK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
ALLOWED_ARTWORK = {".jpg", ".jpeg", ".png", ".webp"}

store.seed_demo()


def _ext_of(filename: str | None, allowed: set[str], kind: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400, detail=f"Unsupported {kind} type: {ext or '?'}"
        )
    return ext


@router.get("", response_model=list[TrackRecord])
def list_tracks() -> list[TrackRecord]:
    return store.list_tracks()


@router.post("", response_model=TrackRecord, status_code=status.HTTP_201_CREATED)
def create_track(track: TrackCreate) -> TrackRecord:
    record = TrackRecord(
        id=f"track-{uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc),
        **track.model_dump(),
    )
    return store.save_track(record)


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
def upload_track(
    background: BackgroundTasks,
    title: str = Form(...),
    artist: str = Form("Unknown Artist"),
    audio: UploadFile = File(...),
    artwork: UploadFile | None = File(None),
) -> dict:
    """Upload audio (+ optional cover art) and kick off the analysis pipeline.

    Returns immediately with the track record and a job id; poll
    GET /tracks/jobs/{job_id} for progress.
    """
    audio_ext = _ext_of(audio.filename, ALLOWED_AUDIO, "audio")
    title = (title or "").strip() or Path(audio.filename or "untitled").stem
    track_id = f"track-{uuid4().hex[:12]}"

    audio_dest = AUDIO_DIR / f"{track_id}{audio_ext}"
    with audio_dest.open("wb") as f:
        while chunk := audio.file.read(1024 * 1024):
            f.write(chunk)

    artwork_path: str | None = None
    if artwork is not None and artwork.filename:
        art_ext = _ext_of(artwork.filename, ALLOWED_ARTWORK, "artwork")
        art_dest = ARTWORK_DIR / f"{track_id}{art_ext}"
        with art_dest.open("wb") as f:
            while chunk := artwork.file.read(1024 * 1024):
                f.write(chunk)
        artwork_path = str(art_dest)

    record = store.save_track(
        TrackRecord(
            id=track_id,
            title=title,
            artist=(artist or "").strip() or "Unknown Artist",
            original_file_path=str(audio_dest),
            artwork_path=artwork_path,
            created_at=datetime.now(timezone.utc),
        )
    )
    job = jobs.create_job(track_id)
    background.add_task(run_pipeline, job["job_id"], track_id, str(audio_dest))
    return {"track": record, "job_id": job["job_id"]}


@router.post("/{track_id}/artwork", response_model=TrackRecord)
def upload_artwork(track_id: str, artwork: UploadFile = File(...)) -> TrackRecord:
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    art_ext = _ext_of(artwork.filename, ALLOWED_ARTWORK, "artwork")
    art_dest = ARTWORK_DIR / f"{track_id}{art_ext}"
    with art_dest.open("wb") as f:
        while chunk := artwork.file.read(1024 * 1024):
            f.write(chunk)
    track.artwork_path = str(art_dest)
    return store.save_track(track)


@router.get("/{track_id}/artwork")
def get_artwork(track_id: str):
    track = store.get_track(track_id)
    if track is None or not track.artwork_path:
        raise HTTPException(status_code=404, detail="Artwork not found")
    path = Path(track.artwork_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artwork not found")
    return FileResponse(path)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)


@router.get("/{track_id}", response_model=TrackRecord)
def get_track(track_id: str) -> TrackRecord:
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return track


@router.get("/{track_id}/analysis")
def get_analysis(track_id: str):
    record = store.get_analysis(track_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record
