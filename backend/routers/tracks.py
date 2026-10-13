"""Track routes — hardened upload/serving surface (see SECURITY.md).

Uploads are streamed to a server-named temp file with a byte cap, validated
by content sniffing (magic bytes) + an actual decode/duration probe, then
renamed to a server-generated name inside the dedicated storage directory.
User-supplied filenames never reach the filesystem, and stored files are
served only when they resolve inside the storage directory.
"""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, HTMLResponse

from backend import jobs, security, store
from backend.adjust import (
    cache_filename,
    clamp_adjust,
    effective_bpm,
    render_adjusted,
    transpose_key,
)
from backend.compare import build_comparison
from backend.config import settings
from backend.models.schemas import (
    JobStatus,
    TrackCreate,
    TrackRecord,
    TrackUpdate,
)
from backend.report import render_comparison_html, render_report_html

router = APIRouter(
    prefix="/tracks",
    tags=["tracks"],
    dependencies=[Depends(security.rate_limit_api)],
)

STORAGE = Path(settings.storage_dir)
AUDIO_DIR = STORAGE / "audio"
ARTWORK_DIR = STORAGE / "artwork"
ADJUSTED_DIR = STORAGE / "adjusted"
for _d in (AUDIO_DIR, ARTWORK_DIR, ADJUSTED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
ALLOWED_ARTWORK = {".jpg", ".jpeg", ".png", ".webp"}
ARTWORK_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

store.seed_demo()


def _ext_of(filename: str | None, allowed: set[str], kind: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported {kind} type: {ext or '?'}")
    return ext


def _stream_to_file(upload: UploadFile, dest: Path, max_bytes: int, kind: str) -> int:
    """Stream an upload to disk with a hard byte cap (413 when exceeded)."""
    written = 0
    with dest.open("wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"{kind} exceeds the {max_bytes // (1024 * 1024)} MB size limit.",
                )
            f.write(chunk)
    return written


def _sniff_head(path: Path, size: int = 16384) -> bytes:
    with path.open("rb") as f:
        return f.read(size)


@router.get("", response_model=list[TrackRecord])
def list_tracks() -> list[TrackRecord]:
    return store.list_tracks()


@router.post(
    "",
    response_model=TrackRecord,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(security.rate_limit_upload)],
)
def create_track(track: TrackCreate) -> TrackRecord:
    record = TrackRecord(
        id=f"track-{uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc),
        **track.model_dump(),
    )
    return store.save_track(record)


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(security.rate_limit_upload)],
)
def upload_track(
    title: str = Form(..., max_length=200),
    artist: str = Form("Unknown Artist", max_length=200),
    audio: UploadFile = File(...),
    artwork: UploadFile | None = File(None),
) -> dict:
    """Upload audio (+ optional cover art) and queue the analysis pipeline.

    Returns immediately with the track record and a job id; poll
    GET /tracks/jobs/{job_id} for progress. Jobs run on a capped executor:
    when all analysis slots are busy the job waits visibly in "queued"
    state instead of being rejected — batch uploads rely on this.
    """
    _ext_of(audio.filename, ALLOWED_AUDIO, "audio")
    title = (title or "").strip() or Path(audio.filename or "untitled").stem
    track_id = f"track-{uuid4().hex[:12]}"

    # 1) Stream to a server-named temp file under a hard byte cap.
    tmp_dest = AUDIO_DIR / f"{track_id}.part"
    _stream_to_file(audio, tmp_dest, settings.max_upload_bytes, "Audio")

    # 2) Validate by content, not by the claimed extension.
    audio_format = security.sniff_audio(_sniff_head(tmp_dest))
    if audio_format is None:
        tmp_dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="File content is not a supported audio format (wav, mp3, flac, ogg, m4a, aac).",
        )
    canonical_ext, _ = security.AUDIO_FORMATS[audio_format]

    # 3) Prove it decodes and enforce the duration cap.
    try:
        probe = security.probe_audio(tmp_dest)
    except ValueError:
        tmp_dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Audio could not be decoded.")
    if probe["duration"] > settings.max_duration_seconds:
        tmp_dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Track duration {probe['duration']:.0f}s exceeds the "
                f"{settings.max_duration_seconds:.0f}s limit."
            ),
        )

    audio_dest = AUDIO_DIR / f"{track_id}{canonical_ext}"
    tmp_dest.rename(audio_dest)

    # 4) Optional artwork: extension allowlist + byte cap (content sniffing
    #    for artwork is deferred — see SECURITY.md).
    artwork_path: str | None = None
    if artwork is not None and artwork.filename:
        try:
            art_ext = _ext_of(artwork.filename, ALLOWED_ARTWORK, "artwork")
            art_dest = ARTWORK_DIR / f"{track_id}{art_ext}"
            _stream_to_file(artwork, art_dest, settings.max_artwork_bytes, "Artwork")
            artwork_path = str(art_dest)
        except HTTPException:
            audio_dest.unlink(missing_ok=True)
            raise

    record = store.save_track(
        TrackRecord(
            id=track_id,
            title=title[:200],
            artist=(artist or "").strip()[:200] or "Unknown Artist",
            original_file_path=str(audio_dest),
            artwork_path=artwork_path,
            created_at=datetime.now(timezone.utc),
        )
    )
    job = jobs.create_job(track_id)
    jobs.enqueue(job["job_id"], track_id, str(audio_dest))
    return {"track": record, "job_id": job["job_id"]}


@router.post(
    "/{track_id}/artwork",
    response_model=TrackRecord,
    dependencies=[Depends(security.rate_limit_upload)],
)
def upload_artwork(track_id: str, artwork: UploadFile = File(...)) -> TrackRecord:
    security.validate_id(track_id, "track id")
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    art_ext = _ext_of(artwork.filename, ALLOWED_ARTWORK, "artwork")
    art_dest = ARTWORK_DIR / f"{track_id}{art_ext}"
    _stream_to_file(artwork, art_dest, settings.max_artwork_bytes, "Artwork")
    track.artwork_path = str(art_dest)
    return store.save_track(track)


@router.get("/{track_id}/artwork")
def get_artwork(track_id: str):
    security.validate_id(track_id, "track id")
    track = store.get_track(track_id)
    if track is None or not track.artwork_path:
        raise HTTPException(status_code=404, detail="Artwork not found")
    path = security.resolve_within_storage(track.artwork_path)
    if path is None:
        raise HTTPException(status_code=404, detail="Artwork not found")
    return FileResponse(
        path,
        media_type=ARTWORK_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        filename=f"{track_id}{path.suffix.lower()}",
    )


@router.api_route("/{track_id}/audio", methods=["GET", "HEAD"])
def get_audio(track_id: str):
    """Serve uploaded audio — only files inside the storage directory."""
    security.validate_id(track_id, "track id")
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    path = security.resolve_within_storage(track.original_file_path)
    if path is None:
        raise HTTPException(status_code=404, detail="Audio not found")
    media_type = "application/octet-stream"
    for fmt, (ext, mime) in security.AUDIO_FORMATS.items():
        if path.suffix.lower() == ext:
            media_type = mime
            break
    return FileResponse(path, media_type=media_type, filename=f"{track_id}{path.suffix.lower()}")


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    security.validate_id(job_id, "job id")
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)


def _comparison_or_404(a: str, b: str) -> dict:
    security.validate_id(a, "track id")
    security.validate_id(b, "track id")
    track_a, track_b = store.get_track(a), store.get_track(b)
    if track_a is None or track_b is None:
        raise HTTPException(status_code=404, detail="Track not found")
    rec_a, rec_b = store.get_analysis(a), store.get_analysis(b)
    if rec_a is None or rec_b is None:
        raise HTTPException(status_code=404, detail="Analysis not found for one of the tracks")
    return build_comparison(track_a, rec_a.telemetry_json, track_b, rec_b.telemetry_json)


@router.get("/compare")
def compare_tracks(a: str = Query(...), b: str = Query(...)) -> dict:
    """Side-by-side measured comparison of two analyzed tracks (Phase E)."""
    return _comparison_or_404(a, b)


@router.get("/compare/report", response_class=HTMLResponse)
def compare_report(a: str = Query(...), b: str = Query(...)) -> HTMLResponse:
    """Branded printable comparison report (Phase E)."""
    return HTMLResponse(content=render_comparison_html(_comparison_or_404(a, b)))


@router.get("/{track_id}", response_model=TrackRecord)
def get_track(track_id: str) -> TrackRecord:
    security.validate_id(track_id, "track id")
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return track


@router.get("/{track_id}/analysis")
def get_analysis(track_id: str):
    security.validate_id(track_id, "track id")
    record = store.get_analysis(track_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record


@router.get("/{track_id}/report", response_class=HTMLResponse)
def get_report(track_id: str) -> HTMLResponse:
    """Customer-facing printable HTML report for a completed analysis."""
    security.validate_id(track_id, "track id")
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    record = store.get_analysis(track_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return HTMLResponse(content=render_report_html(track, record.telemetry_json))


@router.patch("/{track_id}", response_model=TrackRecord)
def update_track(track_id: str, update: TrackUpdate) -> TrackRecord:
    """Organization edits (Phase F): assign/unassign a project, edit notes.

    project_id is explicit-null sensitive: sending null unassigns, omitting
    the field leaves the current assignment alone.
    """
    security.validate_id(track_id, "track id")
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    if "project_id" in update.model_fields_set:
        if update.project_id is not None:
            security.validate_id(update.project_id, "project id")
            if store.get_project(update.project_id) is None:
                raise HTTPException(status_code=404, detail="Project not found")
        track.project_id = update.project_id
    if "notes" in update.model_fields_set and update.notes is not None:
        track.notes = update.notes
    return store.save_track(track)


def _adjusted_meta(track_id: str, semitones: float, speed: float) -> dict:
    record = store.get_analysis(track_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    t = record.telemetry_json
    st, sp = clamp_adjust(semitones, speed)
    return {
        "track_id": track_id,
        "semitones": st,
        "speed": sp,
        "original_bpm": t.bpm,
        "effective_bpm": effective_bpm(t.bpm, sp),
        "original_key": f"{t.musical_key} {t.mode}",
        "effective_key": f"{transpose_key(t.musical_key, st)} {t.mode}",
        "derived": True,
    }


@router.get("/{track_id}/adjusted/meta")
def get_adjusted_meta(
    track_id: str,
    semitones: float = Query(0.0),
    speed: float = Query(1.0),
) -> dict:
    """Derived tempo/key readout for a pitch/speed adjustment (Phase H).

    Values are computed from the original measured analysis and labeled
    derived — the underlying analysis is never modified.
    """
    security.validate_id(track_id, "track id")
    if store.get_track(track_id) is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return _adjusted_meta(track_id, semitones, speed)


@router.get(
    "/{track_id}/adjusted",
    dependencies=[Depends(security.rate_limit_upload)],
)
def get_adjusted_audio(
    track_id: str,
    semitones: float = Query(0.0),
    speed: float = Query(1.0),
):
    """Serve a pitch/speed-adjusted render of the track (Phase H).

    Pitch and speed are applied independently (librosa pitch_shift +
    time_stretch). Renders are cached in the storage dir under a name
    built only from the validated track id and server-formatted floats;
    the original file is never overwritten.
    """
    security.validate_id(track_id, "track id")
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    src = security.resolve_within_storage(track.original_file_path)
    if src is None:
        raise HTTPException(status_code=404, detail="Audio not found")
    st, sp = clamp_adjust(semitones, speed)
    dest = ADJUSTED_DIR / cache_filename(track_id, st, sp)
    if not dest.is_file():
        try:
            render_adjusted(src, dest, st, sp)
        except Exception:  # noqa: BLE001 - surfaced as a clean 500 below
            raise HTTPException(status_code=500, detail="Adjusted render failed.")
    headers = {"X-Adjusted-Semitones": f"{st:+.2f}", "X-Adjusted-Speed": f"{sp:.3f}"}
    record = store.get_analysis(track_id)
    if record is not None:
        t = record.telemetry_json
        headers["X-Effective-BPM"] = f"{effective_bpm(t.bpm, sp):.1f}"
        headers["X-Effective-Key"] = f"{transpose_key(t.musical_key, st)} {t.mode}"
    return FileResponse(
        dest, media_type="audio/wav", filename=dest.name, headers=headers
    )
