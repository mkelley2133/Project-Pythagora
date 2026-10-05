from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from backend.models.schemas import (
    TelemetryBundle,
    TrackAnalysisRecord,
    TrackCreate,
    TrackRecord,
)

router = APIRouter(prefix="/tracks", tags=["tracks"])

# In-memory store backing the starter pipeline; the database layer in
# backend/database.py replaces this once models are wired to SQLAlchemy.
_tracks: dict[str, TrackRecord] = {
    "demo-track-001": TrackRecord(
        id="demo-track-001",
        title="Sample Signal",
        artist="Pythagoras",
        original_file_path="/storage/demo.mp3",
        vocal_stem_path="/storage/demo_vocals.wav",
        instrumental_stem_path="/storage/demo_inst.wav",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
}


@router.get("", response_model=list[TrackRecord])
def list_tracks() -> list[TrackRecord]:
    return list(_tracks.values())


@router.post("", response_model=TrackRecord, status_code=status.HTTP_201_CREATED)
def create_track(track: TrackCreate) -> TrackRecord:
    record = TrackRecord(
        id=f"track-{uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc),
        **track.model_dump(),
    )
    _tracks[record.id] = record
    return record


@router.get("/{track_id}", response_model=TrackRecord)
def get_track(track_id: str) -> TrackRecord:
    try:
        return _tracks[track_id]
    except KeyError:
        raise HTTPException(status_code=404, detail="Track not found")


@router.get("/{track_id}/analysis", response_model=TrackAnalysisRecord)
def get_analysis(track_id: str) -> TrackAnalysisRecord:
    if track_id != "demo-track-001":
        raise HTTPException(status_code=404, detail="Track not found")

    telemetry = TelemetryBundle(
        bpm=92.4,
        musical_key="D",
        mode="minor",
        sub_genres=[{"name": "dream pop", "confidence": 0.82}],
        emotional_profile={"hope": 0.7, "melancholy": 0.8},
        chords=["Dm", "Bb", "F", "C"],
        rhyme_scheme_summary="ABCB",
        musicological_essay="A concise placeholder analysis for the starter pipeline.",
    )
    return TrackAnalysisRecord(id=track_id, telemetry_json=telemetry)
