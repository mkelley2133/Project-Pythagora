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


def _demo_telemetry() -> TelemetryBundle:
    # 92.4 BPM, 32 bars of 4/4 -> bar = 240/92.4 s, total ~= 83.12 s.
    # Placeholder telemetry for the starter dashboard; the analysis workers
    # replace this with measured values.
    return TelemetryBundle(
        bpm=92.4,
        duration=83.12,
        musical_key="D",
        mode="minor",
        sub_genres=[{"name": "dream pop", "confidence": 0.82}],
        emotional_profile={"hope": 0.7, "melancholy": 0.8},
        chords=["Dm", "Bb", "F", "C"],
        segments=[
            {"name": "Intro", "start": 0.0, "end": 10.39},
            {"name": "Verse", "start": 10.39, "end": 31.18},
            {"name": "Chorus", "start": 31.18, "end": 51.95},
            {"name": "Verse", "start": 51.95, "end": 62.34},
            {"name": "Chorus", "start": 62.34, "end": 83.12},
        ],
        rhyme_scheme_summary="ABCB",
        timestamped_lyrics=[
            {"start": 10.5, "end": 15.5, "text": "Neon hums on an empty street", "rhyme": "A"},
            {"start": 15.7, "end": 20.7, "text": "I chase the echo of your heartbeat", "rhyme": "B"},
            {"start": 20.9, "end": 25.9, "text": "Midnight folds the sky in two", "rhyme": "C"},
            {"start": 26.1, "end": 31.0, "text": "Still every road leads back to you", "rhyme": "B"},
            {"start": 31.3, "end": 36.3, "text": "Hold the light, don't let it fade", "rhyme": "A"},
            {"start": 36.5, "end": 41.5, "text": "We are embers in the snow", "rhyme": "B"},
            {"start": 41.7, "end": 46.7, "text": "If the morning takes your hand", "rhyme": "C"},
            {"start": 46.9, "end": 51.8, "text": "I'll be waiting in the snow", "rhyme": "B"},
            {"start": 52.1, "end": 57.1, "text": "Static blooms on the radio", "rhyme": "A"},
            {"start": 57.3, "end": 62.2, "text": "Take me anywhere but slow", "rhyme": "B"},
            {"start": 62.5, "end": 67.5, "text": "Hold the light, don't let it fade", "rhyme": "A"},
            {"start": 67.7, "end": 72.7, "text": "We are embers in the snow", "rhyme": "B"},
            {"start": 72.9, "end": 77.9, "text": "If the morning takes your hand", "rhyme": "C"},
            {"start": 78.1, "end": 83.0, "text": "I'll be waiting in the snow", "rhyme": "B"},
        ],
        musicological_essay=(
            "A concise placeholder analysis for the starter pipeline. "
            "The full essay stage synthesizes locked telemetry — harmonic "
            "movement, rhyme architecture, and emotional contour — into a "
            "grounded narrative of the track's journey from intro to outro."
        ),
    )


@router.get("/{track_id}/analysis", response_model=TrackAnalysisRecord)
def get_analysis(track_id: str) -> TrackAnalysisRecord:
    if track_id != "demo-track-001":
        raise HTTPException(status_code=404, detail="Track not found")

    return TrackAnalysisRecord(id=track_id, telemetry_json=_demo_telemetry())
