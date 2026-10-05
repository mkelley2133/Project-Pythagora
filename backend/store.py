"""Shared in-memory stores for tracks and analyses.

The router and the pipeline both read/write here. This is the right store
for the starter single-process deployment; a production deployment replaces
these dicts with the SQLAlchemy layer in backend/database.py.
"""
from __future__ import annotations

from datetime import datetime, timezone

from backend.models.schemas import (
    TelemetryBundle,
    TrackAnalysisRecord,
    TrackRecord,
)

_tracks: dict[str, TrackRecord] = {}
_analyses: dict[str, TrackAnalysisRecord] = {}


def list_tracks() -> list[TrackRecord]:
    return list(_tracks.values())


def get_track(track_id: str) -> TrackRecord | None:
    return _tracks.get(track_id)


def save_track(track: TrackRecord) -> TrackRecord:
    _tracks[track.id] = track
    return track


def get_analysis(track_id: str) -> TrackAnalysisRecord | None:
    return _analyses.get(track_id)


def save_analysis(record: TrackAnalysisRecord) -> TrackAnalysisRecord:
    _analyses[record.id] = record
    return record


def seed_demo() -> None:
    """Seed the demo track + placeholder telemetry (replaced by real pipeline
    output once a track is uploaded and analyzed)."""
    if _tracks:
        return
    track = TrackRecord(
        id="demo-track-001",
        title="Sample Signal",
        artist="Pythagoras",
        original_file_path="/storage/demo.mp3",
        vocal_stem_path="/storage/demo_vocals.wav",
        instrumental_stem_path="/storage/demo_inst.wav",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _tracks[track.id] = track
    # 92.4 BPM, 32 bars of 4/4 -> bar = 240/92.4 s, total ~= 83.12 s.
    telemetry = TelemetryBundle(
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
    _analyses[track.id] = TrackAnalysisRecord(id=track.id, telemetry_json=telemetry)
