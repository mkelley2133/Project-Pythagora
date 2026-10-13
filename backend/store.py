"""Shared stores for tracks, analyses, and projects.

Tracks/analyses live in in-memory dicts (right for the starter
single-process deployment; a production deployment replaces them with the
SQLAlchemy layer in backend/database.py).

Projects, track metadata, and notes are ALSO persisted to a small JSON
library file (default `<storage_dir>/library.json`, override with the
PYTHAGORAS_LIBRARY_FILE env var) so organization work — projects,
assignments, notes — survives a server restart. Analyses are not
persisted; they can be re-measured from the stored audio.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend.config import settings
from backend.models.schemas import (
    Project,
    TelemetryBundle,
    TrackAnalysisRecord,
    TrackRecord,
)

_tracks: dict[str, TrackRecord] = {}
_analyses: dict[str, TrackAnalysisRecord] = {}
_projects: dict[str, Project] = {}

DEMO_TRACK_ID = "demo-track-001"
DEMO_PROJECT_ID = "demo-project-001"
_DEMO_IDS = {DEMO_TRACK_ID, DEMO_PROJECT_ID}

_persist_lock = threading.Lock()
_loaded = False


# ------------------------------------------------------------- persistence

def _library_file() -> Path:
    override = os.environ.get("PYTHAGORAS_LIBRARY_FILE")
    if override:
        return Path(override)
    return Path(settings.storage_dir) / "library.json"


def _persist() -> None:
    """Write projects + track metadata (incl. notes/assignments) to disk.

    Demo records are session furniture, not user data — never persisted.
    Failures are swallowed: persistence must never break a request.
    """
    with _persist_lock:
        try:
            path = _library_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tracks": [
                    t.model_dump(mode="json")
                    for t in _tracks.values()
                    if t.id not in _DEMO_IDS
                ],
                "projects": [
                    p.model_dump(mode="json")
                    for p in _projects.values()
                    if p.id not in _DEMO_IDS
                ],
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            tmp.replace(path)
        except Exception:  # noqa: BLE001 - best-effort durability
            pass


def load_persisted() -> None:
    """Load projects + track metadata from the library file (once)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        path = _library_file()
        if not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload.get("projects", []):
            project = Project(**raw)
            _projects.setdefault(project.id, project)
        for raw in payload.get("tracks", []):
            track = TrackRecord(**raw)
            _tracks.setdefault(track.id, track)
    except Exception:  # noqa: BLE001 - a corrupt file must not brick boot
        pass


def reset_state() -> None:
    """Clear all in-memory state (tests / maintenance). Does not delete
    the library file; call load_persisted() afterwards to reload it."""
    global _loaded
    _tracks.clear()
    _analyses.clear()
    _projects.clear()
    _loaded = False


# ----------------------------------------------------------------- tracks

def list_tracks() -> list[TrackRecord]:
    return list(_tracks.values())


def get_track(track_id: str) -> TrackRecord | None:
    return _tracks.get(track_id)


def save_track(track: TrackRecord) -> TrackRecord:
    _tracks[track.id] = track
    _persist()
    return track


def get_analysis(track_id: str) -> TrackAnalysisRecord | None:
    return _analyses.get(track_id)


def save_analysis(record: TrackAnalysisRecord) -> TrackAnalysisRecord:
    _analyses[record.id] = record
    return record


# --------------------------------------------------------------- projects

def list_projects() -> list[Project]:
    return list(_projects.values())


def get_project(project_id: str) -> Project | None:
    return _projects.get(project_id)


def save_project(project: Project) -> Project:
    _projects[project.id] = project
    _persist()
    return project


def delete_project(project_id: str) -> bool:
    """Delete a project; its tracks are unassigned, never deleted."""
    if project_id not in _projects:
        return False
    del _projects[project_id]
    for track in _tracks.values():
        if track.project_id == project_id:
            track.project_id = None
    _persist()
    return True


def project_track_ids(project_id: str) -> list[str]:
    return [t.id for t in _tracks.values() if t.project_id == project_id]


# ------------------------------------------------------------------- seed

def seed_demo() -> None:
    """Seed the demo project + track (and load any persisted library)."""
    if DEMO_TRACK_ID not in _tracks:
        _projects.setdefault(
            DEMO_PROJECT_ID,
            Project(
                id=DEMO_PROJECT_ID,
                name="Sample Sessions",
                notes="Demo project — groups the sample track so projects, "
                "sorting, and notes have something to show before your first upload.",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        )
        track = TrackRecord(
            id=DEMO_TRACK_ID,
            title="Sample Signal",
            artist="Pythagoras",
            original_file_path="/storage/demo.mp3",
            project_id=DEMO_PROJECT_ID,
            notes="Placeholder telemetry until a real upload is analyzed.",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        _tracks[track.id] = track
    load_persisted()
    if DEMO_TRACK_ID in _tracks and DEMO_TRACK_ID not in _analyses:
        _analyses[DEMO_TRACK_ID] = TrackAnalysisRecord(
            id=DEMO_TRACK_ID, telemetry_json=_demo_telemetry()
        )


def _demo_telemetry() -> TelemetryBundle:
    # 92.4 BPM, 32 bars of 4/4 -> bar = 240/92.4 s, total ~= 83.12 s.
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
        mix_measurements={
            "sub_pct": 4.2,
            "low_pct": 21.5,
            "mid_pct": 46.8,
            "high_pct": 21.9,
            "air_pct": 5.6,
            "peak_dbfs": -1.2,
            "rms_dbfs": -16.4,
            "crest_db": 15.2,
            "is_stereo": True,
            "stereo_correlation": 0.62,
            "stereo_width": 0.58,
        },
        suggestions=[
            {
                "area": "Harmony",
                "suggestion": "The progression never leaves its 4-chord loop — "
                "consider a bridge or pre-chorus variation to reset the ear.",
                "why": "Only 4 unique chords (Dm, Bb, F, C) across the full 83s demo track.",
                "severity": "low",
            }
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
