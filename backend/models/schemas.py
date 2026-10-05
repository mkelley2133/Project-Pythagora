from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SubGenre(BaseModel):
    name: str = Field(..., description="Hierarchical sub-genre label")
    confidence: float = Field(..., ge=0.0, le=1.0)


class StructuralSegment(BaseModel):
    """One structural section of a track (intro, verse, chorus, ...)."""

    name: str = Field(..., description="Segment label, e.g. Verse, Chorus")
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., gt=0.0, description="End time in seconds")


class LyricLine(BaseModel):
    """A timestamped lyric line with its rhyme-scheme tag."""

    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    text: str
    rhyme: str = Field(default="", description="Rhyme-scheme letter, e.g. A, B")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Transcription confidence 0-1 from Whisper log-probs; None when unknown",
    )


class ChordEvent(BaseModel):
    """One recognized chord with its position and Roman numeral."""

    chord: str = Field(..., description="Chord symbol, e.g. Dm")
    roman: str = Field(default="", description="Roman numeral relative to the track key")
    start: float = Field(default=0.0, ge=0.0)
    end: float = Field(default=0.0, ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VocalProfile(BaseModel):
    """Measured vocal-production features.

    autotune_likelihood is a heuristic (0-1), not a verdict — see
    workers/analysis/vocals.py for its caveats.
    """

    f0_mean_hz: float = Field(default=0.0, ge=0.0)
    f0_range_semitones: float = Field(default=0.0, ge=0.0)
    vibrato_rate_hz: float = Field(default=0.0, ge=0.0)
    vibrato_depth_cents: float = Field(default=0.0, ge=0.0)
    autotune_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    voiced_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class TelemetryBundle(BaseModel):
    bpm: float = Field(..., ge=0.0)
    duration: float = Field(default=0.0, ge=0.0, description="Track length in seconds")
    musical_key: str = Field(default="C")
    mode: str = Field(default="major")
    sub_genres: list[SubGenre] = Field(default_factory=list)
    emotional_profile: dict[str, float] = Field(default_factory=dict)
    chords: list[str] = Field(default_factory=list)
    segments: list[StructuralSegment] = Field(default_factory=list)
    chord_timeline: list[ChordEvent] = Field(default_factory=list)
    vocal_profile: VocalProfile | None = Field(default=None)
    waveform_peaks: list[float] = Field(default_factory=list)
    rhyme_scheme_summary: str = Field(default="")
    timestamped_lyrics: list[LyricLine] = Field(default_factory=list)
    musicological_essay: str = Field(default="")


class TrackCreate(BaseModel):
    """Client-supplied fields for registering a new track.

    The server assigns `id` and `created_at` — clients cannot spoof them.
    """

    title: str = Field(..., min_length=1)
    artist: str = Field(..., min_length=1)
    original_file_path: str = Field(..., min_length=1)


class TrackRecord(BaseModel):
    id: str
    title: str
    artist: str
    original_file_path: str
    vocal_stem_path: str | None = None
    instrumental_stem_path: str | None = None
    artwork_path: str | None = Field(default=None, description="Uploaded cover art")
    created_at: datetime | None = None


class JobStatus(BaseModel):
    """Lifecycle of one analysis pipeline run."""

    job_id: str
    track_id: str
    state: str = Field(description="queued | running | done | failed")
    stage: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    error: str | None = None


class TrackAnalysisRecord(BaseModel):
    id: str
    telemetry_json: TelemetryBundle
    created_at: datetime | None = None
