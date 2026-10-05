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


class TelemetryBundle(BaseModel):
    bpm: float = Field(..., ge=0.0)
    duration: float = Field(default=0.0, ge=0.0, description="Track length in seconds")
    musical_key: str = Field(default="C")
    mode: str = Field(default="major")
    sub_genres: list[SubGenre] = Field(default_factory=list)
    emotional_profile: dict[str, float] = Field(default_factory=dict)
    chords: list[str] = Field(default_factory=list)
    segments: list[StructuralSegment] = Field(default_factory=list)
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
    created_at: datetime | None = None


class TrackAnalysisRecord(BaseModel):
    id: str
    telemetry_json: TelemetryBundle
    created_at: datetime | None = None
