from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SubGenre(BaseModel):
    name: str = Field(..., description="Hierarchical sub-genre label")
    confidence: float = Field(..., ge=0.0, le=1.0)


class TelemetryBundle(BaseModel):
    bpm: float = Field(..., ge=0.0)
    musical_key: str = Field(default="C")
    mode: str = Field(default="major")
    sub_genres: list[SubGenre] = Field(default_factory=list)
    emotional_profile: dict[str, float] = Field(default_factory=dict)
    chords: list[str] = Field(default_factory=list)
    rhyme_scheme_summary: str = Field(default="")
    timestamped_lyrics: list[dict[str, Any]] = Field(default_factory=list)
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
