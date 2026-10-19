from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SubGenre(BaseModel):
    """Legacy label/score pair. No longer populated by the pipeline:
    genre now comes from the trained model (GenrePrediction) and the
    old rule-based style tags are reported as TextureCue entries."""

    name: str = Field(..., description="Hierarchical sub-genre label")
    confidence: float = Field(..., ge=0.0, le=1.0)


class GenrePrediction(BaseModel):
    """One genre prediction from the trained genre model.

    `probability` is the model's own output score for the label
    (PANNs Cnn10: an AudioSet sigmoid score, averaged across analysis
    windows) — never a rank-based placeholder."""

    label: str = Field(..., description="Genre label from the model's taxonomy")
    probability: float = Field(..., ge=0.0, le=1.0)


class SubgenrePrediction(BaseModel):
    """One zero-shot subgenre match from CLAP (workers/analysis/subgenre.py).

    `percent` is a similarity share (softmax over a fixed candidate
    prompt set, in percent) — NOT a trained classifier probability.
    The bundle's subgenre_note states this wherever the values appear.
    """

    label: str = Field(..., description="Subgenre label from the candidate vocabulary")
    family: str = Field(..., description="Broad family the subgenre belongs to")
    percent: float = Field(..., ge=0.0, le=100.0, description="Similarity share (%)")


class EmotionPrediction(BaseModel):
    """One zero-shot emotion match from CLAP (workers/analysis/emotion.py).

    `percent` is a zero-shot emotion similarity share (softmax over a
    fixed candidate prompt set, in percent) — NOT a trained classifier
    probability and NOT a claim about what the track actually feels.
    The bundle's emotion_note states this wherever the values appear.
    """

    label: str = Field(..., description="Emotion label from the candidate vocabulary")
    percent: float = Field(
        ..., ge=0.0, le=100.0, description="Zero-shot emotion similarity share (%)"
    )


class TextureCue(BaseModel):
    """A texture/style cue from the deterministic style rules
    (workers/analysis/audio.py style_tags). Texture is NOT genre:
    e.g. 'Ambient / pads' describes sustained harmonic texture, and
    `note` carries the measured evidence that fired the rule."""

    name: str = Field(..., description="Texture/style cue label")
    note: str = Field(default="", description="Measured evidence behind the cue")


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
        description="Confidence attached to a supplied lyric line; None when unknown",
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
    f0_median_hz: float = Field(default=0.0, ge=0.0)
    sub80_energy_share: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Share of spectral energy below 80 Hz"
    )
    separation_ok: bool = Field(
        default=True,
        description="False when the melodic line could not be separated from bass content",
    )
    refusal_reason: str | None = Field(default=None)


class MixMeasurements(BaseModel):
    """Deterministic mix measurements (workers/analysis/measure.py).

    Band shares are percentages of total spectral energy. Levels are true
    sample levels in dBFS (not normalized-envelope relatives). Stereo
    fields are None for mono sources.
    """

    sub_pct: float = Field(default=0.0, ge=0.0, description="Energy share 20-60 Hz (%)")
    low_pct: float = Field(default=0.0, ge=0.0, description="Energy share 60-250 Hz (%)")
    mid_pct: float = Field(default=0.0, ge=0.0, description="Energy share 250 Hz-2 kHz (%)")
    high_pct: float = Field(default=0.0, ge=0.0, description="Energy share 2-8 kHz (%)")
    air_pct: float = Field(default=0.0, ge=0.0, description="Energy share 8 kHz+ (%)")
    peak_dbfs: float = Field(default=0.0, description="True peak level, dBFS")
    rms_dbfs: float = Field(default=0.0, description="RMS level, dBFS")
    crest_db: float = Field(default=0.0, ge=0.0, description="Peak-to-RMS distance, dB")
    is_stereo: bool = Field(default=False)
    stereo_correlation: float | None = Field(
        default=None, description="Inter-channel correlation -1..1; None when mono"
    )
    stereo_width: float | None = Field(
        default=None, ge=0.0, description="Side/mid RMS ratio; None when mono"
    )


class MixSuggestion(BaseModel):
    """One mix suggestion, fired by a deterministic rule over measured
    values. `why` always quotes the measured value that fired the rule —
    no measured trigger, no suggestion (zero-hallucination rule)."""

    area: str = Field(..., description="Mix area, e.g. 'Low end', 'Levels', 'Stereo'")
    suggestion: str = Field(..., description="What to try")
    why: str = Field(..., description="Measured evidence, with the number")
    severity: str = Field(default="low", description="high | medium | low")


class TelemetryBundle(BaseModel):
    bpm: float = Field(..., ge=0.0)
    duration: float = Field(default=0.0, ge=0.0, description="Track length in seconds")
    musical_key: str = Field(default="C")
    mode: str = Field(default="major")
    key_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Krumhansl-Schmuckler key confidence"
    )
    vocal_note: str | None = Field(
        default=None,
        description="Set when vocal forensics were refused (e.g. insufficient separation)",
    )
    primary_genre: str | None = Field(
        default=None,
        description=(
            "Top genre label from the trained genre model; 'Mixed / uncertain' "
            "when the top probability is below the confidence threshold; "
            "None when the genre model was unavailable"
        ),
    )
    genre_predictions: list[GenrePrediction] = Field(
        default_factory=list,
        description="Top genre predictions with the model's real output probabilities",
    )
    genre_model: str | None = Field(
        default=None, description="Name/version of the trained genre model that ran"
    )
    genre_note: str | None = Field(
        default=None,
        description="Genre caveats: low-confidence explanation, taxonomy limits, or why the model was unavailable",
    )
    primary_subgenre: str | None = Field(
        default=None,
        description=(
            "Top zero-shot subgenre match (CLAP) when its similarity share "
            "is decisive; None when no match commits or the model was unavailable"
        ),
    )
    subgenre_predictions: list[SubgenrePrediction] = Field(
        default_factory=list,
        description="Top subgenre matches with similarity-share percents (zero-shot, not probabilities)",
    )
    subgenre_model: str | None = Field(
        default=None, description="Name/version of the zero-shot subgenre model that ran"
    )
    subgenre_note: str | None = Field(
        default=None,
        description="Subgenre caveats: the similarity-share honesty note, the no-commit explanation, or why the model was unavailable",
    )
    primary_emotion: str | None = Field(
        default=None,
        description=(
            "Top zero-shot emotion match (CLAP) when its similarity share "
            "is decisive; None when no match commits or the model was unavailable. "
            "A similarity read, never a definitive statement of what the track feels"
        ),
    )
    emotion_predictions: list[EmotionPrediction] = Field(
        default_factory=list,
        description="Top emotion matches with zero-shot emotion similarity-share percents (not probabilities)",
    )
    emotion_model: str | None = Field(
        default=None, description="Name/version of the zero-shot emotion model that ran"
    )
    emotion_note: str | None = Field(
        default=None,
        description="Emotion caveats: the zero-shot similarity-share honesty note, the no-commit explanation, or why the model was unavailable",
    )
    texture_cues: list[TextureCue] = Field(
        default_factory=list,
        description="Deterministic texture/style cues (not genre) from the style rules",
    )
    sub_genres: list[SubGenre] = Field(
        default_factory=list,
        description="Deprecated: kept for schema compatibility, no longer populated — see primary_genre / texture_cues",
    )
    emotional_profile: dict[str, float] = Field(default_factory=dict)
    chords: list[str] = Field(default_factory=list)
    segments: list[StructuralSegment] = Field(default_factory=list)
    chord_timeline: list[ChordEvent] = Field(default_factory=list)
    vocal_profile: VocalProfile | None = Field(default=None)
    waveform_peaks: list[float] = Field(default_factory=list)
    mix_measurements: MixMeasurements | None = Field(
        default=None, description="Measured band levels / dynamics / stereo (Phase G)"
    )
    suggestions: list[MixSuggestion] = Field(
        default_factory=list,
        description="Rule-fired mix suggestions; each quotes its measured trigger",
    )
    rhyme_scheme_summary: str = Field(default="")
    timestamped_lyrics: list[LyricLine] = Field(default_factory=list)
    musicological_essay: str = Field(default="")
    stems: list[str] = Field(
        default_factory=list,
        description="Always empty in v1: stem separation is excluded from the customer version",
    )
    separation_model: str | None = Field(
        default=None, description="Always None in v1: no separation runs"
    )
    lyrics_language: str | None = Field(
        default=None, description="Language of a supplied lyric sheet; None when none supplied"
    )
    lyrics_avg_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mean confidence of supplied lyric lines; None when not applicable",
    )
    lyrics_note: str | None = Field(
        default=None,
        description="Note about the lyric sheet (e.g. that none was supplied); never auto-transcription output",
    )


class TrackCreate(BaseModel):
    """Client-supplied fields for registering a new track.

    The server assigns `id` and `created_at` — clients cannot spoof them.
    """

    title: str = Field(..., min_length=1, max_length=200)
    artist: str = Field(..., min_length=1, max_length=200)
    original_file_path: str = Field(..., min_length=1, max_length=1024)


class TrackRecord(BaseModel):
    id: str
    title: str
    artist: str
    original_file_path: str
    artwork_path: str | None = Field(default=None, description="Uploaded cover art")
    project_id: str | None = Field(
        default=None, description="Project this track belongs to; None = unsorted"
    )
    notes: str = Field(default="", description="Free-text notes about this track")
    created_at: datetime | None = None


class TrackUpdate(BaseModel):
    """Organization edits: assign/unassign a project and/or edit notes.

    `project_id` uses explicit-None semantics: send null to unassign, omit
    the field to leave the assignment untouched (the router checks
    model_fields_set).
    """

    project_id: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=4000)


class Project(BaseModel):
    id: str
    name: str = Field(..., min_length=1, max_length=200)
    notes: str = Field(default="", max_length=8000)
    created_at: datetime | None = None


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    notes: str = Field(default="", max_length=8000)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=8000)


class JobStage(BaseModel):
    """One pipeline stage and its live status."""

    name: str
    status: str = Field(default="pending", description="pending | active | done | skipped")


class JobStatus(BaseModel):
    """Lifecycle of one analysis pipeline run."""

    job_id: str
    track_id: str
    state: str = Field(description="queued | running | done | failed")
    stage: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    error: str | None = None
    stage_index: int = Field(default=0, description="Index of the active stage (0-based)")
    stage_total: int = Field(default=0, description="Total number of stages")
    stages: list[JobStage] = Field(default_factory=list)
    started_at: float | None = Field(
        default=None, description="Epoch seconds when the job started running"
    )
    elapsed_seconds: float | None = Field(
        default=None, description="Seconds since the job started (frozen when finished)"
    )


class TrackAnalysisRecord(BaseModel):
    id: str
    telemetry_json: TelemetryBundle
    created_at: datetime | None = None
