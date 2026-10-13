"""Analysis pipeline: audio file -> TelemetryBundle, with job progress.

Stages are deterministic measurements, with two model-based exceptions
in the genre stage: broad genre runs a trained model (PANNs Cnn10,
AudioSet) and reports its real output probabilities — see
workers/analysis/genre.py — and subgenre runs zero-shot CLAP matching
whose similarity shares are labeled as such, never as probabilities —
see workers/analysis/subgenre.py. Both degrade gracefully when their
model files are absent. Anything that cannot be measured or credibly
inferred is skipped or left empty — never invented.

V1 scope (customer version): stem separation and automatic lyric
transcription are excluded — the pipeline analyzes the full mix, and
vocal forensics keeps its refusal guard for bass-heavy material. Lyric
fields on the bundle are only ever populated from lyric sheets supplied
with a track, never from an automatic transcription.
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from backend import jobs
from backend import store
from backend.models.schemas import (
    ChordEvent,
    GenrePrediction,
    MixMeasurements,
    MixSuggestion,
    StructuralSegment,
    SubgenrePrediction,
    TextureCue,
    TelemetryBundle,
    TrackAnalysisRecord,
    VocalProfile,
)
from workers.analysis import audio as audio_mod
from workers.analysis import genre as genre_mod
from workers.analysis import subgenre as subgenre_mod
from workers.analysis.chords import recognize_chords_from_audio
from workers.analysis.measure import measure_mix
from workers.analysis.suggestions import build_suggestions
from workers.analysis.vocals import analyze_vocals
from workers.analysis.waveform import waveform_peaks

# The named pipeline stages, in order — the dashboard stepper mirrors this
# list exactly (jobs.set_stages / jobs.advance_stage drive it).
STAGES = [
    "Loading audio",
    "Measuring tempo & key",
    "Recognizing chords",
    "Mapping sections & texture",
    "Classifying genre",
    "Rendering waveform",
    "Vocal forensics",
    "Measuring mix & suggestions",
    "Finalizing",
]


def _ensure_wav(audio_path: str) -> str:
    """Return a wav path librosa can read; convert via ffmpeg when needed."""
    p = Path(audio_path)
    if p.suffix.lower() == ".wav":
        return str(p)
    if shutil.which("ffmpeg"):
        out = p.with_suffix(".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-ar", "22050", "-ac", "1", str(out)],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            # Surface an honest, user-readable reason on the job record
            # instead of the raw subprocess command (which leaks server
            # paths and tells the user nothing actionable).
            raise RuntimeError(
                "Audio conversion failed: ffmpeg could not decode this file."
            ) from exc
        return str(out)
    return str(p)  # librosa will try (needs audioread for m4a)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def _deterministic_summary(bundle: TelemetryBundle, track_title: str) -> str:
    """Template-built essay from measured values only — no generative model,
    so nothing here can be ungrounded."""
    t = bundle
    chords = ", ".join(dict.fromkeys(e.chord for e in t.chord_timeline[:8])) or "undetermined"
    segs = ", ".join(f"{s.name} ({s.start:.0f}s–{s.end:.0f}s)" for s in t.segments[:6])
    rhyme = (
        f" Rhyme scheme {t.rhyme_scheme_summary} (from the supplied lyric sheet)."
        if t.rhyme_scheme_summary
        else ""
    )
    return (
        f"Measured analysis of '{track_title}': {t.bpm:.1f} BPM in "
        f"{t.musical_key} {t.mode} ({t.duration:.0f}s). Harmonic movement "
        f"centers on {chords}. Structure: {segs or 'not segmented'}.{rhyme} "
        "Every figure above is a deterministic measurement, not an interpretation."
    )


def run_pipeline(job_id: str, track_id: str, audio_path: str) -> dict:
    jobs.set_stages(job_id, STAGES)
    jobs.update_job(job_id, state="running")

    def stage(label: str) -> None:
        jobs.advance_stage(job_id, label)

    try:
        stage("Loading audio")
        wav_path = _ensure_wav(audio_path)
        y, sr = audio_mod.load_audio(wav_path)

        stage("Measuring tempo & key")
        tempo = audio_mod.estimate_tempo(y, sr) or 0.0
        key = audio_mod.estimate_key(y, sr)
        if not key.get("key"):
            # Silent / unpitched audio has no determinable key (the
            # estimator reports None rather than inventing one). Fail
            # the job with that honest reason instead of crashing on
            # the missing value downstream.
            raise ValueError(
                "No musical key could be determined — the audio appears "
                "to be silent or contain no pitched content."
            )
        key_name, mode = key["key"].rsplit(" ", 1)

        stage("Recognizing chords")
        # One HPSS serves the chord, feature, and section stages (each
        # used to run its own copy on the same samples — the pipeline's
        # most expensive repeated computation). If it fails, stages fall
        # back to computing what they need themselves, exactly as before.
        try:
            hpss_pair = audio_mod.hpss_waveform(y)
        except Exception:  # noqa: BLE001 - per-stage fallback preserves behavior
            hpss_pair = None
        chord_events = [
            ChordEvent(**e)
            for e in recognize_chords_from_audio(
                y,
                sr,
                key=key_name,
                mode=mode,
                y_harm=hpss_pair[0] if hpss_pair is not None else None,
            )
        ]

        stage("Mapping sections & texture")
        feats = audio_mod.core_features(y, sr, hpss_pair=hpss_pair)
        sections = [
            StructuralSegment(name=s.get("role") or f"Section {i + 1}", start=float(s["start"]), end=float(s["end"]))
            for i, s in enumerate(
                audio_mod.section_map(
                    y,
                    sr,
                    perc_waveform=hpss_pair[1] if hpss_pair is not None else None,
                )
            )
        ]
        mood = audio_mod.mood_label(feats, tempo or None, key)
        styles = audio_mod.style_tags(feats, tempo or None, key)

        stage("Classifying genre")
        # Trained genre model (PANNs Cnn10). Degrades gracefully: when
        # the model file is missing/unloadable this returns
        # primary_genre=None with an explicit note — never raises.
        genre = genre_mod.analyze_genre(wav_path)
        # Zero-shot subgenre matching (LAION CLAP). Same degradation
        # contract; its percents are similarity shares, not trained
        # probabilities (labeled in subgenre_note and downstream).
        subgenre = subgenre_mod.analyze_subgenre(wav_path)

        stage("Rendering waveform")
        peaks = waveform_peaks(wav_path)

        stage("Vocal forensics")
        # Full mix only in v1 (no stem separation): when the melodic line
        # can't be separated from bass content, analyze_vocals' guard makes
        # us refuse with a reason rather than attach a confident wrong
        # profile.
        vox = analyze_vocals(wav_path)
        vocal_profile = None
        vocal_note: str | None = None
        if (
            vox.get("separation_ok")
            and vox["voiced_fraction"] > 0.15
            and vox["f0_mean_hz"] > 80
        ):
            vocal_profile = VocalProfile(**vox)
        elif vox.get("refusal_reason"):
            vocal_note = vox["refusal_reason"]
        elif vox["voiced_fraction"] <= 0.15:
            vocal_note = (
                "Vocal forensics not reported: too little voiced content was "
                "detected to measure reliably."
            )

        stage("Measuring mix & suggestions")
        mix = measure_mix(wav_path)
        mix_measurements = MixMeasurements(**mix)
        unique_chords = list(dict.fromkeys(e.chord for e in chord_events))
        suggestions = [
            MixSuggestion(**s)
            for s in build_suggestions(
                {
                    "mix": mix,
                    "feats": feats,
                    "vocal_profile": vocal_profile.model_dump() if vocal_profile else None,
                    "key_confidence": key.get("confidence"),
                    "unique_chords": unique_chords,
                    "chord_events": len(chord_events),
                    "duration": float(feats.get("duration", 0.0)),
                }
            )
        ]

        stage("Finalizing")
        track = store.get_track(track_id)
        title = track.title if track else track_id
        emotional_profile = {
            _slug(mood["mood"]): 0.9,
            "arousal": mood["arousal"],
            "valence": mood["valence"],
        }
        key_confidence = key.get("confidence")
        bundle = TelemetryBundle(
            bpm=round(float(tempo), 1),
            duration=round(float(feats.get("duration", 0.0)), 1),
            musical_key=key_name,
            mode=mode,
            key_confidence=(
                round(min(1.0, max(0.0, float(key_confidence))), 3)
                if key_confidence is not None
                else None
            ),
            vocal_note=vocal_note,
            primary_genre=genre.get("primary_genre"),
            genre_predictions=[
                GenrePrediction(**p) for p in genre.get("genre_predictions", [])
            ],
            genre_model=genre.get("genre_model"),
            genre_note=genre.get("genre_note"),
            primary_subgenre=subgenre.get("primary_subgenre"),
            subgenre_predictions=[
                SubgenrePrediction(**p) for p in subgenre.get("subgenre_predictions", [])
            ],
            subgenre_model=subgenre.get("subgenre_model"),
            subgenre_note=subgenre.get("subgenre_note"),
            texture_cues=[
                TextureCue(name=s["tag"], note=s.get("reason", "")) for s in styles[:4]
            ],
            emotional_profile=emotional_profile,
            chords=[e.chord for e in chord_events[:24]],
            chord_timeline=chord_events,
            segments=sections,
            vocal_profile=vocal_profile,
            waveform_peaks=peaks,
            mix_measurements=mix_measurements,
            suggestions=suggestions,
            # v1: no separation, no automatic transcription — these stay
            # at their empty defaults (stems=[], separation_model=None,
            # timestamped_lyrics=[], lyrics_*=None).
        )
        bundle.musicological_essay = _deterministic_summary(bundle, title)

        store.save_analysis(
            TrackAnalysisRecord(
                id=track_id,
                telemetry_json=bundle,
                created_at=datetime.now(timezone.utc),
            )
        )
        jobs.finish_stages(job_id)
        jobs.update_job(job_id, state="done", stage="Done", progress=100)
        return {"job_id": job_id, "track_id": track_id, "state": "done"}
    except Exception as exc:  # noqa: BLE001 - surfaced on the job record
        jobs.update_job(job_id, state="failed", stage="Failed", error=str(exc))
        raise
