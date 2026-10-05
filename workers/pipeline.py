"""Analysis pipeline: audio file -> TelemetryBundle, with job progress.

Stages are deterministic measurements (the ported local analyzer + the new
workers/analysis modules). Anything that cannot be measured is skipped or
left empty — never invented. Transcription is best-effort: when
faster-whisper isn't installed the stage is skipped and the job still
completes.
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
    LyricLine,
    StructuralSegment,
    SubGenre,
    TelemetryBundle,
    TrackAnalysisRecord,
    VocalProfile,
)
from workers.analysis import audio as audio_mod
from workers.analysis.chords import recognize_chords
from workers.analysis.lyrics import rhyme_scheme
from workers.analysis.transcribe import transcribe
from workers.analysis.vocals import analyze_vocals
from workers.analysis.waveform import waveform_peaks


def _ensure_wav(audio_path: str) -> str:
    """Return a wav path librosa can read; convert via ffmpeg when needed."""
    p = Path(audio_path)
    if p.suffix.lower() == ".wav":
        return str(p)
    if shutil.which("ffmpeg"):
        out = p.with_suffix(".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-ar", "22050", "-ac", "1", str(out)],
            check=True,
        )
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
    return (
        f"Measured analysis of '{track_title}': {t.bpm:.1f} BPM in "
        f"{t.musical_key} {t.mode} ({t.duration:.0f}s). Harmonic movement "
        f"centers on {chords}. Structure: {segs or 'not segmented'}. "
        f"Rhyme scheme {t.rhyme_scheme_summary or 'not transcribed'}. "
        "Every figure above is a deterministic measurement, not an interpretation."
    )


def run_pipeline(job_id: str, track_id: str, audio_path: str) -> dict:
    def stage(label: str, pct: int) -> None:
        jobs.update_job(job_id, state="running", stage=label, progress=pct)

    try:
        stage("Loading audio", 5)
        wav_path = _ensure_wav(audio_path)
        y, sr = audio_mod.load_audio(wav_path)

        stage("Measuring tempo & key", 18)
        tempo = audio_mod.estimate_tempo(y, sr) or 0.0
        key = audio_mod.estimate_key(y, sr)
        key_name, mode = key["key"].rsplit(" ", 1)

        stage("Recognizing chords", 32)
        chord_events = [
            ChordEvent(**e) for e in recognize_chords(wav_path, key=key_name, mode=mode)
        ]

        stage("Mapping sections & texture", 48)
        feats = audio_mod.core_features(y, sr)
        sections = [
            StructuralSegment(name=s.get("role") or f"Section {i + 1}", start=float(s["start"]), end=float(s["end"]))
            for i, s in enumerate(audio_mod.section_map(y, sr))
        ]
        mood = audio_mod.mood_label(feats, tempo or None, key)
        styles = audio_mod.style_tags(feats, tempo or None, key)

        stage("Rendering waveform", 62)
        peaks = waveform_peaks(wav_path)

        stage("Vocal forensics", 70)
        vox = analyze_vocals(wav_path)
        vocal_profile = None
        # Refuse instead of guessing: a sub-80 Hz median F0 is bass, not voice.
        if vox["voiced_fraction"] > 0.15 and vox["f0_mean_hz"] > 80:
            vocal_profile = VocalProfile(**vox)

        stage("Transcribing lyrics", 80)
        lyric_lines: list[LyricLine] = []
        rhyme_summary = ""
        tx = transcribe(wav_path)
        if tx.get("available") and tx.get("lines"):
            texts = [seg["text"] for seg in tx["lines"]]
            schemes = rhyme_scheme(texts)
            letters = list(schemes[0]["scheme"]) if schemes else []
            for i, seg in enumerate(tx["lines"]):
                lyric_lines.append(
                    LyricLine(
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        text=seg["text"],
                        rhyme=letters[i] if i < len(letters) else "",
                    )
                )
            rhyme_summary = schemes[0]["scheme"] if schemes else ""

        stage("Finalizing", 94)
        track = store.get_track(track_id)
        title = track.title if track else track_id
        emotional_profile = {
            _slug(mood["mood"]): 0.9,
            "arousal": mood["arousal"],
            "valence": mood["valence"],
        }
        bundle = TelemetryBundle(
            bpm=round(float(tempo), 1),
            duration=round(float(feats.get("duration", 0.0)), 1),
            musical_key=key_name,
            mode=mode,
            sub_genres=[
                SubGenre(name=s["tag"], confidence=round(0.8 - 0.1 * i, 2))
                for i, s in enumerate(styles[:4])
            ],
            emotional_profile=emotional_profile,
            chords=[e.chord for e in chord_events[:24]],
            chord_timeline=chord_events,
            segments=sections,
            vocal_profile=vocal_profile,
            waveform_peaks=peaks,
            rhyme_scheme_summary=rhyme_summary,
            timestamped_lyrics=lyric_lines,
        )
        bundle.musicological_essay = _deterministic_summary(bundle, title)

        store.save_analysis(
            TrackAnalysisRecord(
                id=track_id,
                telemetry_json=bundle,
                created_at=datetime.now(timezone.utc),
            )
        )
        jobs.update_job(job_id, state="done", stage="Done", progress=100)
        return {"job_id": job_id, "track_id": track_id, "state": "done"}
    except Exception as exc:  # noqa: BLE001 - surfaced on the job record
        jobs.update_job(job_id, state="failed", stage="Failed", error=str(exc))
        raise
