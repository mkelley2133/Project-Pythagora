"""Optional LLM narrative for the analysis report — strictly optional.

The report's Summary prose is normally the pipeline's deterministic,
template-built essay (workers/pipeline.py `_deterministic_summary`),
written from the locked TelemetryBundle. This module offers an
alternative rendering of that same prose: WHEN an LLM API key is
present in the environment, an OpenAI-compatible chat model rewrites
the summary from a fact sheet built ONLY out of the locked bundle.

Zero-hallucination contract (the project's design rule, applied to the
one place an LLM is allowed to touch text):
- The model receives telemetry values and nothing else. The system
  prompt forbids inventing, estimating, or altering any measurement,
  and forbids introducing facts that are not in the fact sheet.
- The telemetry itself is never modified here — this module runs at
  report-render time, downstream of the telemetry lock, and returns
  prose only.
- With no key configured, behavior is EXACTLY the pre-existing one:
  the deterministic essay, no network call. Any failure talking to
  the provider (network, HTTP, malformed response, empty text) also
  falls back to the deterministic essay.

Configuration (environment only, following the genre/subgenre
modules' env-override pattern):
- PYTHAGORA_LLM_API_KEY   — required to enable; unset = disabled.
- PYTHAGORA_LLM_API_BASE  — default https://api.openai.com/v1
- PYTHAGORA_LLM_MODEL     — default gpt-4o-mini
"""
from __future__ import annotations

import os

import httpx

from backend.models.schemas import TelemetryBundle, TrackRecord

API_KEY_ENV = "PYTHAGORA_LLM_API_KEY"
API_BASE_ENV = "PYTHAGORA_LLM_API_BASE"
MODEL_ENV = "PYTHAGORA_LLM_MODEL"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 30.0

SYSTEM_PROMPT = (
    "You are a musicologist writing a short summary of a song analysis. "
    "Use ONLY the measured facts in the user message. Never invent, "
    "estimate, round differently, or alter any number, chord, key, "
    "tempo, genre, or score, and do not introduce any fact that is not "
    "in the fact sheet. If a topic is absent from the fact sheet, omit "
    "it entirely. Write two short paragraphs of plain prose, no "
    "markdown, no bullet lists, no headline."
)


def llm_enabled() -> bool:
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def telemetry_facts(track: TrackRecord, telemetry: TelemetryBundle) -> str:
    """A compact fact sheet containing ONLY locked bundle values."""
    t = telemetry
    lines = [
        f"Track: {track.title} by {track.artist}",
        f"Duration: {t.duration:.1f} seconds",
        f"Tempo: {t.bpm:.1f} BPM",
        f"Key: {t.musical_key} {t.mode}"
        + (f" (key confidence {t.key_confidence:.3f})" if t.key_confidence is not None else ""),
    ]
    if t.primary_genre:
        lines.append(f"Primary genre (trained model): {t.primary_genre}")
    for pred in t.genre_predictions[:3]:
        lines.append(f"Genre candidate: {pred.label} (model probability {pred.probability:.4f})")
    if t.primary_subgenre:
        lines.append(f"Primary subgenre match: {t.primary_subgenre}")
    for pred in t.subgenre_predictions[:3]:
        lines.append(
            f"Subgenre similarity: {pred.label} ({pred.family}) — similarity share {pred.percent:.1f}% "
            "(a similarity share, not a probability)"
        )
    if t.primary_emotion:
        lines.append(f"Primary emotion match: {t.primary_emotion}")
    for pred in t.emotion_predictions[:3]:
        lines.append(
            f"Emotion similarity: {pred.label} — zero-shot emotion similarity share {pred.percent:.1f}% "
            "(a similarity share, not a probability, and not a claim about what the track feels)"
        )
    unique_chords = list(dict.fromkeys(e.chord for e in t.chord_timeline)) or t.chords
    if unique_chords:
        lines.append(f"Chords used: {', '.join(unique_chords)}")
    if t.chord_timeline:
        lines.append(f"Chord events: {len(t.chord_timeline)}")
    for seg in t.segments[:6]:
        lines.append(f"Section: {seg.name} from {seg.start:.1f}s to {seg.end:.1f}s")
    if t.vocal_profile is not None:
        v = t.vocal_profile
        lines.append(
            f"Vocals: mean pitch {v.f0_mean_hz:.1f} Hz, pitch range {v.f0_range_semitones:.1f} "
            f"semitones, voiced fraction {v.voiced_fraction:.2f}"
        )
    elif t.vocal_note:
        lines.append(f"Vocals: not measured — {t.vocal_note}")
    if t.mix_measurements is not None:
        m = t.mix_measurements
        lines.append(
            f"Mix: peak {m.peak_dbfs:.1f} dBFS, RMS {m.rms_dbfs:.1f} dBFS, "
            f"sub-bass energy share {m.sub_pct:.1f}%"
        )
    if t.suggestions:
        lines.append(f"Rule-based mix suggestions issued: {len(t.suggestions)}")
    return "\n".join(lines)


def build_narrative(
    track: TrackRecord, telemetry: TelemetryBundle
) -> tuple[str, str]:
    """Return (prose, source) for the report's Summary section.

    source is "llm" only when an LLM actually produced the text;
    "deterministic" when no key is configured (the default, identical
    to the historical behavior); "deterministic-fallback" when a key
    is configured but the call failed or returned nothing usable.
    """
    fallback = telemetry.musicological_essay
    if not llm_enabled():
        return fallback, "deterministic"
    api_base = os.environ.get(API_BASE_ENV, "").strip() or DEFAULT_API_BASE
    model = os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
    try:
        response = httpx.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ[API_KEY_ENV]}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": telemetry_facts(track, telemetry)},
                ],
                "temperature": 0.3,
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        if text:
            return text, "llm"
    except Exception:  # noqa: BLE001 - fallback is the contract
        pass
    return fallback, "deterministic-fallback"
