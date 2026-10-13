"""Track comparison (Phase E): two telemetry bundles -> aligned rows.

Every row carries both measured values plus a `different` flag so the
dashboard and the printed comparison report highlight real differences —
and only real differences. Nothing is estimated here that wasn't already
measured by the pipeline.
"""
from __future__ import annotations

from backend.models.schemas import TelemetryBundle, TrackRecord


def _key_label(t: TelemetryBundle) -> str:
    return f"{t.musical_key} {t.mode}"


def _level(t: TelemetryBundle, field: str) -> float | None:
    if t.mix_measurements is None:
        return None
    return getattr(t.mix_measurements, field)


def _vocal_summary(t: TelemetryBundle) -> str:
    if t.vocal_profile is not None:
        v = t.vocal_profile
        return (
            f"F0 mean {v.f0_mean_hz:.0f} Hz · range {v.f0_range_semitones:.1f} st · "
            f"voiced {v.voiced_fraction:.0%} · autotune heuristic {v.autotune_likelihood:.2f}"
        )
    return t.vocal_note or "Not measured"


def build_comparison(
    track_a: TrackRecord,
    bundle_a: TelemetryBundle,
    track_b: TrackRecord,
    bundle_b: TelemetryBundle,
) -> dict:
    rows: list[dict] = []

    def row(field, label, a, b, *, delta=None, fmt=None):
        rows.append(
            {
                "field": field,
                "label": label,
                "a": a,
                "b": b,
                "delta": delta,
                "different": a != b,
            }
        )

    row("duration", "Duration (s)", bundle_a.duration, bundle_b.duration,
        delta=round(bundle_b.duration - bundle_a.duration, 1))
    row("bpm", "Tempo (BPM)", bundle_a.bpm, bundle_b.bpm,
        delta=round(bundle_b.bpm - bundle_a.bpm, 1))
    row(
        "bpm_readings",
        "Half / double-time readings (BPM)",
        f"{bundle_a.bpm / 2:.1f} / {bundle_a.bpm * 2:.1f}",
        f"{bundle_b.bpm / 2:.1f} / {bundle_b.bpm * 2:.1f}",
    )
    row("key", "Key", _key_label(bundle_a), _key_label(bundle_b))
    row("key_confidence", "Key confidence", bundle_a.key_confidence, bundle_b.key_confidence)
    row("peak_dbfs", "Peak (dBFS)", _level(bundle_a, "peak_dbfs"), _level(bundle_b, "peak_dbfs"))
    row("rms_dbfs", "RMS (dBFS)", _level(bundle_a, "rms_dbfs"), _level(bundle_b, "rms_dbfs"))
    row("crest_db", "Crest / dynamics (dB)", _level(bundle_a, "crest_db"), _level(bundle_b, "crest_db"))
    row("sub_pct", "Sub-bass share (%)", _level(bundle_a, "sub_pct"), _level(bundle_b, "sub_pct"))
    row("segments", "Sections", len(bundle_a.segments), len(bundle_b.segments),
        delta=len(bundle_b.segments) - len(bundle_a.segments))

    chords_a = list(dict.fromkeys(e.chord for e in bundle_a.chord_timeline)) or bundle_a.chords
    chords_b = list(dict.fromkeys(e.chord for e in bundle_b.chord_timeline)) or bundle_b.chords
    row("unique_chords", "Unique chords", len(chords_a), len(chords_b),
        delta=len(chords_b) - len(chords_a))
    row("vocal", "Vocal forensics", _vocal_summary(bundle_a), _vocal_summary(bundle_b))
    row("suggestions", "Mix suggestions", len(bundle_a.suggestions), len(bundle_b.suggestions),
        delta=len(bundle_b.suggestions) - len(bundle_a.suggestions))

    shared = [c for c in chords_a if c in chords_b]
    return {
        "a": {"id": track_a.id, "title": track_a.title, "artist": track_a.artist},
        "b": {"id": track_b.id, "title": track_b.title, "artist": track_b.artist},
        "rows": rows,
        "chords": {
            "a": chords_a,
            "b": chords_b,
            "shared": shared,
            "only_a": [c for c in chords_a if c not in chords_b],
            "only_b": [c for c in chords_b if c not in chords_a],
        },
        "summary": {
            "fields_compared": len(rows),
            "fields_different": sum(1 for r in rows if r["different"]),
        },
    }
