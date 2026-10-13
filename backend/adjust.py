"""Semitone + speed adjustment (Phase H) — server-side rendered export.

Pitch shift and time stretch are applied independently with librosa
(already a project dependency — no model downloads): pitch_shift preserves
tempo, time_stretch preserves pitch. The original file and its analysis
are never touched; renders are cached under the storage dir keyed by
(track, semitones, speed) where the numbers are server-formatted floats,
so no user input ever reaches a filename.

Derived readouts (effective BPM / transposed key) are computed from the
original analysis values and labeled as derived wherever shown.
"""
from __future__ import annotations

from pathlib import Path

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SEMITONE_MIN, SEMITONE_MAX = -12.0, 12.0
SPEED_MIN, SPEED_MAX = 0.5, 2.0


def clamp_adjust(semitones: float, speed: float) -> tuple[float, float]:
    """Clamp request values into the supported ranges."""
    try:
        st = float(semitones)
    except (TypeError, ValueError):
        st = 0.0
    try:
        sp = float(speed)
    except (TypeError, ValueError):
        sp = 1.0
    st = min(SEMITONE_MAX, max(SEMITONE_MIN, st))
    sp = min(SPEED_MAX, max(SPEED_MIN, sp))
    return round(st, 2), round(sp, 3)


def transpose_key(key: str, semitones: float) -> str:
    """Transpose a pitch-class name by (possibly fractional) semitones.

    Fractional shifts land between pitch classes; the readout rounds to
    the nearest class and the caller labels it derived/approximate.
    """
    if key not in PITCH_NAMES:
        return key
    steps = int(round(float(semitones)))
    return PITCH_NAMES[(PITCH_NAMES.index(key) + steps) % 12]


def effective_bpm(bpm: float, speed: float) -> float:
    return round(float(bpm) * float(speed), 1)


def cache_filename(track_id: str, semitones: float, speed: float) -> str:
    """Cache name built ONLY from the server-validated track id and
    server-formatted floats — no raw user strings."""
    return f"{track_id}_st{semitones:+.2f}_sp{speed:.3f}.wav"


def render_adjusted(
    src_path: Path, dest: Path, semitones: float, speed: float
) -> Path:
    """Render the adjusted audio to `dest` (wav). Identity adjustments are
    still rendered so the export path is uniform."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(str(src_path), sr=22050, mono=True)
    if speed != 1.0:
        y = librosa.effects.time_stretch(y, rate=float(speed))
    if semitones != 0.0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitones))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), y, sr, subtype="PCM_16")
    return dest
