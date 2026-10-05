"""Vocal production forensics — deterministic, heuristic, honestly labeled.

Measures fundamental-frequency behavior from a vocal stem (or a full mix,
with reduced reliability — callers should note the source):

- F0 mean / range via PYIN
- Vibrato rate + depth from F0 modulation in the 4-9 Hz band
- Pitch-correction ("autotune") likelihood from semitone-deviation
  clustering and abrupt pitch jumps

The autotune score is a *heuristic* (documented as such in the schema), not
a ground-truth detector: heavy vibrato, portamento, and certain vocal
styles can move it. The zero-hallucination rule applies here too — report
the number with its caveats, never as a verdict.
"""
from __future__ import annotations

import numpy as np

try:
    import librosa

    _HAVE_LIBROSA = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_LIBROSA = False


def _require_librosa() -> None:
    if not _HAVE_LIBROSA:
        raise RuntimeError("librosa is required for vocal analysis")


def _vibrato(cents: np.ndarray, sr: int, hop: int) -> tuple[float, float]:
    """Estimate vibrato rate (Hz) and depth (cents) from a voiced F0 contour.

    cents: F0 expressed in cents relative to its own median, voiced frames only.
    """
    if cents.size < 64:
        return 0.0, 0.0
    frame_rate = sr / hop
    x = cents - np.median(cents)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / frame_rate)
    band = (freqs >= 4.0) & (freqs <= 9.0)
    if not np.any(band) or np.max(spectrum[band]) < 1e-9:
        return 0.0, 0.0
    # Parabolic interpolation around the peak for sub-bin accuracy.
    peak = int(np.argmax(spectrum[band]))
    idx = np.flatnonzero(band)[peak]
    rate = float(freqs[idx])
    # Depth: RMS of the band-limited signal, converted peak amplitude.
    mask = np.zeros_like(spectrum)
    lo = max(0, idx - 2)
    hi = min(spectrum.size, idx + 3)
    mask[lo:hi] = 1.0
    filtered = np.fft.irfft(spectrum * mask * np.exp(1j * np.angle(np.fft.rfft(x))), n=x.size)
    depth = float(np.sqrt(2.0) * np.sqrt(np.mean(filtered**2)))
    return round(rate, 2), round(depth, 1)


def analyze_vocals(audio_path: str, sr: int = 22050) -> dict:
    """Analyze vocal production from an audio file.

    Best run on an isolated vocal stem; on a full mix the F0 tracker may
    lock onto the loudest pitched element instead.
    """
    _require_librosa()
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    if y.size == 0:
        return _empty()

    hop = 512
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop,
    )
    voiced = voiced_flag & ~np.isnan(f0)
    voiced_fraction = round(float(np.mean(voiced_flag)), 3)
    if int(np.sum(voiced)) < 32:
        result = _empty()
        result["voiced_fraction"] = voiced_fraction
        return result

    f0v = f0[voiced]
    f0_mean = float(np.mean(f0v))
    f0_range_st = round(float(12 * np.log2(np.max(f0v) / max(np.min(f0v), 1e-9))), 2)

    cents = 1200 * np.log2(f0v / np.median(f0v))
    vib_rate, vib_depth = _vibrato(cents, sr, hop)

    # Pitch-correction heuristic: corrected vocals cluster tightly around
    # semitone centers and jump abruptly between them. Deviation is measured
    # against absolute concert pitch (A440) — not the signal's own median —
    # so stepped notes read as "on the grid" regardless of the song's key.
    abs_cents = 1200 * np.log2(f0v / 440.0)
    dev = abs_cents - np.round(abs_cents / 100.0) * 100.0  # [-50, 50]
    tight_frac = float(np.mean(np.abs(dev) < 12.0))
    jumps = np.abs(np.diff(cents))
    jump_frac = float(np.mean(jumps[~np.isnan(jumps)] > 60.0)) if jumps.size else 0.0
    autotune = round(min(1.0, 0.65 * tight_frac + 0.35 * min(jump_frac * 6.0, 1.0)), 3)

    return {
        "f0_mean_hz": round(f0_mean, 1),
        "f0_range_semitones": f0_range_st,
        "vibrato_rate_hz": vib_rate,
        "vibrato_depth_cents": vib_depth,
        "autotune_likelihood": autotune,
        "voiced_fraction": voiced_fraction,
    }


def _empty() -> dict:
    return {
        "f0_mean_hz": 0.0,
        "f0_range_semitones": 0.0,
        "vibrato_rate_hz": 0.0,
        "vibrato_depth_cents": 0.0,
        "autotune_likelihood": 0.0,
        "voiced_fraction": 0.0,
    }
