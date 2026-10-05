"""Waveform peak envelopes for the dashboard timeline.

Computes a downsampled peak-amplitude envelope so the frontend can render
the track's *real* waveform instead of the seeded placeholder it uses when
no peak data is present.
"""
from __future__ import annotations

import numpy as np

try:
    import librosa

    _HAVE_LIBROSA = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_LIBROSA = False


def waveform_peaks(audio_path: str, n_peaks: int = 600, sr: int = 22050) -> list[float]:
    """Return n_peaks normalized peak amplitudes in [0, 1]."""
    if not _HAVE_LIBROSA:
        raise RuntimeError("librosa is required for waveform peaks")
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    if y.size == 0:
        return [0.0] * n_peaks

    frame_length, hop_length = 2048, 1024
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    env = np.abs(frames).max(axis=0)
    peak = float(np.max(env))
    if peak < 1e-9:
        return [0.0] * n_peaks
    env = env / peak

    # Resample the envelope to exactly n_peaks points.
    xs_old = np.linspace(0.0, 1.0, env.size)
    xs_new = np.linspace(0.0, 1.0, n_peaks)
    resampled = np.interp(xs_new, xs_old, env)
    return [round(float(v), 4) for v in resampled]
