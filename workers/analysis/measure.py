"""Mix measurements for mix suggestions (Phase G).

Deterministic signal measurements only — band energy shares, true sample
levels (dBFS), and stereo correlation/width. These feed the rule engine in
workers/analysis/suggestions.py; nothing here interprets or recommends.
"""
from __future__ import annotations

import numpy as np

try:
    import librosa

    _HAVE_LIBROSA = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_LIBROSA = False

# Band edges in Hz: sub / low / mid / high / air.
_BANDS = [
    ("sub_pct", 20.0, 60.0),
    ("low_pct", 60.0, 250.0),
    ("mid_pct", 250.0, 2000.0),
    ("high_pct", 2000.0, 8000.0),
    ("air_pct", 8000.0, float("inf")),
]

_EPS = 1e-12


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), _EPS)))


def band_shares(y: np.ndarray, sr: int) -> dict[str, float]:
    """Percentage of total spectral energy per band (STFT magnitude sums)."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    total = float(S.sum()) + 1e-9
    out: dict[str, float] = {}
    for name, lo, hi in _BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = round(float(S[mask].sum() / total * 100.0), 1)
    return out


def level_stats(y: np.ndarray) -> dict[str, float]:
    """True peak / RMS / crest from the samples themselves (dBFS)."""
    if y.size == 0:
        return {"peak_dbfs": -120.0, "rms_dbfs": -120.0, "crest_db": 0.0}
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    peak_db = _db(peak)
    rms_db = _db(rms)
    return {
        "peak_dbfs": round(peak_db, 1),
        "rms_dbfs": round(rms_db, 1),
        "crest_db": round(max(0.0, peak_db - rms_db), 1),
    }


def stereo_stats(y_multi: np.ndarray) -> dict:
    """Inter-channel correlation + side/mid width for a (channels, samples)
    array. Mono input returns is_stereo=False with None stats."""
    if y_multi.ndim != 2 or y_multi.shape[0] < 2:
        return {"is_stereo": False, "stereo_correlation": None, "stereo_width": None}
    left = y_multi[0].astype(np.float64)
    right = y_multi[1].astype(np.float64)
    n = min(left.size, right.size)
    if n == 0:
        return {"is_stereo": False, "stereo_correlation": None, "stereo_width": None}
    left, right = left[:n], right[:n]
    denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    corr = float(np.sum(left * right) / denom) if denom > _EPS else 1.0
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    mid_rms = float(np.sqrt(np.mean(mid**2)))
    side_rms = float(np.sqrt(np.mean(side**2)))
    width = float(side_rms / mid_rms) if mid_rms > _EPS else 0.0
    return {
        "is_stereo": True,
        "stereo_correlation": round(max(-1.0, min(1.0, corr)), 3),
        "stereo_width": round(width, 3),
    }


def measure_mix(audio_path: str, sr: int = 22050) -> dict:
    """Full measurement pass over an audio file (mono + stereo loads)."""
    if not _HAVE_LIBROSA:
        raise RuntimeError("librosa is required for mix measurements")
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    out: dict = {}
    out.update(band_shares(y, sr))
    out.update(level_stats(y))
    try:
        y_multi, _ = librosa.load(audio_path, sr=sr, mono=False)
        if y_multi.ndim == 1:
            y_multi = y_multi[np.newaxis, :]
        out.update(stereo_stats(y_multi))
    except Exception:  # noqa: BLE001 - stereo is best-effort
        out.update({"is_stereo": False, "stereo_correlation": None, "stereo_width": None})
    return out
