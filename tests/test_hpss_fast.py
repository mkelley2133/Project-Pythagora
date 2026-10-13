"""Fast exact HPSS — equivalence tests.

The optimization contract: hpss_waveform must produce the SAME numbers
as librosa.effects.hpss (defaults) — only faster. The sliding-median
kernel is checked bit-for-bit against scipy.ndimage.median_filter, and
the composed HPSS against librosa itself on a synthetic signal.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import median_filter

from workers.analysis import audio as audio_mod


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shape", [(37, 53), (64, 200), (129, 33)])
def test_sliding_median_matches_scipy_exactly(shape, axis):
    rng = np.random.default_rng(7)
    a = rng.standard_normal(shape).astype(np.float32)
    size = [1, 1]
    size[axis] = audio_mod._HPSS_KERNEL
    ref = median_filter(a, size=size, mode="reflect")
    got = audio_mod._median_filter_axis(a, audio_mod._HPSS_KERNEL, axis)
    assert got.dtype == a.dtype
    assert np.array_equal(ref, got)  # bit-identical, not just close


def test_sliding_median_small_axis_falls_back():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 90)).astype(np.float32)
    ref = median_filter(a, size=(31, 1), mode="reflect")
    got = audio_mod._median_filter_axis(a, 31, 0)
    assert np.array_equal(ref, got)


def _synthetic_mix(seconds: float = 4.0, sr: int = 22050) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 330.0 * t)
    rng = np.random.default_rng(11)
    clicks = np.zeros_like(t)
    clicks[::sr // 2] = 0.8  # percussive ticks every half second
    return (tone + 0.05 * rng.standard_normal(t.size) + clicks).astype(np.float32)


def test_hpss_waveform_matches_librosa():
    librosa = pytest.importorskip("librosa")
    y = _synthetic_mix()
    h_ref, p_ref = librosa.effects.hpss(y)
    h_got, p_got = audio_mod.hpss_waveform(y)
    assert h_got.shape == h_ref.shape and p_got.shape == p_ref.shape
    assert float(np.abs(h_ref - h_got).max()) < 1e-4
    assert float(np.abs(p_ref - p_got).max()) < 1e-4


def test_core_features_share_unchanged_by_fast_hpss():
    librosa = pytest.importorskip("librosa")
    y = _synthetic_mix()
    feats = audio_mod.core_features(y, 22050)
    h_ref, p_ref = librosa.effects.hpss(y)
    rms_h = float(np.sqrt(np.mean(h_ref**2)))
    rms_p = float(np.sqrt(np.mean(p_ref**2)))
    expected = round(rms_p / (rms_h + rms_p + 1e-9), 3)
    assert feats["percussive_share"] == expected
    assert feats["harmonic_share"] == round(1 - expected, 3)
