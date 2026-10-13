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


def _brute_reflect_median(a, k, axis):
    """Ground truth: sliding median with repeated half-sample
    reflection, computed element by element. (scipy's median_filter
    cannot be the oracle for short axes — see the test below.)"""
    n = a.shape[axis]
    half = k // 2

    def reflect(i):
        while i < 0 or i >= n:
            i = -i - 1 if i < 0 else 2 * n - 1 - i
        return i

    out = np.empty_like(a)
    for pos in range(n):
        idx = [reflect(pos + d - half) for d in range(k)]
        window = np.take(a, idx, axis=axis)
        out[(pos, slice(None)) if axis == 0 else (slice(None), pos)] = np.median(window, axis=axis)
    return out


def test_sliding_median_small_axis_falls_back():
    # Regression (root-caused 2026-10-13): this (2, 90) / k=31 case
    # failed intermittently for months because BOTH fast paths are
    # invalid here — the numba kernel reflects edges only once (needs
    # axis > k//2), and scipy's median_filter returns heap-dependent
    # garbage when a 31-wide footprint hits a length-2 axis with
    # mode="reflect" (57-67 of 180 cells wrong vs brute force, varying
    # by process heap state). _median_filter_axis now computes short
    # axes exactly; the oracle is brute force, not scipy.
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 90)).astype(np.float32)
    ref = _brute_reflect_median(a, 31, 0)
    got = audio_mod._median_filter_axis(a, 31, 0)
    assert np.array_equal(ref, got)


def test_sliding_median_short_axes_match_exact():
    # Short axes (<= k//2) take the exact gather path; longer ones stay
    # on the kernel. All must equal the brute-force reflected median.
    rng = np.random.default_rng(5)
    for shape in [(2, 90), (5, 40), (15, 20), (16, 64)]:
        for axis in (0, 1):
            a = rng.standard_normal(shape).astype(np.float32)
            ref = _brute_reflect_median(a, audio_mod._HPSS_KERNEL, axis)
            got = audio_mod._median_filter_axis(a, audio_mod._HPSS_KERNEL, axis)
            assert np.array_equal(ref, got), (shape, axis)


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
