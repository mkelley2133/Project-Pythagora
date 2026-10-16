"""Temporal-smoothing tests for chord recognition (added 2026-10-16).

Two layers:
- Unit tests of the two label-agnostic filters in workers/analysis/chords.py
  (_median_smooth_sims over the similarity matrix, _majority_smooth_labels
  over the Viterbi path) on synthetic sequences with exact expected outputs.
- An end-to-end regression for the documented trap failure mode: the third
  of a sustained tonic triad alternating minor/major every beat (the same
  flicker family as the human-verified Cm/C alternation behind _BLEED_BETA).
  The recognizer must report one sustained chord region, not a beat-by-beat
  alternation — while a genuine chord change right after it still lands.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from workers.analysis.chords import (
    _majority_smooth_labels,
    _median_smooth_sims,
    recognize_chords,
)

SR = 22050
_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9}


# --------------------------------------------------------------------------
# _median_smooth_sims
# --------------------------------------------------------------------------

def test_median_smooth_kills_single_segment_evidence_island():
    # One segment of contrary evidence inside a sustained region: the
    # median of each 3-window outvotes it (the filter's actual contract;
    # a perfectly periodic alternation is the Viterbi penalty's job, and
    # the end-to-end flicker test below covers the realistic chain).
    sims = np.array(
        [[0.9, 0.2], [0.9, 0.2], [0.2, 0.9], [0.9, 0.2], [0.9, 0.2]]
    )
    out = _median_smooth_sims(sims)
    assert np.argmax(out, axis=1).tolist() == [0, 0, 0, 0, 0]


def test_median_smooth_preserves_a_sustained_step():
    sims = np.array([[0.9, 0.2]] * 3 + [[0.2, 0.9]] * 3)
    out = _median_smooth_sims(sims)
    assert np.argmax(out, axis=1).tolist() == [0, 0, 0, 1, 1, 1]


def test_median_smooth_does_not_mutate_input():
    sims = np.array([[0.9, 0.2], [0.2, 0.9], [0.9, 0.2]])
    before = sims.copy()
    _median_smooth_sims(sims)
    assert np.array_equal(sims, before)


def test_median_smooth_short_input_passthrough():
    sims = np.array([[0.9, 0.2], [0.2, 0.9]])
    out = _median_smooth_sims(sims)
    assert np.allclose(out, sims)


# --------------------------------------------------------------------------
# _majority_smooth_labels
# --------------------------------------------------------------------------

def test_majority_removes_single_segment_island():
    labels = np.array([0, 0, 1, 0, 0])
    assert _majority_smooth_labels(labels).tolist() == [0, 0, 0, 0, 0]


def test_majority_preserves_sustained_change():
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert _majority_smooth_labels(labels).tolist() == [0, 0, 0, 1, 1, 1]


def test_majority_preserves_two_segment_event():
    labels = np.array([0, 0, 1, 1, 0, 0])
    out = _majority_smooth_labels(labels)
    assert out[2] == 1 and out[3] == 1


def test_majority_short_input_passthrough():
    labels = np.array([1, 0])
    assert _majority_smooth_labels(labels).tolist() == [1, 0]


# --------------------------------------------------------------------------
# End-to-end: third-flicker over a sustained tonic (the trap failure mode)
# --------------------------------------------------------------------------

def _freq(pc: int, octave: int) -> float:
    return 440.0 * 2 ** ((12 * (octave + 1) + pc - 69) / 12)


def _triad_block(root: str, minor: bool, seconds: float, seed: int) -> np.ndarray:
    """One full-harmonic-series triad block (same synth family as
    tests/test_chord_quality.py — the harmonic series is what makes the
    major/minor decision non-trivial)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    r = _NOTE[root]
    third = (r + (3 if minor else 4)) % 12
    fifth = (r + 7) % 12
    y = np.zeros(n)
    for pc, gain, octv in ((r, 1.0, 3), (r, 0.6, 4), (fifth, 0.8, 4), (third, 1.0, 4)):
        f = _freq(pc, octv)
        for h in range(1, 7):
            y += gain * (1.0 / h) * np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6.28))
    env = np.minimum(1.0, np.minimum(t / 0.03, (seconds - t) / 0.03))
    y = y * env + rng.normal(0, 0.004, n)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.7).astype(np.float32)


def _with_beat_grid(y: np.ndarray) -> np.ndarray:
    """Kick on every 0.5 s beat + hat ticks off-beat, as in the catalog."""
    rng = np.random.default_rng(7)
    beat = int(SR * 0.5)
    for start in range(0, len(y), beat):
        n = min(int(0.12 * SR), len(y) - start)
        t = np.arange(n) / SR
        f = 120.0 - 70.0 * (t / (n / SR))
        y[start:start + n] += 0.55 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 25)
        off = start + beat // 2
        if off < len(y):
            m = min(int(0.03 * SR), len(y) - off)
            y[off:off + m] += 0.12 * rng.normal(0, 1, m) * np.exp(-np.arange(m) / SR * 120)
    return y


def test_third_flicker_reads_as_one_sustained_chord(tmp_path):
    # 4 s of A tonic whose third alternates minor/major every beat,
    # then a genuine change: 2 s of steady F major.
    blocks = [
        _triad_block("A", minor=(i % 2 == 0), seconds=0.5, seed=200 + i)
        for i in range(8)
    ]
    blocks.append(_triad_block("F", minor=False, seconds=2.0, seed=300))
    y = _with_beat_grid(np.concatenate(blocks))
    path = tmp_path / "flicker.wav"
    sf.write(str(path), y, SR)

    events = recognize_chords(str(path), key="A", mode="minor")

    # The flicker region must not shatter into beat-length events.
    flicker_events = [e for e in events if e["start"] >= 0.1 and e["end"] <= 3.9]
    assert all(e["end"] - e["start"] >= 1.0 for e in flicker_events), events
    assert len(events) <= 4, events
    # Whichever quality wins the flicker, the region is the A tonic…
    mid = next(e for e in events if e["start"] <= 2.0 < e["end"])
    assert mid["chord"] in ("Am", "A"), events
    # …and the genuine change to F still lands.
    late = next(e for e in events if e["start"] <= 5.0 < e["end"])
    assert late["chord"] == "F", events
