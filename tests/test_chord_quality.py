"""Chord-quality regression tests for the harmonic-bleed correction.

Ground truth is exact by construction: additive triads with a full
harmonic series (1/h gains) and a controlled third gain, over a kick/hat
beat grid. The full series matters — a root's 5th harmonic is its major
third, and that bleed is what flipped weak minor thirds to major before
the correction in workers/analysis/chords.py (see _BLEED_BETA). A synth
with only 2-3 harmonics cannot reproduce the failure mode.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from workers.analysis.chords import recognize_chords

SR = 22050
_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9}


def _freq(pc: int, octave: int) -> float:
    return 440.0 * 2 ** ((12 * (octave + 1) + pc - 69) / 12)


def _triad(root: str, minor: bool, seconds: float, third_gain: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    r = _NOTE[root]
    third = (r + (3 if minor else 4)) % 12
    fifth = (r + 7) % 12
    y = np.zeros(n)
    for pc, gain, octv in ((r, 1.0, 3), (r, 0.6, 4), (fifth, 0.8, 4), (third, third_gain, 4)):
        f = _freq(pc, octv)
        for h in range(1, 7):
            y += gain * (1.0 / h) * np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6.28))
    env = np.minimum(1.0, np.minimum(t / 0.05, (seconds - t) / 0.05))
    y = y * env + rng.normal(0, 0.004, n)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.7).astype(np.float32)


def _progression(tmp_path, name, chords, third_gain, seed=100):
    parts = [
        _triad(root, minor, 2.0, third_gain, seed + i)
        for i, (root, minor) in enumerate(chords)
    ]
    y = np.concatenate(parts)
    # Beat grid (kick on beats, hat ticks off) so beat tracking locks on,
    # as it does on the drum-driven catalog this recognizer serves.
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
    path = tmp_path / name
    sf.write(str(path), y, SR)
    return str(path)


def _labels_at_midpoints(events, chords):
    out = []
    for i, (root, minor) in enumerate(chords):
        mid = i * 2.0 + 1.0
        ev = next((e for e in events if e["start"] <= mid < e["end"]), None)
        out.append(ev["chord"] if ev else None)
    return out


def test_weak_third_minor_progression_reads_minor(tmp_path):
    """At third gain 0.12 the pre-correction recognizer flipped Dm->D and
    Em->E (root 5th-harmonic bleed outweighing the real minor third)."""
    chords = [("A", True), ("D", True), ("E", True), ("A", True)]
    path = _progression(tmp_path, "weak_minor.wav", chords, third_gain=0.12)
    events = recognize_chords(path, key="A", mode="minor")
    assert _labels_at_midpoints(events, chords) == ["Am", "Dm", "Em", "Am"]


def test_weak_third_major_progression_stays_major(tmp_path):
    """The correction must not over-correct: equally weak major thirds
    (same gain) still read major."""
    chords = [("C", False), ("F", False), ("G", False), ("C", False)]
    path = _progression(tmp_path, "weak_major.wav", chords, third_gain=0.12)
    events = recognize_chords(path, key="C", mode="major")
    assert _labels_at_midpoints(events, chords) == ["C", "F", "G", "C"]


def test_strong_third_minor_progression_reads_minor(tmp_path):
    chords = [("A", True), ("D", True), ("E", True), ("A", True)]
    path = _progression(tmp_path, "strong_minor.wav", chords, third_gain=1.0)
    events = recognize_chords(path, key="A", mode="minor")
    assert _labels_at_midpoints(events, chords) == ["Am", "Dm", "Em", "Am"]
