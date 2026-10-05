"""Tests for the deterministic measurement workers.

All audio is synthesized with known ground truth (exact chords, exact
vibrato rate, stepped vs. gliding pitch), so every assertion checks a
measurement against a fact — the same standard the project holds itself to.
"""
import numpy as np
import pytest
from scipy.io import wavfile

from workers.analysis.chords import recognize_chords, roman_numeral
from workers.analysis.vocals import analyze_vocals
from workers.analysis.waveform import waveform_peaks

SR = 22050


def _write(path, y):
    y = np.asarray(y, dtype=np.float64)
    y = y / max(1e-9, np.max(np.abs(y))) * 0.8
    wavfile.write(str(path), SR, y.astype(np.float32))


def _triad(freqs, seconds):
    t = np.arange(int(SR * seconds)) / SR
    return sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)


def test_roman_numerals():
    assert roman_numeral("C", "C", "major") == "I"
    assert roman_numeral("F", "C", "major") == "IV"
    assert roman_numeral("G", "C", "major") == "V"
    assert roman_numeral("Am", "C", "major") == "vi"
    assert roman_numeral("Dm", "D", "minor") == "i"
    assert roman_numeral("Bb", "D", "minor") == "VI"


def test_chord_recognition_progression(tmp_path):
    # C - F - G, 2 seconds each, pure sine triads.
    seq = [
        ("C", [261.63, 329.63, 392.00]),
        ("F", [174.61, 220.00, 261.63]),
        ("G", [196.00, 246.94, 392.00]),
    ]
    y = np.concatenate([_triad(freqs, 2.0) for _, freqs in seq])
    p = tmp_path / "prog.wav"
    _write(p, y)

    events = recognize_chords(str(p), key="C", mode="major")
    found = {e["chord"] for e in events}
    assert {"C", "F", "G"} <= found, f"missed chords, got {found}"
    assert all(e["confidence"] > 0.3 for e in events)
    romans = {e["chord"]: e["roman"] for e in events}
    assert romans["C"] == "I"
    assert romans["F"] == "IV"
    assert romans["G"] == "V"
    # Events are contiguous and ordered.
    for a, b in zip(events, events[1:]):
        assert b["start"] >= a["start"]


def test_vibrato_detection(tmp_path):
    # 220 Hz carrier with 6 Hz FM at ~50 cents depth.
    fm_rate, depth_cents = 6.0, 50.0
    t = np.arange(int(SR * 2.0)) / SR
    delta_f = 220.0 * (2 ** (depth_cents / 1200.0) - 1)
    beta = delta_f / fm_rate
    y = np.sin(2 * np.pi * 220.0 * t + beta * np.sin(2 * np.pi * fm_rate * t))
    p = tmp_path / "vib.wav"
    _write(p, y)

    res = analyze_vocals(str(p))
    assert res["voiced_fraction"] > 0.8
    assert abs(res["vibrato_rate_hz"] - fm_rate) < 1.5, res
    assert abs(res["vibrato_depth_cents"] - depth_cents) < 25, res
    assert 200 < res["f0_mean_hz"] < 240


def test_autotune_heuristic_stepped_vs_glide(tmp_path):
    # Stepped pitch (autotune-like): constant notes with hard jumps.
    stepped = np.concatenate(
        [np.sin(2 * np.pi * f * np.arange(int(SR * 0.4)) / SR) for f in (220.0, 246.94, 261.63, 293.66)]
    )
    p1 = tmp_path / "stepped.wav"
    _write(p1, stepped)

    # Smooth glide: linear chirp 200 -> 320 Hz.
    t = np.arange(int(SR * 2.0)) / SR
    phase = 2 * np.pi * (200 * t + (120 / 2 / 2.0) * t**2)
    glide = np.sin(phase)
    p2 = tmp_path / "glide.wav"
    _write(p2, glide)

    s = analyze_vocals(str(p1))
    g = analyze_vocals(str(p2))
    assert s["autotune_likelihood"] > g["autotune_likelihood"], (s, g)
    assert s["autotune_likelihood"] > 0.5


def test_waveform_peaks(tmp_path):
    t = np.arange(int(SR * 3.0)) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # slow swell
    y = env * np.sin(2 * np.pi * 220.0 * t)
    p = tmp_path / "env.wav"
    _write(p, y)

    peaks = waveform_peaks(str(p), n_peaks=600)
    assert len(peaks) == 600
    assert all(0.0 <= v <= 1.0 for v in peaks)
    assert max(peaks) == pytest.approx(1.0, abs=0.01)
    # env = 0.5 + 0.5*sin(pi*t): loud near t=0.5s and t=2.5s, quiet near t=1.5s.
    loud = np.mean(peaks[75:125])    # t ~= 0.375-0.625s
    quiet = np.mean(peaks[275:325])  # t ~= 1.375-1.625s
    assert loud > quiet
