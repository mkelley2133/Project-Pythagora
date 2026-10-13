"""Phase-3 analysis upgrades: Viterbi chord smoothing, beat-sync stability,
and the vocal separation refusal guard."""
import io
import wave

import numpy as np

from workers.analysis.chords import _viterbi_smooth, recognize_chords
from workers.analysis.vocals import _separation_verdict, analyze_vocals


def _write_wav(path, y, sr=22050):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return str(path)


# ------------------------------------------------------------ Viterbi unit


def test_viterbi_suppresses_single_segment_flip():
    # Label 0 wins everywhere except one segment where label 1 is slightly
    # stronger — a flip there costs two switches, so it must be suppressed.
    sims = np.array(
        [
            [0.90, 0.10],
            [0.90, 0.10],
            [0.45, 0.55],
            [0.90, 0.10],
            [0.90, 0.10],
        ]
    )
    path = _viterbi_smooth(sims, transition_penalty=0.30)
    assert list(path) == [0, 0, 0, 0, 0]


def test_viterbi_allows_sustained_change():
    sims = np.array(
        [
            [0.90, 0.10],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.10, 0.90],
            [0.10, 0.90],
        ]
    )
    path = _viterbi_smooth(sims, transition_penalty=0.30)
    assert list(path) == [0, 0, 1, 1, 1]


# ------------------------------------------------------- separation guard


def test_separation_verdict_refuses_sub_bass_lock():
    ok, reason = _separation_verdict(55.0, 0.92)
    assert ok is False
    assert "Insufficient separation" in reason


def test_separation_verdict_refuses_bass_dominated_low_line():
    ok, _ = _separation_verdict(100.0, 0.60)
    assert ok is False


def test_separation_verdict_accepts_vocal_range_line():
    ok, reason = _separation_verdict(220.0, 0.05)
    assert ok is True
    assert reason is None


def test_analyze_vocals_refuses_pure_sub_tone(tmp_path):
    sr = 22050
    t = np.arange(3 * sr) / sr
    y = 0.6 * np.sin(2 * np.pi * 55.0 * t)
    result = analyze_vocals(_write_wav(tmp_path / "sub.wav", y))
    assert result["separation_ok"] is False
    assert result["refusal_reason"]


def test_analyze_vocals_accepts_vocal_range_tone(tmp_path):
    sr = 22050
    t = np.arange(3 * sr) / sr
    y = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    result = analyze_vocals(_write_wav(tmp_path / "vox.wav", y))
    assert result["separation_ok"] is True
    assert result["refusal_reason"] is None


# ---------------------------------------------------------- chord stability


def test_chord_recognition_ignores_brief_blip(tmp_path):
    sr = 22050
    seconds = 9.0
    y = np.zeros(int(seconds * sr))
    c_triad = (261.63, 329.63, 392.00)
    g_triad = (392.00, 493.88, 587.33)

    def triad(freqs, n, start):
        tt = np.arange(n) / sr
        seg = sum(a * np.sin(2 * np.pi * f * tt) for a, f in zip((1.0, 0.7, 0.5), freqs)) / 2.2
        y[start : start + n] = seg

    triad(c_triad, int(4.5 * sr), 0)
    triad(g_triad, int(0.3 * sr), int(4.5 * sr))  # 0.3s blip — must not survive
    triad(c_triad, int(4.2 * sr), int(4.8 * sr))
    events = recognize_chords(_write_wav(tmp_path / "blip.wav", y), key="C", mode="major")
    chords_found = {e["chord"] for e in events}
    assert chords_found == {"C"}
    # Events are contiguous and ordered.
    for prev, nxt in zip(events, events[1:]):
        assert nxt["start"] >= prev["start"]
