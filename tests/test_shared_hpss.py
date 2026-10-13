"""Shared-HPSS equivalence pins (efficiency pass, 2026-10-13).

The pipeline computes HPSS once and shares it across the chord,
feature, and section stages instead of running it three times on the
same samples. Two of the three sharings are bit-identical by
construction and are pinned as exact equalities here; the third —
section_map slicing the full-track percussive component instead of
re-running HPSS per window — is NOT bit-identical (HPSS boundary
effects at section edges), so it is pinned as a bounded difference
with identical section roles.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from workers.analysis import audio as audio_mod
from workers.analysis.chords import recognize_chords, recognize_chords_from_audio

SR = 22050


def _progression(tmp_path):
    """12s C-F-G synth progression with a kick grid (as in the upload tests)."""
    chords = [(261.63, 329.63, 392.00), (349.23, 440.00, 523.25), (392.00, 493.88, 587.33)]
    y = np.zeros(12 * SR)
    for i, (r, t, f) in enumerate(chords):
        n = 4 * SR
        tt = np.arange(n) / SR
        seg = (np.sin(2 * np.pi * r * tt) + 0.7 * np.sin(2 * np.pi * t * tt)
               + 0.5 * np.sin(2 * np.pi * f * tt)) / 2.2
        y[i * n:(i + 1) * n] = seg
    kick_n = int(0.09 * SR)
    kt = np.arange(kick_n) / SR
    kick = np.sin(2 * np.pi * 55 * kt) * np.exp(-kt * 35)
    for beat in range(24):
        s = int(beat * 0.5 * SR)
        y[s:s + kick_n] += kick[: max(0, min(kick_n, len(y) - s))]
    path = tmp_path / "prog.wav"
    sf.write(str(path), y.astype(np.float32), SR)
    return str(path), y


def test_chords_from_audio_matches_path_entry(tmp_path):
    path, y = _progression(tmp_path)
    via_path = recognize_chords(path, key="C", mode="major")
    via_audio = recognize_chords_from_audio(y, SR, key="C", mode="major")
    assert via_audio == via_path
    # And with a precomputed harmonic component (the pipeline's sharing).
    y_harm, _ = audio_mod.hpss_waveform(y)
    via_shared = recognize_chords_from_audio(y, SR, key="C", mode="major", y_harm=y_harm)
    assert via_shared == via_path


def test_core_features_shared_hpss_is_identical(tmp_path):
    _, y = _progression(tmp_path)
    pair = audio_mod.hpss_waveform(y)
    assert audio_mod.core_features(y, SR, hpss_pair=pair) == audio_mod.core_features(y, SR)


def test_section_map_sliced_percussion_bounded(tmp_path):
    _, y = _progression(tmp_path)
    _, perc = audio_mod.hpss_waveform(y)
    windowed = audio_mod.section_map(y, SR)
    sliced = audio_mod.section_map(y, SR, perc_waveform=perc)
    assert len(windowed) == len(sliced)
    for w, s in zip(windowed, sliced):
        assert w["start"] == s["start"] and w["end"] == s["end"]
        assert w["role"] == s["role"]
        assert abs(w["percussive_share"] - s["percussive_share"]) <= 0.01
        assert w["energy"] == s["energy"] and w["centroid"] == s["centroid"]
