"""Decisive primary-genre rule (PYTHAGORA_GENRE_DECISIVE) — contract tests.

The flag is an evaluation aid, OFF by default:
- flag off  -> behavior is byte-for-byte the strict 0.35 rule;
- flag on   -> primary commits when top >= 0.10 AND top >= 1.5x runner-up;
- the genre_note discloses when the decisive rule produced the label;
- predictions/probabilities are untouched in both modes.
"""
from __future__ import annotations

import numpy as np
import pytest

from workers.analysis import genre as genre_mod

HIP_HOP = 217
ELECTRONIC = 239
COUNTRY = 229


def _vec(scores: dict[int, float]) -> np.ndarray:
    v = np.zeros(527, dtype=np.float64)
    for idx, val in scores.items():
        v[idx] = val
    return v


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.delenv("PYTHAGORA_GENRE_DECISIVE", raising=False)


def test_default_off_keeps_strict_behavior():
    # This Side's real scores: hip hop clearly leads but is below 0.35.
    out = genre_mod.summarize_genre_outputs(_vec({HIP_HOP: 0.156, ELECTRONIC: 0.0919}))
    assert out["primary_genre"] == "Mixed / uncertain"
    assert "below" in out["genre_note"]


def test_decisive_commits_clear_relative_winner(monkeypatch):
    monkeypatch.setenv("PYTHAGORA_GENRE_DECISIVE", "1")
    out = genre_mod.summarize_genre_outputs(_vec({HIP_HOP: 0.156, ELECTRONIC: 0.0919}))
    assert out["primary_genre"] == "Hip hop music"
    assert "decisive" in out["genre_note"]
    # Probabilities pass through untouched in both modes.
    assert out["genre_predictions"][0]["probability"] == 0.156


def test_decisive_requires_dominance(monkeypatch):
    monkeypatch.setenv("PYTHAGORA_GENRE_DECISIVE", "1")
    # Every Time's real scores: 0.1184 vs 0.1028 — a lead, not dominance.
    out = genre_mod.summarize_genre_outputs(_vec({HIP_HOP: 0.1184, ELECTRONIC: 0.1028}))
    assert out["primary_genre"] == "Mixed / uncertain"


def test_decisive_requires_absolute_floor(monkeypatch):
    monkeypatch.setenv("PYTHAGORA_GENRE_DECISIVE", "1")
    # 3x dominance but below the 0.10 floor: still no claim.
    out = genre_mod.summarize_genre_outputs(_vec({HIP_HOP: 0.09, ELECTRONIC: 0.03}))
    assert out["primary_genre"] == "Mixed / uncertain"
    # Drowning's real top score is below the floor too.
    out = genre_mod.summarize_genre_outputs(_vec({COUNTRY: 0.0435, HIP_HOP: 0.02}))
    assert out["primary_genre"] == "Mixed / uncertain"


def test_decisive_boundary_values_commit(monkeypatch):
    monkeypatch.setenv("PYTHAGORA_GENRE_DECISIVE", "1")
    out = genre_mod.summarize_genre_outputs(_vec({HIP_HOP: 0.10, ELECTRONIC: 0.06}))
    assert out["primary_genre"] == "Hip hop music"


def test_decisive_does_not_relabel_strict_passes(monkeypatch):
    monkeypatch.setenv("PYTHAGORA_GENRE_DECISIVE", "1")
    out = genre_mod.summarize_genre_outputs(_vec({HIP_HOP: 0.5, ELECTRONIC: 0.1}))
    assert out["primary_genre"] == "Hip hop music"
    assert out["genre_note"] == genre_mod.TAXONOMY_NOTE  # strict wording


def test_flag_values_other_than_1_stay_off(monkeypatch):
    monkeypatch.setenv("PYTHAGORA_GENRE_DECISIVE", "true")
    out = genre_mod.summarize_genre_outputs(_vec({HIP_HOP: 0.156, ELECTRONIC: 0.0919}))
    assert out["primary_genre"] == "Mixed / uncertain"
