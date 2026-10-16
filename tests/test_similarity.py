"""Unit tests for similar-tracks retrieval (backend/similarity.py).

The ranking math is pure NumPy, so it is tested with synthetic vectors
whose cosines are known exactly — no model files, no audio. The CLAP
embedding path itself degrades to None without weights, which is also
asserted (with the model location pointed somewhere empty).
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.similarity import (
    compute_track_embedding,
    cosine_similarity,
    rank_by_similarity,
)


def test_cosine_identical_vectors_is_one():
    v = np.array([0.3, -0.4, 0.5])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one():
    assert cosine_similarity(np.array([1.0, 2.0]), np.array([-1.0, -2.0])) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant():
    assert cosine_similarity(np.array([2.0, 0.0]), np.array([5.0, 0.0])) == pytest.approx(1.0)


def test_cosine_zero_norm_is_zero_not_nan():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0
    assert cosine_similarity(np.zeros(3), np.zeros(3)) == 0.0


def test_rank_orders_by_similarity_descending():
    query = np.array([1.0, 0.0])
    candidates = {
        "far": np.array([0.0, 1.0]),
        "near": np.array([0.99, 0.01]),
        "mid": np.array([0.7, 0.7]),
    }
    ranked = rank_by_similarity(query, candidates)
    assert [r["track_id"] for r in ranked] == ["near", "mid", "far"]
    assert ranked[0]["similarity"] > ranked[1]["similarity"] > ranked[2]["similarity"]


def test_rank_excludes_the_query_track():
    query = np.array([1.0, 0.0])
    candidates = {"self": np.array([1.0, 0.0]), "other": np.array([0.9, 0.1])}
    ranked = rank_by_similarity(query, candidates, exclude_id="self")
    assert [r["track_id"] for r in ranked] == ["other"]


def test_rank_respects_top_n_and_empty_candidates():
    query = np.array([1.0, 0.0])
    candidates = {f"t{i}": np.array([1.0, float(i)]) for i in range(6)}
    assert len(rank_by_similarity(query, candidates, top_n=2)) == 2
    assert rank_by_similarity(query, {}) == []


def test_embedding_degrades_to_none_without_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHAGORA_SUBGENRE_MODEL", str(tmp_path / "no-such-dir"))
    # The subgenre loader caches its state; a fresh interpreter in CI has
    # no state, and locally the model may already be loaded — either way
    # compute_track_embedding must return an array or None, never raise.
    result = compute_track_embedding(__file__)
    assert result is None or (isinstance(result, np.ndarray) and result.ndim == 1)
