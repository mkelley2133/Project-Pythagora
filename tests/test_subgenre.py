"""Zero-shot subgenre matching (LAION CLAP) — contract tests.

The honesty contract under test:
- percents are similarity shares from a softmax over the fixed
  candidate set (they sum to 100% across ALL candidates) — never
  dressed up as trained probabilities, and the note says so;
- top-5 truncation is by share, descending;
- the primary subgenre commits only on a decisive share (>= 15% or
  >= 1.5x the runner-up); otherwise it stays None with an explanation;
- a missing/unloadable model degrades to empty predictions + note;
- the report renders subgenre matches in their own section, with the
  similarity (not probability) labeling.
"""
from __future__ import annotations

import numpy as np
import pytest

from workers.analysis import subgenre as sub_mod

N = len(sub_mod.SUBGENRES)
IDX = {label: i for i, (label, _fam, _p) in enumerate(sub_mod.SUBGENRES)}


def _sims(peaks: dict[str, float], base: float = 0.0) -> np.ndarray:
    v = np.full(N, base, dtype=np.float64)
    for label, val in peaks.items():
        v[IDX[label]] = val
    return v


def test_vocabulary_shape_and_families():
    assert 40 <= N <= 60
    labels = [label for label, _f, _p in sub_mod.SUBGENRES]
    assert len(set(labels)) == N  # no duplicates
    for required in ("trap", "melodic trap", "emo rap", "drill", "boom bap",
                     "phonk", "trap soul", "bedroom pop", "shoegaze",
                     "ambient electronic", "country", "classical"):
        assert required in labels


def test_softmax_shares_top5_and_primary_commit():
    scale = 18.66
    sims = _sims({"melodic trap": 0.40, "emo rap": 0.30, "trap": 0.20})
    out = sub_mod.summarize_subgenre_outputs(sims, scale)
    preds = out["subgenre_predictions"]
    assert len(preds) == sub_mod.TOP_N == 5
    assert preds[0]["label"] == "melodic trap"
    assert preds[0]["family"] == "Hip hop"
    # expected share recomputed independently
    logits = scale * sims
    shares = np.exp(logits - logits.max())
    shares /= shares.sum()
    assert preds[0]["percent"] == round(float(shares[IDX["melodic trap"]]) * 100, 1)
    assert preds[0]["percent"] >= 15.0
    assert out["primary_subgenre"] == "melodic trap"
    percents = [p["percent"] for p in preds]
    assert percents == sorted(percents, reverse=True)
    assert sum(percents) <= 100.0
    assert out["subgenre_model"] == sub_mod.MODEL_ID
    assert "NOT trained classifier probabilities" in out["subgenre_note"]


def test_primary_commits_on_dominance_below_share_floor():
    # top share ~11.9% (< 15% floor) but ~1.7x the runner-up -> commits.
    sims = _sims({"drill": 0.10, "trap": 0.0711})
    out = sub_mod.summarize_subgenre_outputs(sims, 18.66)
    top = out["subgenre_predictions"][0]
    assert top["label"] == "drill" and top["percent"] < 15.0
    assert out["primary_subgenre"] == "drill"


def test_flat_similarities_commit_nothing():
    out = sub_mod.summarize_subgenre_outputs(np.full(N, 0.25), 18.66)
    assert out["primary_subgenre"] is None
    assert "not decisive" in out["subgenre_note"]
    # uniform over N candidates -> every share is 100/N
    assert out["subgenre_predictions"][0]["percent"] == pytest.approx(100.0 / N, abs=0.1)


def test_wrong_score_count_is_reported_not_faked():
    out = sub_mod.summarize_subgenre_outputs(np.zeros(7), 18.66)
    assert out["primary_subgenre"] is None
    assert out["subgenre_predictions"] == []
    assert "7 scores" in out["subgenre_note"]


def test_missing_model_degrades_without_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHAGORA_SUBGENRE_MODEL", str(tmp_path / "no-such-dir"))
    sub_mod._STATE = None
    sub_mod._STATE_ERROR = None
    try:
        out = sub_mod.predict_subgenre(np.zeros(48000, dtype=np.float32))
    finally:
        sub_mod._STATE = None
        sub_mod._STATE_ERROR = None
    assert out["primary_subgenre"] is None
    assert out["subgenre_predictions"] == []
    assert "unavailable" in out["subgenre_note"]


def test_bundle_subgenre_fields_default_empty():
    from backend.models.schemas import TelemetryBundle

    bundle = TelemetryBundle(bpm=120.0)
    assert bundle.primary_subgenre is None
    assert bundle.subgenre_predictions == []
    assert bundle.subgenre_model is None
    assert bundle.subgenre_note is None


def test_report_subgenre_section_is_similarity_labeled():
    from backend.models.schemas import TelemetryBundle, TrackRecord
    from backend.report import render_report_html

    track = TrackRecord(id="ts", title="Sub", artist="A", original_file_path="/x.wav")
    bundle = TelemetryBundle(
        bpm=95.0,
        duration=60.0,
        primary_subgenre="melodic trap",
        subgenre_predictions=[
            {"label": "melodic trap", "family": "Hip hop", "percent": 24.6},
            {"label": "emo rap", "family": "Hip hop", "percent": 16.8},
        ],
        subgenre_model=sub_mod.MODEL_ID,
        subgenre_note=sub_mod.SIMILARITY_NOTE,
    )
    html_out = render_report_html(track, bundle)
    assert "<h3>Subgenre matches — zero-shot similarity</h3>" in html_out
    section = html_out.split("<h3>Subgenre matches")[1].split("</section>")[0]
    assert "melodic trap" in section and "24.6%" in section
    assert "Hip hop" in section
    assert "similarity shares" in section  # honesty labeling present
    # subgenre values must not leak into the trained-genre section
    if "<h3>Genre — trained model</h3>" in html_out:
        genre_section = html_out.split("<h3>Genre — trained model</h3>")[1].split("</section>")[0]
        assert "melodic trap" not in genre_section


@pytest.mark.skipif(
    not sub_mod.model_dir().is_dir(),
    reason="CLAP subgenre model not downloaded (see README)",
)
def test_concurrent_first_load_is_race_free():
    """Regression: pipeline jobs run on a thread pool, and transformers'
    lazy imports race under concurrent first use (live failure:
    'cannot import name ClapModel'). The loader lock must make four
    simultaneous first predictions all succeed."""
    from concurrent.futures import ThreadPoolExecutor

    sub_mod._STATE = None
    sub_mod._STATE_ERROR = None
    rng = np.random.default_rng(1)
    y = (rng.standard_normal(48000) * 0.05).astype(np.float32)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outs = list(pool.map(lambda _i: sub_mod.predict_subgenre(y), range(4)))
    assert all(o["subgenre_predictions"] for o in outs)
    assert all(o["subgenre_model"] == sub_mod.MODEL_ID for o in outs)


@pytest.mark.skipif(
    not sub_mod.model_dir().is_dir(),
    reason="CLAP subgenre model not downloaded (see README)",
)
def test_real_model_runs_and_shares_sum_to_100():
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(48000 * 3) * 0.05).astype(np.float32)
    out = sub_mod.predict_subgenre(y)
    assert out["subgenre_model"] == sub_mod.MODEL_ID
    preds = out["subgenre_predictions"]
    assert len(preds) == sub_mod.TOP_N
    labels = {label for label, _f, _p in sub_mod.SUBGENRES}
    assert all(p["label"] in labels for p in preds)
    percents = [p["percent"] for p in preds]
    assert percents == sorted(percents, reverse=True)
    assert all(0.0 <= p <= 100.0 for p in percents)
    assert sum(percents) <= 100.05
