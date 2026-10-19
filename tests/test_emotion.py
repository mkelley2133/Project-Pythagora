"""Zero-shot emotion matching (LAION CLAP) — contract tests.

The honesty contract under test:
- percents are zero-shot emotion similarity shares from a softmax over
  the fixed candidate set (they sum to 100% across ALL candidates) —
  never dressed up as trained probabilities or as a reading of what a
  song feels, and the note says so;
- top-5 truncation is by share, descending; the full distribution is
  carried in emotion_all_shares;
- the primary emotion commits only on a decisive share (>= 15% or
  >= 1.5x the runner-up) — the same floors as the subgenre stage;
  otherwise it stays None with an explanation;
- a missing/unloadable model degrades to empty predictions + note;
- the report renders emotion matches in their own section, with the
  zero-shot similarity (not probability / not a feeling) labeling.
"""
from __future__ import annotations

import numpy as np
import pytest

from workers.analysis import emotion as emo_mod

N = len(emo_mod.EMOTIONS)
IDX = {label: i for i, (label, _p) in enumerate(emo_mod.EMOTIONS)}


def _sims(peaks: dict[str, float], base: float = 0.0) -> np.ndarray:
    v = np.full(N, base, dtype=np.float64)
    for label, val in peaks.items():
        v[IDX[label]] = val
    return v


def _expected_shares(sims: np.ndarray, scale: float) -> np.ndarray:
    logits = scale * sims
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def test_vocabulary_shape_and_required_labels():
    assert 10 <= N <= 14
    labels = [label for label, _p in emo_mod.EMOTIONS]
    assert len(set(labels)) == N  # no duplicates
    for required in ("heartbroken confession", "defiant anthem",
                     "numb dissociation", "quiet grief", "tender devotion"):
        assert required in labels
    for _label, prompt in emo_mod.EMOTIONS:
        assert prompt.strip()  # every candidate carries a disclosed prompt


def test_softmax_shares_top5_all_shares_and_primary_commit():
    scale = 18.66
    sims = _sims({"heartbroken confession": 0.40, "quiet grief": 0.30,
                  "numb dissociation": 0.20})
    out = emo_mod.summarize_emotion_outputs(sims, scale)
    preds = out["emotion_predictions"]
    assert len(preds) == emo_mod.TOP_N == 5
    assert preds[0]["label"] == "heartbroken confession"
    # expected shares recomputed independently; top-5 are the head of
    # the full distribution, and every candidate's share is exact
    shares = _expected_shares(sims, scale)
    order = np.argsort(-shares, kind="stable")
    all_shares = out["emotion_all_shares"]
    assert len(all_shares) == N
    for entry, i in zip(all_shares, order):
        assert entry["label"] == emo_mod.EMOTIONS[i][0]
        assert entry["percent"] == round(float(shares[i]) * 100.0, 1)
    assert preds == all_shares[: emo_mod.TOP_N]
    # shares sum to ~100% across ALL candidates (rounding slack only)
    assert sum(e["percent"] for e in all_shares) == pytest.approx(100.0, abs=0.7)
    assert preds[0]["percent"] >= 15.0
    assert out["primary_emotion"] == "heartbroken confession"
    percents = [p["percent"] for p in preds]
    assert percents == sorted(percents, reverse=True)
    assert out["emotion_model"] == emo_mod.MODEL_ID


def test_labeling_language_present_in_payload():
    out = emo_mod.summarize_emotion_outputs(
        _sims({"defiant anthem": 0.40}), 18.66
    )
    note = out["emotion_note"]
    assert "zero-shot emotion similarity shares" in note
    assert "NOT trained classifier probabilities" in note
    assert "not a claim about what the song actually feels" in note
    # the schema carries the same labeling on the wire format
    from backend.models.schemas import EmotionPrediction, TelemetryBundle

    desc = EmotionPrediction.model_fields["percent"].description or ""
    assert "similarity share" in desc
    bundle = TelemetryBundle(
        bpm=90.0,
        primary_emotion=out["primary_emotion"],
        emotion_predictions=out["emotion_predictions"],
        emotion_model=out["emotion_model"],
        emotion_note=out["emotion_note"],
    )
    payload = bundle.model_dump(mode="json")
    assert payload["primary_emotion"] == "defiant anthem"
    assert "zero-shot emotion similarity shares" in payload["emotion_note"]


def test_primary_commits_on_share_floor():
    sims = _sims({"euphoric celebration": 0.35, "reckless joy": 0.10})
    out = emo_mod.summarize_emotion_outputs(sims, 18.66)
    assert out["emotion_predictions"][0]["percent"] >= 15.0
    assert out["primary_emotion"] == "euphoric celebration"


def test_primary_commits_on_dominance_below_share_floor():
    # top share ~13.6% (< 15% floor) but ~1.6x the runner-up -> commits.
    sims = _sims({"defiant anthem": 0.23, "heartbroken confession": 0.205},
                 base=0.20)
    out = emo_mod.summarize_emotion_outputs(sims, 18.66)
    top = out["emotion_predictions"][0]
    assert top["label"] == "defiant anthem" and top["percent"] < 15.0
    assert out["primary_emotion"] == "defiant anthem"


def test_below_both_thresholds_stays_unclaimed():
    # one hair above uniform: top share ~8.5%, ratio to runner ~1.02x —
    # below the 15% floor AND below 1.5x dominance, so nothing commits.
    sims = _sims({"heartbroken confession": 0.251}, base=0.25)
    out = emo_mod.summarize_emotion_outputs(sims, 18.66)
    assert out["primary_emotion"] is None
    assert "not decisive" in out["emotion_note"]
    assert out["emotion_predictions"][0]["percent"] < 15.0


def test_flat_similarities_commit_nothing():
    out = emo_mod.summarize_emotion_outputs(np.full(N, 0.25), 18.66)
    assert out["primary_emotion"] is None
    assert "not decisive" in out["emotion_note"]
    # uniform over N candidates -> every share is 100/N
    assert out["emotion_predictions"][0]["percent"] == pytest.approx(100.0 / N, abs=0.1)


def test_wrong_score_count_is_reported_not_faked():
    out = emo_mod.summarize_emotion_outputs(np.zeros(7), 18.66)
    assert out["primary_emotion"] is None
    assert out["emotion_predictions"] == []
    assert out["emotion_all_shares"] == []
    assert "7 scores" in out["emotion_note"]


def test_missing_model_degrades_without_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHAGORA_EMOTION_MODEL", str(tmp_path / "no-such-dir"))
    emo_mod._STATE = None
    emo_mod._STATE_ERROR = None
    try:
        out = emo_mod.predict_emotion(np.zeros(48000, dtype=np.float32))
    finally:
        emo_mod._STATE = None
        emo_mod._STATE_ERROR = None
    assert out["primary_emotion"] is None
    assert out["emotion_predictions"] == []
    assert "unavailable" in out["emotion_note"]


def test_bundle_emotion_fields_default_empty():
    from backend.models.schemas import TelemetryBundle

    bundle = TelemetryBundle(bpm=120.0)
    assert bundle.primary_emotion is None
    assert bundle.emotion_predictions == []
    assert bundle.emotion_model is None
    assert bundle.emotion_note is None


def test_report_emotion_section_is_similarity_labeled():
    from backend.models.schemas import TelemetryBundle, TrackRecord
    from backend.report import render_report_html

    track = TrackRecord(id="te", title="Emo", artist="A", original_file_path="/x.wav")
    bundle = TelemetryBundle(
        bpm=95.0,
        duration=60.0,
        primary_emotion="heartbroken confession",
        emotion_predictions=[
            {"label": "heartbroken confession", "percent": 31.2},
            {"label": "quiet grief", "percent": 17.4},
        ],
        emotion_model=emo_mod.MODEL_ID,
        emotion_note=emo_mod.SIMILARITY_NOTE,
    )
    html_out = render_report_html(track, bundle)
    assert "<h3>Emotion matches — zero-shot similarity</h3>" in html_out
    section = html_out.split("<h3>Emotion matches")[1].split("</section>")[0]
    assert "heartbroken confession" in section and "31.2%" in section
    assert "similarity shares" in section  # honesty labeling present
    assert "not a claim about what the song" in section
    # emotion values must not leak into the trained-genre section
    if "<h3>Genre — trained model</h3>" in html_out:
        genre_section = html_out.split("<h3>Genre — trained model</h3>")[1].split("</section>")[0]
        assert "heartbroken confession" not in genre_section


@pytest.mark.skipif(
    not emo_mod.model_dir().is_dir(),
    reason="CLAP model not downloaded (see README)",
)
def test_concurrent_first_load_is_race_free():
    """Regression mirror of the subgenre race test: the loader lock must
    make four simultaneous first predictions all succeed."""
    from concurrent.futures import ThreadPoolExecutor

    emo_mod._STATE = None
    emo_mod._STATE_ERROR = None
    rng = np.random.default_rng(1)
    y = (rng.standard_normal(48000) * 0.05).astype(np.float32)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outs = list(pool.map(lambda _i: emo_mod.predict_emotion(y), range(4)))
    assert all(o["emotion_predictions"] for o in outs)
    assert all(o["emotion_model"] == emo_mod.MODEL_ID for o in outs)


@pytest.mark.skipif(
    not emo_mod.model_dir().is_dir(),
    reason="CLAP model not downloaded (see README)",
)
def test_real_model_runs_and_shares_sum_to_100():
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(48000 * 3) * 0.05).astype(np.float32)
    out = emo_mod.predict_emotion(y)
    assert out["emotion_model"] == emo_mod.MODEL_ID
    preds = out["emotion_predictions"]
    assert len(preds) == emo_mod.TOP_N
    labels = {label for label, _p in emo_mod.EMOTIONS}
    assert all(p["label"] in labels for p in preds)
    percents = [p["percent"] for p in preds]
    assert percents == sorted(percents, reverse=True)
    assert all(0.0 <= p <= 100.0 for p in percents)
    all_shares = out["emotion_all_shares"]
    assert len(all_shares) == N
    assert sum(e["percent"] for e in all_shares) == pytest.approx(100.0, abs=0.7)
