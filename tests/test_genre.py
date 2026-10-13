"""Trained genre model (PANNs Cnn10) — contract tests.

The honesty contract under test:
- genre_predictions carry the model's REAL output probabilities
  (passthrough from summarize_genre_outputs — no canned rank scores);
- only AudioSet genre classes are mapped, top-N sorted;
- a weak top score yields 'Mixed / uncertain', never a forced label;
- a missing/unloadable model degrades to primary_genre=None + note;
- the report keeps genre (model) and texture cues in separate sections.
"""
from __future__ import annotations

import numpy as np
import pytest

from workers.analysis import genre as genre_mod


def _vec(scores: dict[int, float]) -> np.ndarray:
    v = np.zeros(527, dtype=np.float64)
    for idx, val in scores.items():
        v[idx] = val
    return v


def test_summarize_maps_genre_classes_and_passes_probabilities_through():
    # 217 = Hip hop music, 216 = Pop music, 246 = Ambient music;
    # 137 = Music (not a genre class) and 36 = Rapping (not a genre
    # class) must be ignored even with high scores.
    out = genre_mod.summarize_genre_outputs(
        _vec({217: 0.6123, 216: 0.2049, 246: 0.1111, 137: 0.99, 36: 0.95})
    )
    assert out["primary_genre"] == "Hip hop music"
    labels = [p["label"] for p in out["genre_predictions"]]
    assert labels[:3] == ["Hip hop music", "Pop music", "Ambient music"]
    probs = [p["probability"] for p in out["genre_predictions"]]
    assert probs[:3] == [0.6123, 0.2049, 0.1111]  # exact passthrough
    assert "Music" not in labels and "Rapping" not in labels
    assert out["genre_model"] == genre_mod.MODEL_ID


def test_summarize_never_emits_canned_rank_scores():
    out = genre_mod.summarize_genre_outputs(_vec({235: 0.5, 229: 0.4999}))
    probs = [p["probability"] for p in out["genre_predictions"][:2]]
    assert probs == [0.5, 0.4999]
    assert 0.8 not in probs and 0.7 not in probs  # the old 0.8/0.7 rank scores


def test_low_confidence_reads_mixed_uncertain_but_shows_candidates():
    out = genre_mod.summarize_genre_outputs(
        _vec({229: 0.12, 227: 0.09, 217: 0.05})
    )
    assert out["primary_genre"] == "Mixed / uncertain"
    assert out["genre_predictions"][0]["label"] == "Country"
    assert out["genre_predictions"][0]["probability"] == 0.12
    assert "below" in out["genre_note"]


def test_threshold_boundary_is_confident():
    out = genre_mod.summarize_genre_outputs(
        _vec({217: genre_mod.LOW_CONFIDENCE_THRESHOLD})
    )
    assert out["primary_genre"] == "Hip hop music"


def test_missing_model_degrades_without_raising(tmp_path, monkeypatch):
    missing = tmp_path / "no-such-model.pth"
    monkeypatch.setenv("PYTHAGORA_GENRE_MODEL", str(missing))
    genre_mod._MODEL = None
    genre_mod._MODEL_ERROR = None
    try:
        out = genre_mod.predict_genre(np.zeros(32000, dtype=np.float32))
    finally:
        genre_mod._MODEL = None
        genre_mod._MODEL_ERROR = None
    assert out["primary_genre"] is None
    assert out["genre_predictions"] == []
    assert "unavailable" in out["genre_note"]


def test_bundle_genre_fields_default_empty():
    from backend.models.schemas import TelemetryBundle

    bundle = TelemetryBundle(bpm=120.0)
    assert bundle.primary_genre is None
    assert bundle.genre_predictions == []
    assert bundle.genre_model is None
    assert bundle.texture_cues == []


def test_report_separates_genre_from_texture():
    from backend.models.schemas import TelemetryBundle, TrackRecord
    from backend.report import render_report_html

    track = TrackRecord(id="tg", title="Sep", artist="A", original_file_path="/x.wav")
    bundle = TelemetryBundle(
        bpm=95.0,
        duration=60.0,
        primary_genre="Hip hop music",
        genre_predictions=[
            {"label": "Hip hop music", "probability": 0.42},
            {"label": "Pop music", "probability": 0.2},
        ],
        genre_model=genre_mod.MODEL_ID,
        genre_note="taxonomy note",
        texture_cues=[{"name": "Ambient / pads", "note": "dominant sustained harmonic texture"}],
    )
    html_out = render_report_html(track, bundle)
    assert "<h3>Genre — trained model</h3>" in html_out
    assert "<h3>Texture cues</h3>" in html_out
    genre_section = html_out.split("<h3>Genre — trained model</h3>")[1].split("</section>")[0]
    texture_section = html_out.split("<h3>Texture cues</h3>")[1].split("</section>")[0]
    assert "Hip hop music" in genre_section and "42.0%" in genre_section
    assert "Ambient / pads" in texture_section
    assert "Ambient / pads" not in genre_section  # texture is never a genre


def test_report_shows_genre_note_when_model_unavailable():
    from backend.models.schemas import TelemetryBundle, TrackRecord
    from backend.report import render_report_html

    track = TrackRecord(id="tg2", title="NoModel", artist="A", original_file_path="/x.wav")
    bundle = TelemetryBundle(
        bpm=95.0,
        duration=60.0,
        primary_genre=None,
        genre_note="Genre model unavailable: model file not found",
        genre_model=genre_mod.MODEL_ID,
    )
    html_out = render_report_html(track, bundle)
    assert "not classified" in html_out
    assert "model file not found" in html_out


@pytest.mark.skipif(
    not genre_mod.model_path().is_file(),
    reason="genre checkpoint not downloaded (see README)",
)
def test_real_model_runs_on_silence_free_noise():
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(32000 * 3) * 0.05).astype(np.float32)
    out = genre_mod.predict_genre(y)
    assert out["genre_model"] == genre_mod.MODEL_ID
    assert out["primary_genre"] is not None  # a label or 'Mixed / uncertain'
    assert 1 <= len(out["genre_predictions"]) <= genre_mod.TOP_N
    labels = {label for _, label in genre_mod.GENRE_CLASSES}
    assert all(p["label"] in labels for p in out["genre_predictions"])
    assert all(0.0 <= p["probability"] <= 1.0 for p in out["genre_predictions"])
