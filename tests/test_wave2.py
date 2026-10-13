"""Wave 2 coverage: v1 exclusions, real progress (C), batch uploads (D),
track comparison (E), projects/notes persistence (F), mix suggestions (G),
and pitch/speed adjustment (H)."""
import io
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from backend import jobs, store
from backend.models.schemas import (
    Project,
    TelemetryBundle,
    TrackAnalysisRecord,
    TrackRecord,
)

client = TestClient(app)


def _synth_wav(seconds=4, sr=22050, freq=261.63) -> bytes:
    tt = np.arange(int(seconds * sr)) / sr
    y = 0.4 * np.sin(2 * np.pi * freq * tt) + 0.3 * np.sin(2 * np.pi * freq * 1.26 * tt)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _wait_job(job_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/tracks/jobs/{job_id}").json()
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def _upload(title: str, wav: bytes, name: str = "t.wav"):
    return client.post(
        "/tracks/upload",
        data={"title": title, "artist": "Test Bot"},
        files={"audio": (name, wav, "audio/wav")},
    )


# ------------------------------------------------------------- Task 0 / v1

def test_report_has_no_stem_or_transcription_claims():
    from backend.report import render_report_html

    track = TrackRecord(id="t9", title="Plain", artist="A", original_file_path="/x.wav")
    html_out = render_report_html(track, TelemetryBundle(bpm=120.0, duration=30.0))
    assert "<h3>Stems</h3>" not in html_out
    assert "AI transcription" not in html_out
    assert "<h3>Lyrics</h3>" not in html_out  # no supplied sheet -> no section


def test_report_labels_supplied_lyrics_as_supplied():
    from backend.report import render_report_html

    track = TrackRecord(id="t10", title="Sheet", artist="A", original_file_path="/x.wav")
    bundle = TelemetryBundle(
        bpm=120.0,
        duration=30.0,
        timestamped_lyrics=[{"start": 1.0, "end": 3.0, "text": "a supplied line"}],
        suggestions=[
            {
                "area": "Levels",
                "suggestion": "Leave headroom.",
                "why": "True peak is -0.1 dBFS, within 0.3 dB of full scale.",
                "severity": "high",
            }
        ],
    )
    html_out = render_report_html(track, bundle)
    assert "<h3>Lyrics</h3>" in html_out
    assert "supplied" in html_out.lower()
    assert "AI transcription" not in html_out
    assert "<h3>Mix Suggestions</h3>" in html_out
    assert "-0.1 dBFS" in html_out


# ------------------------------------------------------------- Phase C

def test_job_stage_progress_is_stage_weighted():
    job = jobs.create_job("track-progress-unit")
    names = ["A", "B", "C", "D"]
    jobs.set_stages(job["job_id"], names)
    jobs.advance_stage(job["job_id"], "A")
    got = jobs.get_job(job["job_id"])
    assert got["progress"] == 0  # nothing completed yet
    assert got["stage_index"] == 0 and got["stage_total"] == 4
    assert got["stages"][0]["status"] == "active"
    assert got["started_at"] is not None
    assert got["elapsed_seconds"] is not None and got["elapsed_seconds"] >= 0

    jobs.advance_stage(job["job_id"], "C")
    got = jobs.get_job(job["job_id"])
    assert got["progress"] == 50  # A and B completed
    assert [s["status"] for s in got["stages"]] == ["done", "done", "active", "pending"]

    jobs.skip_stage(job["job_id"], "C")
    got = jobs.get_job(job["job_id"])
    assert got["progress"] == 75  # skipped counts as completed
    assert got["stages"][2]["status"] == "skipped"

    jobs.finish_stages(job["job_id"])
    got = jobs.get_job(job["job_id"])
    assert got["progress"] == 100
    assert got["stages"][3]["status"] == "done"


def test_job_api_exposes_stage_fields():
    r = _upload("Progress API", _synth_wav(seconds=4))
    assert r.status_code == 202, r.text
    job = _wait_job(r.json()["job_id"])
    assert job["state"] == "done"
    assert job["progress"] == 100
    from workers.pipeline import STAGES

    assert job["stage_total"] == len(STAGES)
    assert [s["name"] for s in job["stages"]] == STAGES
    assert all(s["status"] == "done" for s in job["stages"])
    assert job["started_at"] is not None
    assert job["elapsed_seconds"] is not None and job["elapsed_seconds"] >= 0


# ------------------------------------------------------------- Phase D

def test_batch_uploads_all_complete_and_bad_file_isolated():
    good1 = _upload("Batch One", _synth_wav(seconds=4, freq=220.0), "b1.wav")
    good2 = _upload("Batch Two", _synth_wav(seconds=4, freq=330.0), "b2.wav")
    assert good1.status_code == 202 and good2.status_code == 202

    bad = client.post(
        "/tracks/upload",
        data={"title": "Batch Bad"},
        files={"audio": ("bad.wav", b"this is not audio at all", "audio/wav")},
    )
    assert bad.status_code == 400  # rejected at the door, no job created

    for resp in (good1, good2):
        job = _wait_job(resp.json()["job_id"])
        assert job["state"] == "done"
        track_id = resp.json()["track"]["id"]
        assert client.get(f"/tracks/{track_id}/analysis").status_code == 200


def test_batch_job_failure_does_not_kill_sibling(monkeypatch):
    import workers.pipeline as pipe

    real_run = pipe.run_pipeline
    bad_ids: set[str] = set()

    def flaky(job_id, track_id, audio_path):
        if track_id in bad_ids:
            jobs.update_job(job_id, state="failed", stage="Failed", error="boom")
            raise RuntimeError("boom")
        return real_run(job_id, track_id, audio_path)

    monkeypatch.setattr(pipe, "run_pipeline", flaky)

    ok = _upload("Batch OK", _synth_wav(seconds=4, freq=262.0), "ok.wav")
    doomed = _upload("Batch Doomed", _synth_wav(seconds=4, freq=294.0), "doomed.wav")
    bad_ids.add(doomed.json()["track"]["id"])

    # The doomed job was already enqueued before we flagged it only if the
    # executor hadn't started it; either outcome is acceptable except the
    # sibling failing.
    ok_job = _wait_job(ok.json()["job_id"])
    assert ok_job["state"] == "done"
    doomed_job = _wait_job(doomed.json()["job_id"])
    assert doomed_job["state"] in ("done", "failed")
    if doomed_job["state"] == "failed":
        assert doomed_job["error"] == "boom"


# ------------------------------------------------------------- Phase E

def _seed_compared_tracks():
    a = TrackRecord(id="cmp-a", title="Alpha", artist="X", original_file_path="/storage/a.mp3")
    b = TrackRecord(id="cmp-b", title="Beta", artist="X", original_file_path="/storage/b.mp3")
    store.save_track(a)
    store.save_track(b)
    store.save_analysis(
        TrackAnalysisRecord(
            id="cmp-a",
            telemetry_json=TelemetryBundle(
                bpm=100.0, duration=60.0, musical_key="C", mode="major",
                key_confidence=0.9,
                chords=["C", "F", "G"],
                mix_measurements={"sub_pct": 5.0, "peak_dbfs": -1.0, "rms_dbfs": -14.0, "crest_db": 13.0},
            ),
        )
    )
    store.save_analysis(
        TrackAnalysisRecord(
            id="cmp-b",
            telemetry_json=TelemetryBundle(
                bpm=120.0, duration=75.0, musical_key="D", mode="minor",
                key_confidence=0.8,
                chords=["Dm", "Bb"],
                mix_measurements={"sub_pct": 15.0, "peak_dbfs": -0.2, "rms_dbfs": -9.0, "crest_db": 8.8},
            ),
        )
    )
    return a, b


def test_comparison_endpoint_shape_and_diffs():
    _seed_compared_tracks()
    r = client.get("/tracks/compare", params={"a": "cmp-a", "b": "cmp-b"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["a"]["title"] == "Alpha" and body["b"]["title"] == "Beta"
    rows = {row["field"]: row for row in body["rows"]}
    assert rows["bpm"]["a"] == 100.0 and rows["bpm"]["b"] == 120.0
    assert rows["bpm"]["different"] is True and rows["bpm"]["delta"] == 20.0
    assert rows["key"]["different"] is True
    assert body["chords"]["shared"] == []
    assert body["chords"]["only_a"] == ["C", "F", "G"]
    assert body["summary"]["fields_compared"] == len(body["rows"])
    assert body["summary"]["fields_different"] >= 3


def test_comparison_404s():
    _seed_compared_tracks()
    assert client.get("/tracks/compare", params={"a": "cmp-a", "b": "nope"}).status_code == 404
    bare = TrackRecord(id="cmp-bare", title="Bare", artist="X", original_file_path="/storage/c.mp3")
    store.save_track(bare)
    # Track exists but has no analysis yet.
    assert client.get("/tracks/compare", params={"a": "cmp-a", "b": "cmp-bare"}).status_code == 404


def test_comparison_report_page():
    _seed_compared_tracks()
    r = client.get("/tracks/compare/report", params={"a": "cmp-a", "b": "cmp-b"})
    assert r.status_code == 200
    assert "TRACK COMPARISON REPORT" in r.text
    assert "Alpha" in r.text and "Beta" in r.text
    assert 'class="diff"' in r.text


# ------------------------------------------------------------- Phase F

def test_projects_crud_and_assignment():
    r = client.post("/projects", json={"name": "Debut EP", "notes": "first notes"})
    assert r.status_code == 201, r.text
    project = r.json()
    pid = project["id"]
    assert project["track_count"] == 0

    r = client.patch(f"/projects/{pid}", json={"name": "Debut EP (deluxe)", "notes": "updated"})
    assert r.status_code == 200
    assert r.json()["name"] == "Debut EP (deluxe)"
    assert r.json()["notes"] == "updated"

    t = client.post(
        "/tracks",
        json={"title": "Org Song", "artist": "X", "original_file_path": "/storage/org.mp3"},
    ).json()
    r = client.patch(f"/tracks/{t['id']}", json={"project_id": pid, "notes": "lead single"})
    assert r.status_code == 200
    assert r.json()["project_id"] == pid
    assert r.json()["notes"] == "lead single"

    view = client.get(f"/projects/{pid}").json()
    assert view["track_count"] == 1 and view["track_ids"] == [t["id"]]

    # Assigning to a phantom project is a 404.
    assert client.patch(f"/tracks/{t['id']}", json={"project_id": "project-nope"}).status_code == 404

    # Deleting the project unassigns — never deletes — its tracks.
    assert client.delete(f"/projects/{pid}").status_code == 204
    assert client.get(f"/projects/{pid}").status_code == 404
    after = client.get(f"/tracks/{t['id']}").json()
    assert after["project_id"] is None
    assert after["notes"] == "lead single"


def test_project_notes_and_assignment_survive_reload(tmp_path, monkeypatch):
    lib = tmp_path / "library.json"
    monkeypatch.setenv("PYTHAGORAS_LIBRARY_FILE", str(lib))
    store.reset_state()
    store.load_persisted()

    project = Project(id="project-persist", name="Persisted", notes="keep me")
    store.save_project(project)
    track = TrackRecord(
        id="track-persist",
        title="Durable",
        artist="X",
        original_file_path="/storage/d.mp3",
        project_id="project-persist",
        notes="verse needs work",
    )
    store.save_track(track)
    assert lib.is_file()

    store.reset_state()
    assert store.get_project("project-persist") is None
    store.load_persisted()
    restored_project = store.get_project("project-persist")
    restored_track = store.get_track("track-persist")
    assert restored_project is not None and restored_project.notes == "keep me"
    assert restored_track is not None
    assert restored_track.project_id == "project-persist"
    assert restored_track.notes == "verse needs work"

    # Restore the shared demo state for any later tests in the session.
    store.seed_demo()


# ------------------------------------------------------------- Phase G

def test_suggestion_sub_heavy_fires_with_number():
    from workers.analysis.suggestions import build_suggestions

    out = build_suggestions({"mix": {"sub_pct": 25.0}, "duration": 60.0})
    sub = [s for s in out if s["area"] == "Low end"]
    assert sub and sub[0]["severity"] == "high"
    assert "25.0%" in sub[0]["why"]


def test_suggestion_near_clipping_fires():
    from workers.analysis.suggestions import build_suggestions

    out = build_suggestions({"mix": {"peak_dbfs": -0.1}})
    levels = [s for s in out if s["area"] == "Levels"]
    assert levels and "-0.1 dBFS" in levels[0]["why"]


def test_suggestion_quiet_dynamic_track_gets_no_level_advice():
    from workers.analysis.suggestions import build_suggestions

    ctx = {
        "mix": {
            "sub_pct": 3.0, "low_pct": 20.0, "mid_pct": 50.0, "high_pct": 20.0,
            "air_pct": 4.0, "peak_dbfs": -18.0, "rms_dbfs": -32.0, "crest_db": 14.0,
            "is_stereo": True, "stereo_correlation": 0.6,
        },
        "feats": {"percussive_share": 0.4, "harmonic_share": 0.6},
        "key_confidence": 0.9,
        "unique_chords": ["C", "F", "G"],
        "chord_events": 3,
        "duration": 60.0,
    }
    out = build_suggestions(ctx)
    assert out == []  # well-behaved control: nothing to flag


def test_suggestion_loop_and_mono_rules():
    from workers.analysis.suggestions import build_suggestions

    out = build_suggestions(
        {
            "mix": {"is_stereo": True, "stereo_correlation": 0.995},
            "unique_chords": ["Dm", "Bb", "F"],
            "chord_events": 14,
            "duration": 120.0,
        }
    )
    areas = {s["area"] for s in out}
    assert "Stereo" in areas and "Harmony" in areas
    loop = [s for s in out if s["area"] == "Harmony"][0]
    assert "Dm" in loop["why"] and "14" in loop["why"]


def test_suggestion_cap_and_severity_order():
    from workers.analysis.suggestions import MAX_SUGGESTIONS, build_suggestions

    ctx = {
        "mix": {
            "sub_pct": 30.0, "low_pct": 50.0, "high_pct": 40.0, "air_pct": 0.5,
            "peak_dbfs": -0.05, "crest_db": 3.0,
            "is_stereo": True, "stereo_correlation": 0.05,
        },
        "feats": {"percussive_share": 0.8, "harmonic_share": 0.9},
        "vocal_profile": {"autotune_likelihood": 0.9, "voiced_fraction": 0.2},
        "key_confidence": 0.2,
        "unique_chords": ["C"],
        "chord_events": 20,
        "duration": 200.0,
    }
    out = build_suggestions(ctx)
    assert len(out) <= MAX_SUGGESTIONS
    ranks = {"high": 0, "medium": 1, "low": 2}
    severities = [ranks[s["severity"]] for s in out]
    assert severities == sorted(severities)
    # Every suggestion quotes a number in its why.
    assert all(any(ch.isdigit() for ch in s["why"]) for s in out)


def test_measure_levels_and_stereo():
    from workers.analysis.measure import level_stats, stereo_stats

    tt = np.arange(22050) / 22050
    y = 0.5 * np.sin(2 * np.pi * 220 * tt)
    stats = level_stats(y)
    assert stats["peak_dbfs"] == pytest.approx(-6.0, abs=0.2)
    assert stats["crest_db"] == pytest.approx(3.0, abs=0.3)

    both = np.vstack([y, y])
    st = stereo_stats(both)
    assert st["is_stereo"] is True
    assert st["stereo_correlation"] == pytest.approx(1.0, abs=0.01)
    assert st["stereo_width"] == pytest.approx(0.0, abs=0.01)

    inverted = np.vstack([y, -y])
    assert stereo_stats(inverted)["stereo_correlation"] == pytest.approx(-1.0, abs=0.01)
    assert stereo_stats(y)["is_stereo"] is False


# ------------------------------------------------------------- Phase H

def test_adjust_math_and_clamps():
    from backend.adjust import cache_filename, clamp_adjust, effective_bpm, transpose_key

    assert clamp_adjust(99, 5.0) == (12.0, 2.0)
    assert clamp_adjust(-99, 0.1) == (-12.0, 0.5)
    assert clamp_adjust(2, 1.1) == (2.0, 1.1)
    assert transpose_key("C", 2) == "D"
    assert transpose_key("B", 1) == "C"
    assert transpose_key("A", -3) == "F#"
    assert effective_bpm(100.0, 1.1) == 110.0
    name = cache_filename("track-abc123", 3.0, 1.25)
    assert name == "track-abc123_st+3.00_sp1.250.wav"
    assert "/" not in name and ".." not in name


def test_adjusted_meta_endpoint():
    _seed_compared_tracks()
    r = client.get("/tracks/cmp-a/adjusted/meta", params={"semitones": 2, "speed": 1.1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["effective_bpm"] == 110.0
    assert body["effective_key"] == "D major"  # C major transposed +2 semitones
    assert body["derived"] is True


def test_adjusted_audio_endpoint_renders_and_caches():
    r = _upload("Adjust Me", _synth_wav(seconds=4, freq=261.63), "adj.wav")
    assert r.status_code == 202, r.text
    track_id = r.json()["track"]["id"]
    _wait_job(r.json()["job_id"])

    resp = client.get(f"/tracks/{track_id}/adjusted", params={"semitones": 3, "speed": 1.0})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("audio/")
    assert resp.headers["x-adjusted-semitones"] == "+3.00"
    original = client.get(f"/tracks/{track_id}/audio")
    assert resp.content != original.content  # a real render, not the source

    from backend.config import settings

    cache = Path(settings.storage_dir) / "adjusted" / f"{track_id}_st+3.00_sp1.000.wav"
    assert cache.is_file()

    # Clamped request -> clamped header, same cache discipline.
    resp2 = client.get(f"/tracks/{track_id}/adjusted", params={"semitones": 99})
    assert resp2.headers["x-adjusted-semitones"] == "+12.00"

    # The original analysis is untouched by rendering.
    before = client.get(f"/tracks/{track_id}/analysis").json()["telemetry_json"]
    assert before["bpm"] == client.get(f"/tracks/{track_id}/analysis").json()["telemetry_json"]["bpm"]
    again = client.get(f"/tracks/{track_id}/adjusted", params={"semitones": 3, "speed": 1.0})
    assert again.content == resp.content  # cache hit serves the same render
    after = client.get(f"/tracks/{track_id}/analysis").json()["telemetry_json"]
    assert after == before
