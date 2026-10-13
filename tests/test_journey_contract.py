"""Phase D (overnight pass): upload-journey + UI/backend contract tests.

Pins the adjusted-render M4A regression found live (every adjusted
render of an .m4a upload returned HTTP 500 — libsndfile cannot decode
AAC and no audioread backend saved it) and the error-shape contract
the dashboard depends on: every failure is JSON with a `detail`, never
an HTML error page.
"""
import io
import shutil
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)

FFMPEG = shutil.which("ffmpeg")


def _synth_wav(seconds=4, sr=22050, freq=261.63) -> bytes:
    tt = np.arange(int(seconds * sr)) / sr
    y = 0.4 * np.sin(2 * np.pi * freq * tt)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _m4a_bytes(tmp_path: Path) -> bytes:
    wav_path = tmp_path / "src.wav"
    wav_path.write_bytes(_synth_wav())
    m4a_path = tmp_path / "src.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
         "-c:a", "aac", str(m4a_path)],
        check=True,
    )
    return m4a_path.read_bytes()


def _wait_job(job_id: str, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/tracks/jobs/{job_id}").json()
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


# ------------------------------------------------------------- the M4A bug

@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_render_adjusted_handles_m4a_container(tmp_path):
    from backend.adjust import render_adjusted

    src = tmp_path / "song.m4a"
    src.write_bytes(_m4a_bytes(tmp_path))
    out = render_adjusted(src, tmp_path / "out" / "adj.wav", 2.0, 1.0)

    import soundfile as sf

    info = sf.info(str(out))
    assert 3.5 < info.frames / info.samplerate < 4.5  # ~4s in, ~4s out
    # No decode temp files are left behind.
    assert not list((tmp_path / "out").glob("tmp*.wav"))


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_adjusted_endpoint_renders_for_m4a_upload(tmp_path):
    r = client.post(
        "/tracks/upload",
        data={"title": "M4A Journey", "artist": "Test Bot"},
        files={"audio": ("song.m4a", _m4a_bytes(tmp_path), "audio/mp4")},
    )
    assert r.status_code == 202, r.text
    job = _wait_job(r.json()["job_id"])
    assert job["state"] == "done", job
    track_id = r.json()["track"]["id"]

    resp = client.get(
        f"/tracks/{track_id}/adjusted", params={"semitones": 2, "speed": 1.0}
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("audio/")
    assert len(resp.content) > 10_000


# ------------------------------------------------------------- error contract

def _assert_json_error(resp, status):
    assert resp.status_code == status, resp.text[:200]
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json().get("detail")


def test_missing_resources_are_json_404():
    _assert_json_error(client.get("/tracks/track-nope"), 404)
    _assert_json_error(client.get("/tracks/track-nope/analysis"), 404)
    _assert_json_error(client.get("/tracks/jobs/job-nope"), 404)
    _assert_json_error(client.get("/tracks/track-nope/report"), 404)
    _assert_json_error(client.get("/tracks/track-nope/audio"), 404)
    _assert_json_error(client.get("/tracks/track-nope/artwork"), 404)
    _assert_json_error(client.get("/tracks/track-nope/adjusted/meta"), 404)
    _assert_json_error(client.delete("/projects/proj-nope"), 404)


def test_compare_error_cases_are_clean():
    _assert_json_error(client.get("/tracks/compare", params={"a": "demo-track-001"}), 422)
    _assert_json_error(
        client.get("/tracks/compare", params={"a": "demo-track-001", "b": "track-nope"}),
        404,
    )


def test_adjusted_bad_params_are_422_not_500():
    _assert_json_error(
        client.get("/tracks/demo-track-001/adjusted", params={"semitones": "abc"}),
        422,
    )


def test_adjusted_meta_clamps_extreme_params():
    r = client.get(
        "/tracks/demo-track-001/adjusted/meta", params={"semitones": 99, "speed": 50}
    )
    assert r.status_code == 200, r.text
    assert r.json()["semitones"] == 12.0
    assert r.json()["speed"] == 2.0


def test_upload_rejections_are_clean_json():
    bad_ext = client.post(
        "/tracks/upload",
        data={"title": "x"},
        files={"audio": ("x.exe", b"MZ junk", "application/octet-stream")},
    )
    _assert_json_error(bad_ext, 400)

    corrupt = client.post(
        "/tracks/upload",
        data={"title": "x"},
        files={"audio": ("fake.mp3", b"definitely not audio", "audio/mpeg")},
    )
    _assert_json_error(corrupt, 400)

    oversize = client.post(
        "/tracks/upload",
        data={"title": "big"},
        files={"audio": ("big.wav", b"\0" * (51 * 1024 * 1024), "audio/wav")},
    )
    _assert_json_error(oversize, 413)
