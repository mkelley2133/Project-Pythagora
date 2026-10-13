"""Upload loop: multipart upload -> queued pipeline -> job tracking -> analysis.

Builds a 12s synthesized C-F-G progression so the pipeline has real audio
to measure. V1 feature set: no stem separation, no automatic lyric
transcription — the pipeline analyzes the full mix.
"""
import io
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _synth_wav(seconds=12, sr=22050) -> bytes:
    chords = [(261.63, 329.63, 392.00), (349.23, 440.00, 523.25), (392.00, 493.88, 587.33)]
    per = seconds / len(chords)
    y = np.zeros(int(seconds * sr))
    for i, (r, t, f) in enumerate(chords):
        n = int(per * sr)
        tt = np.arange(n) / sr
        seg = (
            np.sin(2 * np.pi * r * tt)
            + 0.7 * np.sin(2 * np.pi * t * tt)
            + 0.5 * np.sin(2 * np.pi * f * tt)
        ) / 2.2
        y[i * n:(i + 1) * n] = seg
    # 120 BPM kick pulse so tempo is measurable
    kick_n = int(0.09 * sr)
    kt = np.arange(kick_n) / sr
    kick = np.sin(2 * np.pi * 55 * kt) * np.exp(-kt * 35)
    for beat in range(int(seconds * 2)):
        s = int(beat * 0.5 * sr)
        y[s:s + kick_n] += kick[: max(0, min(kick_n, len(y) - s))]
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
        r = client.get(f"/tracks/jobs/{job_id}")
        assert r.status_code == 200
        job = r.json()
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(1.0)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_upload_starts_pipeline_and_produces_analysis():
    wav = _synth_wav()
    r = client.post(
        "/tracks/upload",
        data={"title": "Pipeline Test", "artist": "Test Bot"},
        files={"audio": ("test.wav", wav, "audio/wav")},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    track_id, job_id = body["track"]["id"], body["job_id"]
    assert body["track"]["title"] == "Pipeline Test"

    job = _wait_job(job_id)
    assert job["state"] == "done", job
    assert job["progress"] == 100

    r = client.get(f"/tracks/{track_id}/analysis")
    assert r.status_code == 200, r.text
    telemetry = r.json()["telemetry_json"]
    assert telemetry["bpm"] > 0
    assert telemetry["musical_key"] in {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    }
    assert len(telemetry["waveform_peaks"]) > 0
    assert telemetry["musicological_essay"]  # deterministic summary
    assert len(telemetry["segments"]) > 0


def test_upload_with_artwork_and_artwork_endpoint():
    wav = _synth_wav(seconds=4)
    # Real PNG magic bytes — artwork is content-sniffed since the
    # deep-dive pass (see tests/test_deep_dive.py for rejection cases).
    png = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload"
    r = client.post(
        "/tracks/upload",
        data={"title": "Art Test", "artist": "Test Bot"},
        files={
            "audio": ("test.wav", wav, "audio/wav"),
            "artwork": ("cover.png", png, "image/png"),
        },
    )
    assert r.status_code == 202, r.text
    track_id, job_id = r.json()["track"]["id"], r.json()["job_id"]
    assert r.json()["track"]["artwork_path"]

    r = client.get(f"/tracks/{track_id}/artwork")
    assert r.status_code == 200
    assert r.content == png

    _wait_job(job_id)


def test_artwork_upload_to_existing_track():
    r = client.post(
        "/tracks",
        json={"title": "Bare", "artist": "X", "original_file_path": "/storage/bare.mp3"},
    )
    track_id = r.json()["id"]
    r = client.post(
        f"/tracks/{track_id}/artwork",
        files={"artwork": ("c.jpg", b"\xff\xd8\xff\xe0" + b"jpeg-payload", "image/jpeg")},
    )
    assert r.status_code == 200
    assert r.json()["artwork_path"]
    r = client.get(f"/tracks/{track_id}/artwork")
    assert r.status_code == 200


def test_job_not_found():
    assert client.get("/tracks/jobs/nope").status_code == 404


def test_upload_rejects_bad_audio_type():
    r = client.post(
        "/tracks/upload",
        data={"title": "Bad"},
        files={"audio": ("x.exe", b"nope", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_audio_endpoint_serves_uploaded_file():
    wav = _synth_wav(seconds=4)
    r = client.post(
        "/tracks/upload",
        data={"title": "Audio Test", "artist": "Test Bot"},
        files={"audio": ("a.wav", wav, "audio/wav")},
    )
    assert r.status_code == 202
    track_id, job_id = r.json()["track"]["id"], r.json()["job_id"]
    _wait_job(job_id)
    r = client.get(f"/tracks/{track_id}/audio")
    assert r.status_code == 200
    assert r.content == wav


def test_audio_404_when_file_missing():
    r = client.post(
        "/tracks",
        json={"title": "Ghost", "artist": "X", "original_file_path": "/storage/nope.mp3"},
    )
    assert client.get(f"/tracks/{r.json()['id']}/audio").status_code == 404


def test_pipeline_v1_feature_set():
    """V1: the pipeline analyzes the full mix — no stems, no automatic
    transcription — and reports mix measurements + suggestions."""
    wav = _synth_wav(seconds=6)
    r = client.post(
        "/tracks/upload",
        data={"title": "V1 Set", "artist": "Test Bot"},
        files={"audio": ("v1.wav", wav, "audio/wav")},
    )
    assert r.status_code == 202, r.text
    track_id, job_id = r.json()["track"]["id"], r.json()["job_id"]
    job = _wait_job(job_id)
    assert job["state"] == "done"

    telemetry = client.get(f"/tracks/{track_id}/analysis").json()["telemetry_json"]
    assert telemetry["stems"] == []
    assert telemetry["separation_model"] is None
    assert telemetry["timestamped_lyrics"] == []
    assert telemetry["lyrics_language"] is None
    assert telemetry["lyrics_note"] is None
    # Phase G outputs are present and measured.
    assert telemetry["mix_measurements"] is not None
    assert telemetry["mix_measurements"]["peak_dbfs"] <= 0.0
    assert isinstance(telemetry["suggestions"], list)

    # The stems endpoint is gone in v1.
    assert client.get(f"/tracks/{track_id}/stems/vocals").status_code == 404


def test_audio_head_probe_used_by_player():
    """The frontend probes audio availability with HEAD; it must not 404."""
    wav = _synth_wav(seconds=4)
    r = client.post(
        "/tracks/upload",
        data={"title": "Head Test", "artist": "Test Bot"},
        files={"audio": ("h.wav", wav, "audio/wav")},
    )
    track_id, job_id = r.json()["track"]["id"], r.json()["job_id"]
    _wait_job(job_id)
    h = client.head(f"/tracks/{track_id}/audio")
    assert h.status_code == 200, "HEAD probe failed — player falls back to synth"
