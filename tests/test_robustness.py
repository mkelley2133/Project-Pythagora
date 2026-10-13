"""Robustness regressions from the 2026-10-12 overnight pass.

Found by edge-case uploads against a live server:
- a silent wav upload crashed the pipeline with
  `'NoneType' object has no attribute 'rsplit'` (estimate_key had no
  finite correlation to report and returned a None key into a string
  split). Silence now fails the job with an honest, readable reason.
- an ffmpeg conversion failure surfaced the raw subprocess command
  (server paths included) as the job error. It now surfaces a clean
  "Audio conversion failed" message.
"""
from __future__ import annotations

import io
import subprocess
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from workers.analysis import audio as audio_mod
from workers.pipeline import _ensure_wav

client = TestClient(app)


def _silent_wav(seconds: int = 3, sr: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(int(seconds * sr), dtype=np.int16).tobytes())
    return buf.getvalue()


def test_estimate_key_on_silence_is_undetermined_not_crash():
    y = np.zeros(22050 * 3, dtype=np.float32)
    out = audio_mod.estimate_key(y, 22050)
    assert out["key"] is None
    assert out["confidence"] == 0.0


def test_mood_label_tolerates_undetermined_key():
    feats = {
        "duration": 3.0,
        "sub_bass_share": 0.0,
        "percussive_share": 0.0,
        "harmonic_share": 0.0,
        "spectral_centroid": 0.0,
        "spectral_rolloff": 0.0,
        "zero_crossing_rate": 0.0,
        "spectral_contrast": [0.0] * 7,
        "onset_density": 0.0,
        "energy": 0.0,
    }
    out = audio_mod.mood_label(feats, None, {"key": None, "confidence": 0.0})
    assert out["mood"]  # a label, no exception


def test_silent_upload_job_fails_with_honest_reason():
    r = client.post(
        "/tracks/upload",
        data={"title": "Silence", "artist": "Test Bot"},
        files={"audio": ("silent.wav", _silent_wav(), "audio/wav")},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + 120
    job = None
    while time.time() < deadline:
        job = client.get(f"/tracks/jobs/{job_id}").json()
        if job["state"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job is not None and job["state"] == "failed", job
    error = job["error"] or ""
    assert "key" in error.lower()
    assert "rsplit" not in error and "NoneType" not in error


def test_ensure_wav_conversion_failure_is_a_clean_error(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise subprocess.CalledProcessError(228, args[0] if args else "ffmpeg")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr("workers.pipeline.shutil.which", lambda _: "/usr/bin/ffmpeg")
    src = tmp_path / "broken.m4a"
    src.write_bytes(b"not really m4a")
    with pytest.raises(RuntimeError, match="Audio conversion failed"):
        _ensure_wav(str(src))
