"""Edge-case battery from the 2026-10-13 robustness probes.

Pins the contracts the probes established against the live API:
- a DC (constant, pitch-free) signal fails honestly at the key stage
  instead of completing with an invented key;
- odd sample rates, unicode filenames, 24-bit stereo, and FLAC
  containers analyze end to end;
- a truncated WAV analyzes only its surviving prefix and reports that
  shorter duration honestly (libsndfile clamps the frame count);
- load_audio covers the full 900 s upload cap (it silently truncated
  at 300 s before the efficiency pass).
"""

import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SR = 22050


def _tone(seconds=3.0, sr=SR, freq=440.0, amp=0.5):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _wav_bytes(y, sr=SR, subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def _run_job(client, title, data, filename, mime="audio/wav"):
    r = client.post(
        "/tracks/upload",
        data={"title": title, "artist": "Edge"},
        files={"audio": (filename, data, mime)},
    )
    assert r.status_code == 202, r.text
    payload = r.json()
    job_id, track = payload["job_id"], payload["track"]
    deadline = time.time() + 240
    job = None
    while time.time() < deadline:
        job = client.get(f"/tracks/jobs/{job_id}").json()
        if job["state"] in ("done", "failed"):
            break
        time.sleep(0.25)
    assert job is not None and job["state"] in ("done", "failed")
    return job, track


def test_dc_signal_fails_honestly():
    dc = np.full(SR * 3, 0.3, dtype=np.float32)
    job, _track = _run_job(client, "DC", _wav_bytes(dc), "dc.wav")
    assert job["state"] == "failed"
    assert "key" in (job["error"] or "").lower()


def test_estimate_key_refuses_pitch_free_signals():
    from workers.analysis.audio import estimate_key

    assert estimate_key(np.full(SR, 0.3, dtype=np.float32), SR)["key"] is None
    assert estimate_key(np.zeros(SR, dtype=np.float32), SR)["key"] is None
    assert estimate_key(_tone(2.0), SR)["key"] is not None


def test_odd_sample_rate_and_unicode_filename_complete():
    job, _track = _run_job(client, "OddSR", _wav_bytes(_tone(3.0, sr=8000), sr=8000), "töne ♪.wav")
    assert job["state"] == "done", job


def test_stereo_24bit_and_flac_complete():
    stereo = np.stack([_tone(3.0), _tone(3.0, freq=554.37)], axis=1)
    job, _track = _run_job(client, "Stereo24", _wav_bytes(stereo, subtype="PCM_24"), "stereo.wav")
    assert job["state"] == "done", job

    buf = io.BytesIO()
    sf.write(buf, _tone(3.0), SR, format="FLAC")
    job, _track = _run_job(client, "FLAC", buf.getvalue(), "tone.flac", mime="audio/flac")
    assert job["state"] == "done", job


def test_truncated_wav_reports_honest_duration():
    full = _wav_bytes(_tone(6.0))
    truncated = full[: int(len(full) * 0.4)]
    job, track = _run_job(client, "Truncated", truncated, "truncated.wav")
    assert job["state"] == "done", job
    bundle = client.get(f"/tracks/{track['id']}/analysis").json()["telemetry_json"]
    # surviving prefix only (~2.4 s), not the 6 s the WAV header claims
    assert 2.0 < bundle["duration"] < 3.0


def test_load_audio_covers_full_upload_cap(tmp_path):
    from workers.analysis.audio import load_audio

    path = tmp_path / "long.wav"
    sf.write(str(path), _tone(305.0), SR, format="WAV", subtype="PCM_16")
    y, sr = load_audio(str(path))
    assert len(y) / sr > 300.0
