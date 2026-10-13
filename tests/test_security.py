"""Security controls: upload caps, content sniffing, traversal safety,
rate limiting, API-key enforcement, security headers, and the report route.

See SECURITY.md for the control list these tests pin down.
"""
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from backend import security
from backend.config import settings

client = TestClient(app)


def _wav_bytes(seconds: float = 2.0, freq: float = 440.0, sr: int = 22050) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    y = 0.4 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


@pytest.fixture()
def patched_settings():
    """Temporarily override frozen Settings fields, restoring afterwards."""
    originals = {}

    def _set(name, value):
        if name not in originals:
            originals[name] = getattr(settings, name)
        object.__setattr__(settings, name, value)

    yield _set
    for name, value in originals.items():
        object.__setattr__(settings, name, value)


@pytest.fixture(autouse=True)
def _reset_limiters():
    security.upload_limiter.reset()
    security.api_limiter.reset()
    yield
    security.upload_limiter.reset()
    security.api_limiter.reset()


def _upload(data: bytes, filename: str = "song.wav", title: str = "Sec Test"):
    return client.post(
        "/tracks/upload",
        data={"title": title, "artist": "Tester"},
        files={"audio": (filename, io.BytesIO(data), "application/octet-stream")},
    )


# ---------------------------------------------------------------- uploads


def test_oversize_upload_rejected_413(patched_settings):
    patched_settings("max_upload_bytes", 2048)
    response = _upload(_wav_bytes(2.0))
    assert response.status_code == 413


def test_non_audio_content_rejected_400():
    response = _upload(b"this is definitely not an audio file", "song.wav")
    assert response.status_code == 400


def test_disguised_file_rejected_by_sniffing_400():
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    response = _upload(png_bytes, "song.mp3")
    assert response.status_code == 400


def test_undecodable_audio_rejected_400():
    fake_wav = b"RIFF" + (100).to_bytes(4, "little") + b"WAVE" + b"\x00" * 64
    response = _upload(fake_wav, "song.wav")
    assert response.status_code == 400


def test_duration_cap_rejected_400(patched_settings):
    patched_settings("max_duration_seconds", 1.0)
    response = _upload(_wav_bytes(4.0))
    assert response.status_code == 400
    assert "duration" in response.json()["detail"].lower()


def test_upload_filename_traversal_is_neutralized(tmp_path):
    response = _upload(_wav_bytes(1.0), filename="../../evil.wav")
    assert response.status_code == 202
    track = response.json()["track"]
    # Stored under a server-generated name inside the storage dir.
    assert "evil" not in track["original_file_path"]
    assert "storage" in track["original_file_path"]
    assert not (tmp_path / "evil.wav").exists()


# ------------------------------------------------------------- traversal


def test_client_registered_path_outside_storage_not_served():
    created = client.post(
        "/tracks",
        json={"title": "T", "artist": "A", "original_file_path": "/etc/passwd"},
    )
    assert created.status_code == 201
    track_id = created.json()["id"]
    response = client.get(f"/tracks/{track_id}/audio")
    assert response.status_code == 404
    assert b"root:" not in response.content


def test_malformed_track_id_rejected():
    response = client.get("/tracks/..%2F..%2Fsecret/analysis")
    assert response.status_code in (400, 404)
    response = client.get("/tracks/bad id!/analysis")
    assert response.status_code == 400


# ----------------------------------------------------------- rate limits


def test_upload_rate_limit_triggers_429():
    original = security.upload_limiter.limit
    security.upload_limiter.limit = 2
    try:
        assert _upload(_wav_bytes(1.0)).status_code == 202
        assert _upload(_wav_bytes(1.0)).status_code == 202
        third = _upload(_wav_bytes(1.0))
        assert third.status_code == 429
        assert "Retry-After" in third.headers
    finally:
        security.upload_limiter.limit = original
        security.upload_limiter.reset()


# --------------------------------------------------------------- API key


def test_api_key_enforced_when_set(monkeypatch):
    monkeypatch.setenv("PYTHAGORA_API_KEY", "test-key-123")
    assert client.get("/tracks").status_code == 401
    assert client.get("/tracks", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.get("/tracks", headers={"X-API-Key": "test-key-123"})
    assert ok.status_code == 200
    # Health stays public so demos/monitoring keep working.
    assert client.get("/health").status_code == 200


def test_api_key_off_by_default(monkeypatch):
    monkeypatch.delenv("PYTHAGORA_API_KEY", raising=False)
    assert client.get("/tracks").status_code == 200


# --------------------------------------------------------------- headers


def test_security_headers_present():
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["referrer-policy"] == "no-referrer"


def test_error_responses_are_json_without_stack_traces():
    response = client.get("/tracks/does-not-exist/analysis")
    assert response.status_code == 404
    assert response.json() == {"detail": "Analysis not found"}
    assert b"Traceback" not in response.content


# ---------------------------------------------------------------- report


def test_report_renders_for_demo_track():
    response = client.get("/tracks/demo-track-001/report")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "PROJECT PYTHAGORAS" in body
    assert "Sample Signal" in body
    assert "Chord Progression" in body
    assert "Vocal Forensics" in body


def test_report_404_for_unknown_track():
    assert client.get("/tracks/does-not-exist/report").status_code == 404
