"""Phase E (overnight deep-dive): artwork content sniffing coverage.

Before this pass, artwork was gated by extension + size only (the gap
SECURITY.md listed as deferred). These tests pin the magic-byte check
on both artwork routes: correctly-typed art is accepted and served;
mislabeled or non-image content is a clean 400 and nothing is stored.
"""
import base64
import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from backend.security import sniff_image

client = TestClient(app, raise_server_exceptions=False)

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
JPEG_HEAD = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP_HEAD = b"RIFF" + (20).to_bytes(4, "little") + b"WEBP" + b"\x00" * 8


def _synth_wav(seconds=2, sr=22050) -> bytes:
    tt = np.arange(int(seconds * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * 261.63 * tt)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def test_sniff_image_kinds():
    assert sniff_image(PNG_1PX) == "png"
    assert sniff_image(JPEG_HEAD) == "jpeg"
    assert sniff_image(WEBP_HEAD) == "webp"
    assert sniff_image(b"definitely not an image at all") is None
    assert sniff_image(b"\x89PNG") is None  # too short to trust


def _upload_with_artwork(art_name: str, art_bytes: bytes, art_type: str):
    return client.post(
        "/tracks/upload",
        data={"title": "Art Test", "artist": "Test Bot"},
        files={
            "audio": ("t.wav", _synth_wav(), "audio/wav"),
            "artwork": (art_name, art_bytes, art_type),
        },
    )


def test_upload_accepts_real_png_artwork():
    r = _upload_with_artwork("cover.png", PNG_1PX, "image/png")
    assert r.status_code == 202, r.text
    track_id = r.json()["track"]["id"]
    assert r.json()["track"]["artwork_path"]
    art = client.get(f"/tracks/{track_id}/artwork")
    assert art.status_code == 200
    assert art.headers["content-type"] == "image/png"
    assert art.content == PNG_1PX


def test_upload_rejects_mislabeled_artwork():
    # PNG content wearing a .jpg name: content wins, upload rejected.
    r = _upload_with_artwork("cover.jpg", PNG_1PX, "image/jpeg")
    assert r.status_code == 400, r.text
    assert "Artwork content" in r.json()["detail"]


def test_upload_rejects_non_image_artwork():
    r = _upload_with_artwork("cover.png", b"plain text, not a png........", "image/png")
    assert r.status_code == 400, r.text


def test_artwork_endpoint_rejects_mismatch_and_stores_nothing():
    created = client.post(
        "/tracks",
        json={"title": "Bare", "artist": "T", "original_file_path": "/storage/x.mp3"},
    )
    track_id = created.json()["id"]
    r = client.post(
        f"/tracks/{track_id}/artwork",
        files={"artwork": ("cover.png", JPEG_HEAD, "image/png")},
    )
    assert r.status_code == 400, r.text
    after = client.get(f"/tracks/{track_id}").json()
    assert after["artwork_path"] is None

    ok = client.post(
        f"/tracks/{track_id}/artwork",
        files={"artwork": ("cover.webp", WEBP_HEAD, "image/webp")},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["artwork_path"].endswith(".webp")
