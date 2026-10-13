"""Security controls for the Pythagoras API.

Implements the controls listed in SECURITY.md (OWASP File Upload Cheat Sheet
+ OWASP API Security Top 10, scoped to a small commercial upload service):

- content sniffing for uploaded audio (magic bytes, never trust extensions)
- decode/duration probing via soundfile, falling back to ffprobe
  (subprocess list-args only — never shell=True with user input)
- server-generated storage names + storage-dir containment checks
- per-IP sliding-window rate limiting (upload + general API scopes)
- optional API-key auth for /tracks/* (env PYTHAGORA_API_KEY, read live so
  it can be rotated/tested without a restart)
- security headers / request-size middleware

Nothing in this module trusts client-supplied filenames, extensions, or
Content-Type headers.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from backend.config import settings

logger = logging.getLogger("pythagoras.security")

# --------------------------------------------------------------------------
# ID validation — every resource id is server-shaped; anything else is a 400.
# --------------------------------------------------------------------------

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_id(value: str, kind: str = "id") -> str:
    if not ID_PATTERN.match(value or ""):
        raise HTTPException(status_code=400, detail=f"Invalid {kind}.")
    return value


# --------------------------------------------------------------------------
# API key (optional). Read from the environment on every request so tests
# and rotations work without restarting the process.
# --------------------------------------------------------------------------


def current_api_key() -> str:
    return os.environ.get("PYTHAGORA_API_KEY") or settings.api_key or ""


def api_key_required() -> bool:
    return bool(current_api_key())


def check_api_key(request: Request) -> None:
    expected = current_api_key()
    if not expected:
        return
    provided = request.headers.get("x-api-key", "")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Enforce the API key on /tracks/* and /projects/* when
    PYTHAGORA_API_KEY is set.

    /health and the static dashboard stay public so the demo keeps working.
    """

    async def dispatch(self, request: Request, call_next):
        gated = request.url.path.startswith(("/tracks", "/projects"))
        if gated and api_key_required():
            provided = request.headers.get("x-api-key", "")
            if not provided or not hmac.compare_digest(provided, current_api_key()):
                return JSONResponse(
                    status_code=401, content={"detail": "Missing or invalid API key."}
                )
        return await call_next(request)


# --------------------------------------------------------------------------
# Rate limiting — per-IP sliding window, in-process (single-node service).
# Limits are plain attributes so tests/operators can tune them live.
# --------------------------------------------------------------------------


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a hit. Returns None when allowed, else seconds to retry."""
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return max(1.0, self.window_seconds - (now - bucket[0]))
            bucket.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


upload_limiter = SlidingWindowLimiter(settings.upload_rate_limit, settings.rate_window_seconds)
api_limiter = SlidingWindowLimiter(settings.api_rate_limit, settings.rate_window_seconds)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limit_upload(request: Request) -> None:
    retry = upload_limiter.check(_client_key(request))
    if retry is not None:
        raise HTTPException(
            status_code=429,
            detail="Upload rate limit exceeded. Try again shortly.",
            headers={"Retry-After": str(int(retry))},
        )


def rate_limit_api(request: Request) -> None:
    retry = api_limiter.check(_client_key(request))
    if retry is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again shortly.",
            headers={"Retry-After": str(int(retry))},
        )


# --------------------------------------------------------------------------
# Content sniffing — magic bytes only; the client filename/extension and
# Content-Type header are never consulted.
# --------------------------------------------------------------------------

# canonical format -> (file extension, media type)
AUDIO_FORMATS: dict[str, tuple[str, str]] = {
    "wav": (".wav", "audio/wav"),
    "mp3": (".mp3", "audio/mpeg"),
    "flac": (".flac", "audio/flac"),
    "ogg": (".ogg", "audio/ogg"),
    "m4a": (".m4a", "audio/mp4"),
    "aac": (".aac", "audio/aac"),
}

_FTYP_BRANDS = {b"M4A ", b"M4B ", b"M4P ", b"mp41", b"mp42", b"isom", b"iso2", b"dash"}


def sniff_audio(data: bytes) -> str | None:
    """Identify an audio format from its leading bytes, or None."""
    if len(data) < 12:
        return None
    if data[0:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[0:4] == b"fLaC":
        return "flac"
    if data[0:4] == b"OggS":
        return "ogg"
    if data[4:8] == b"ftyp" and data[8:12] in _FTYP_BRANDS:
        return "m4a"
    if data[0:3] == b"ID3":
        return "mp3"
    # Raw MPEG audio / ADTS sync: 11-bit frame sync 0xFFF.
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        version = (data[1] >> 3) & 0x03
        layer = (data[1] >> 1) & 0x03
        if version != 0x01:  # reserved version = not a valid frame header
            if layer == 0x00:
                return "aac"  # ADTS AAC has layer bits 00
            bitrate_index = (data[2] >> 4) & 0x0F
            if bitrate_index != 0x0F:
                return "mp3"
    return None


# Extensions each sniffed image kind may legitimately carry.
IMAGE_KIND_EXTS: dict[str, tuple[str, ...]] = {
    "jpeg": (".jpg", ".jpeg"),
    "png": (".png",),
    "webp": (".webp",),
}


def sniff_image(data: bytes) -> str | None:
    """Identify an image format from its leading bytes, or None.

    Magic-byte check only (the project carries no imaging library, so
    there is no decode proof for artwork the way audio gets one); it
    still stops mislabeled or non-image content riding an image
    extension, which is the gap SECURITY.md had listed as deferred.
    """
    if len(data) < 12:
        return None
    if data[0:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[0:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


# --------------------------------------------------------------------------
# Decode + duration probing. soundfile first (no subprocess); ffprobe as a
# fallback for containers libsndfile cannot read (e.g. M4A/AAC).
# --------------------------------------------------------------------------


def probe_audio(path: Path) -> dict:
    """Return {"duration": float seconds, "format": str} or raise ValueError."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames > 0 and info.samplerate > 0:
            return {"duration": info.frames / float(info.samplerate), "format": info.format}
    except Exception:
        pass  # fall through to ffprobe

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        duration = float(result.stdout.strip())
        if duration > 0:
            return {"duration": duration, "format": "ffprobe"}
    except Exception as exc:  # noqa: BLE001 — probe failure means reject
        raise ValueError(f"Audio could not be decoded: {exc}") from exc
    raise ValueError("Audio could not be decoded.")


# --------------------------------------------------------------------------
# Storage containment — uploads live under one dedicated directory with
# server-generated names; stored paths are served only from inside it.
# --------------------------------------------------------------------------


def storage_root() -> Path:
    return Path(settings.storage_dir).resolve()


def resolve_within_storage(stored_path: str) -> Path | None:
    """Resolve a stored file path, returning None unless it is an existing
    file inside the storage directory. This is the traversal guard for the
    audio/artwork serving endpoints: client-registered paths (POST /tracks)
    can point anywhere, but only files we actually stored get served."""
    if not stored_path:
        return None
    try:
        candidate = Path(stored_path).resolve()
        root = storage_root()
        if candidate == root or root not in candidate.parents:
            return None
        if not candidate.is_file():
            return None
        return candidate
    except (OSError, RuntimeError):
        return None


# --------------------------------------------------------------------------
# Middleware: security headers + early request-size rejection.
# --------------------------------------------------------------------------

_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "media-src 'self' blob:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Content-Security-Policy", _CSP)
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if settings.enable_hsts:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized requests from the Content-Length header before the
    body is parsed. Streaming per-file caps still apply during upload."""

    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        # Multipart overhead (artwork + form fields) rides along with audio.
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    return JSONResponse(
                        status_code=413, content={"detail": "Request body too large."}
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400, content={"detail": "Invalid Content-Length header."}
                )
        return await call_next(request)
