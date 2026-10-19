import logging
import os
import threading
from functools import lru_cache
from importlib import import_module
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import security
from backend.config import settings
from backend.routers.projects import router as projects_router
from backend.routers.tracks import router as tracks_router

logger = logging.getLogger("pythagoras.api")


def _warm_models() -> None:
    """Pre-load the genre/subgenre/emotion model singletons (best effort).

    First-track latency otherwise includes the model loads (~20 s).
    Runs on a daemon thread so boot is never blocked; a failed warmup
    changes nothing — the stages degrade lazily exactly as before.
    """
    try:
        from workers.analysis import emotion, genre, subgenre

        genre._load_model()
        subgenre._load_model()
        emotion._load_model()
        logger.info("Model warmup complete (genre + subgenre + emotion singletons loaded).")
    except Exception:  # noqa: BLE001 - warmup is an optimization, never fatal
        logger.exception("Model warmup failed; models will load lazily.")


# Opt-in via PYTHAGORA_WARMUP_MODELS=1 (default off so tests and the
# demo stay light). The thread starts at import: uvicorn imports this
# module before serving, so the models are typically ready before the
# first upload lands.
if os.environ.get("PYTHAGORA_WARMUP_MODELS", "") == "1":
    threading.Thread(target=_warm_models, daemon=True, name="pythagora-warmup").start()

# debug stays off: no stack traces or interactive docs internals leak into
# production error responses (see the 500 handler below and SECURITY.md).
app = FastAPI(title=settings.app_name, version=settings.app_version, debug=False)
app.include_router(tracks_router)
app.include_router(projects_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log the real error server-side; return a generic JSON 500."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# Middleware (last added runs outermost):
#   SecurityHeaders — applied to every response, including errors.
#   RequestSizeLimit — rejects oversized bodies before parsing.
#   ApiKey — enforces PYTHAGORA_API_KEY on /tracks/* when set.
#   CORS — explicit origins from env (PYTHAGORA_CORS_ORIGINS); no wildcards.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.add_middleware(security.ApiKeyMiddleware)
app.add_middleware(
    security.RequestSizeLimitMiddleware,
    max_bytes=settings.max_upload_bytes + settings.max_artwork_bytes + 1024 * 1024,
)
app.add_middleware(security.SecurityHeadersMiddleware)


def _is_importable(name: str) -> bool:
    try:
        import_module(name)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _available_packages() -> tuple[str, ...]:
    """Import checks are cached so /health doesn't re-import torch/librosa
    on every call."""
    modules = [
        "fastapi",
        "celery",
        "redis",
        "psycopg",
        "boto3",
        "minio",
        "librosa",
        "pydantic",
        "pytest",
        "requests",
    ]
    return tuple(name for name in modules if _is_importable(name))


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "available_packages": list(_available_packages()),
    }


@app.get("/config")
def client_config() -> dict:
    """Public boot config for the dashboard (and for humans troubleshooting
    with curl): proves this origin is the real backend, not a static host."""
    return {
        "served_by": "backend",
        "app": settings.app_name,
        "version": settings.app_version,
        "api_key_required": bool(settings.api_key),
    }


# ---------------------------------------------------------------------------
# Dashboard serving with a LIVE marker.
#
# The frontend's mode logic keys off one unmistakable fact: was this page
# served by the backend? StaticFiles would serve index.html verbatim, so
# the root document goes through this route instead, which injects
#   <meta name="pythagoras-served-by" content="backend">
# into <head>. When that marker is present the dashboard must NEVER fall
# back to demo data — it retries the API until it answers (Codespaces cold
# starts can take a while) and shows a LIVE badge once connected. The raw
# file on disk deliberately does NOT contain the marker, so GitHub Pages
# (which serves the raw file) can never fake live mode.
# ---------------------------------------------------------------------------
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_INDEX_HTML = _FRONTEND_DIR / "index.html"
_LIVE_MARKER = '<meta name="pythagoras-served-by" content="backend" />'


def _dashboard_html() -> str:
    html = _INDEX_HTML.read_text(encoding="utf-8")
    if 'name="pythagoras-served-by"' not in html:
        html = html.replace("</head>", f"    {_LIVE_MARKER}\n  </head>", 1)
    return html


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    # no-cache: the marker (and any future boot config) must never go stale.
    return HTMLResponse(_dashboard_html(), headers={"Cache-Control": "no-cache"})


# Serve the dashboard's static assets from the API itself: one process, one
# URL (http://127.0.0.1:8000) — no mixed-content issues, uploads actually
# analyze. Mounted last so every API route (/tracks/*, /health, /, …)
# matches first; the mount serves styles/scripts/images only in practice,
# since "/" and "/index.html" are claimed by the route above.
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
