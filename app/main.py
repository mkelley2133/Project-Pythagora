from functools import lru_cache
from importlib import import_module
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.routers.tracks import router as tracks_router

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(tracks_router)


# Permissive local-dev CORS so the frontend shell (or a Vite dev server)
# can call the API. Tighten `allow_origins` before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "faster_whisper",
        "torch",
        "music21",
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


# Serve the dashboard from the API itself: one process, one URL
# (http://127.0.0.1:8000) — no mixed-content issues, uploads actually analyze.
# Mounted last so every API route (/tracks/*, /health, …) matches first.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
