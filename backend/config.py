"""Central configuration for Project Pythagoras.

All secrets come from the environment — never hardcode credentials in this
file. Copy `.env.example` to `.env` for local development (`.env` is
git-ignored).

Security-relevant knobs (see SECURITY.md):
  PYTHAGORA_API_KEY              — when set, /tracks/* requires X-API-Key.
  PYTHAGORA_CORS_ORIGINS         — comma-separated allowed origins.
  PYTHAGORA_STORAGE_DIR          — dedicated upload storage directory.
  PYTHAGORA_MAX_UPLOAD_BYTES     — audio upload cap (default 50 MiB).
  PYTHAGORA_MAX_ARTWORK_BYTES    — artwork upload cap (default 10 MiB).
  PYTHAGORA_MAX_DURATION_SECONDS — analysis duration cap (default 900s).
  PYTHAGORA_UPLOAD_RATE_LIMIT    — uploads allowed per window per IP.
  PYTHAGORA_API_RATE_LIMIT       — API calls allowed per window per IP.
  PYTHAGORA_RATE_WINDOW_SECONDS  — rate-limit window (default 60).
  PYTHAGORA_MAX_CONCURRENT_JOBS  — simultaneous analysis jobs (default 4).
  PYTHAGORA_ENABLE_HSTS          — "1" to send Strict-Transport-Security.
  PYTHAGORA_WARMUP_MODELS        — "1" pre-loads genre/subgenre models at boot.
  PYTHAGORA_GENRE_DECISIVE       — the decisive primary-genre rule is
                                   the default; "0" restores strict-only
                                   behavior (see workers/analysis/genre.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


_DEV_CORS_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: _env("APP_NAME", "Project Pythagoras"))
    app_version: str = field(default_factory=lambda: _env("APP_VERSION", "0.1.0"))

    # e.g. postgresql+psycopg://pythagoras:secret@localhost:5432/pythagoras
    postgres_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "postgresql+psycopg://localhost:5432/pythagoras")
    )
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    minio_endpoint: str = field(default_factory=lambda: _env("MINIO_ENDPOINT", "http://localhost:9000"))
    minio_access_key: str = field(default_factory=lambda: _env("MINIO_ACCESS_KEY", "minioadmin"))
    minio_secret_key: str = field(default_factory=lambda: _env("MINIO_SECRET_KEY", "minioadmin"))
    minio_bucket: str = field(default_factory=lambda: _env("MINIO_BUCKET", "project-pythagoras"))

    # --- security / commercial-service knobs ---
    api_key: str = field(default_factory=lambda: os.getenv("PYTHAGORA_API_KEY", ""))
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("PYTHAGORA_CORS_ORIGINS", _DEV_CORS_ORIGINS)
    )
    storage_dir: str = field(default_factory=lambda: _env("PYTHAGORA_STORAGE_DIR", "storage"))
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("PYTHAGORA_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)
    )
    max_artwork_bytes: int = field(
        default_factory=lambda: _env_int("PYTHAGORA_MAX_ARTWORK_BYTES", 10 * 1024 * 1024)
    )
    max_duration_seconds: float = field(
        default_factory=lambda: float(_env_int("PYTHAGORA_MAX_DURATION_SECONDS", 900))
    )
    upload_rate_limit: int = field(
        default_factory=lambda: _env_int("PYTHAGORA_UPLOAD_RATE_LIMIT", 30)
    )
    api_rate_limit: int = field(default_factory=lambda: _env_int("PYTHAGORA_API_RATE_LIMIT", 1000))
    rate_window_seconds: int = field(
        default_factory=lambda: _env_int("PYTHAGORA_RATE_WINDOW_SECONDS", 60)
    )
    max_concurrent_jobs: int = field(
        default_factory=lambda: _env_int("PYTHAGORA_MAX_CONCURRENT_JOBS", 4)
    )
    enable_hsts: bool = field(default_factory=lambda: os.getenv("PYTHAGORA_ENABLE_HSTS", "0") == "1")


settings = Settings()
