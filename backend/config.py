"""Central configuration for Project Pythagoras.

All secrets come from the environment — never hardcode credentials in this
file. Copy `.env.example` to `.env` for local development (`.env` is
git-ignored).
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "Project Pythagoras"
    app_version: str = "0.1.0"

    # e.g. postgresql+psycopg://pythagoras:secret@localhost:5432/pythagoras
    postgres_url: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg://localhost:5432/pythagoras"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "project-pythagoras")


settings = Settings()
