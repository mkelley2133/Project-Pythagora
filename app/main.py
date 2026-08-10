from importlib import import_module

from fastapi import FastAPI

from backend.routers.tracks import router as tracks_router

app = FastAPI(title="Project Pythagoras", version="0.1.0")
app.include_router(tracks_router)


@app.get("/")
def read_root() -> dict:
    return {
        "message": "Project Pythagora backend is ready",
        "status": "ok",
    }


@app.get("/health")
def health_check() -> dict:
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

    available = []
    for name in modules:
        try:
            import_module(name)
            available.append(name)
        except Exception:
            continue

    return {
        "status": "ok",
        "available_packages": available,
    }
