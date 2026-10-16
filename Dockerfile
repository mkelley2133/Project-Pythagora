# Project Pythagoras — one image, two entrypoints (see docker-compose.yml):
# the FastAPI app (default CMD) and the Celery worker (command override).
# Base matches the repo's development runtime (Python 3.12).
FROM python:3.12-slim

# ffmpeg: the pipeline converts non-wav uploads to wav with it
# (workers/pipeline.py _ensure_wav) and librosa's audioread fallback
# needs it too. libsndfile1: soundfile runtime. curl: healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch first: the default PyPI wheel bundles CUDA libraries
# (several GB) that a CPU inference service never loads. Installing it
# ahead of requirements.txt leaves torch>=2.4 already satisfied there.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
    && pip install -r requirements.txt

COPY app ./app
COPY backend ./backend
COPY workers ./workers
COPY frontend ./frontend
COPY tools ./tools

# Model weights and the track library are NOT baked into the image
# (both are gitignored; the weights alone are ~1 GB). docker-compose
# mounts ./models and ./storage from the host. Without weights the
# genre/subgenre stages degrade gracefully by design; without the
# storage mount the library starts empty.
RUN useradd --create-home pythagora && chown -R pythagora /app
USER pythagora

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
