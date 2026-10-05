# Project Pythagoras

The **Zero-Hallucination Deep Musicology & Distribution Agent** — an AI system
for independent artists and musicologists that bridges deterministic signal
processing with constrained generative AI. Mathematical signal processing is
completely separated from generative text, so musical attributes are never
guessed by an AI: ground-truth telemetry is computed and locked down before
any narrative interpretation occurs.

## What it does

- **Ingestion & audio separation** — isolates vocals from instrumentation so
  downstream analysis isn't skewed by background noise.
- **Deterministic harmonic & structural analysis** — exact key, mode, tempo,
  and structural segments (intro/verse/chorus/bridge), chord recognition,
  functional harmony, and Roman-numeral progressions.
- **Verbatim transcription & rhyme mapping** — word-level timestamped lyrics
  with programmatic line-ending phoneme analysis for exact rhyme schemes.
- **Emotional & mood profiling** — hard numerical confidence scores for
  multi-label emotional classifications.
- **Vocal inflection & production analysis** — F0 variation, vibrato rate,
  pitch-bend range, pitch-correction artifacts, room acoustics.
- **Hierarchical genre classification** — nested styles, sub-genres, and
  instrument taxonomies cross-referenced with lyrical motifs.
- **Grounded musicological storytelling** — immutable telemetry synthesized
  into literary essays (narrative arc, lyrical-sonic dissonance, rhyme
  architecture, intro-to-outro journey).
- **Streaming-style dashboard** — persistent track bundles in scrollable
  carousels, a dual audio player with structural timeline waveforms, and an
  interactive time-synced lyric viewer with clickable rhyme-scheme tags.

## Repository layout

```
app/            FastAPI application entrypoint
backend/
  config.py     Settings — everything secret comes from the environment
  database.py   SQLAlchemy session manager (graceful fallback without it)
  models/       Pydantic schemas (track + telemetry bundles)
  routers/      API routes (/tracks, /tracks/{id}/analysis)
workers/        Celery background tasks (audio analysis pipeline)
training/       Model training pipeline configuration
tests/          pytest suite
frontend/       Starter dashboard shell (vanilla JS)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in real values; .env is git-ignored
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs for the interactive API.

Background workers (needs Redis running):

```bash
celery -A workers.tasks.celery_app worker --loglevel=info
```

Frontend shell: serve `frontend/` with any static server and point it at the
API with `window.PYTHAGORAS_API_BASE` before `src/main.js` loads:

```html
<script>window.PYTHAGORAS_API_BASE = 'http://127.0.0.1:8000';</script>
```

## API

| Method | Route                     | Description                        |
| ------ | ------------------------- | ---------------------------------- |
| GET    | `/`                       | Service banner                     |
| GET    | `/health`                 | Dependency availability report     |
| GET    | `/tracks`                 | List tracks                        |
| POST   | `/tracks`                 | Register a track (201, id assigned) |
| GET    | `/tracks/{id}`            | Get one track                      |
| GET    | `/tracks/{id}/analysis`   | Telemetry bundle for a track       |

Run the tests with `pytest`.

## Environment variables

| Variable           | Default                              | Purpose              |
| ------------------ | ------------------------------------ | -------------------- |
| `DATABASE_URL`     | `postgresql+psycopg://localhost:5432/pythagoras` | Postgres connection |
| `REDIS_URL`        | `redis://localhost:6379/0`           | Celery broker/backend |
| `MINIO_ENDPOINT`   | `http://localhost:9000`              | Object storage       |
| `MINIO_ACCESS_KEY` | `minioadmin`                         | Object storage       |
| `MINIO_SECRET_KEY` | `minioadmin`                         | Object storage       |
| `MINIO_BUCKET`     | `project-pythagoras`                 | Object storage       |

## Roadmap

- Real audio ingestion + Demucs-style source separation worker
- librosa/music21 deterministic analysis workers feeding `TelemetryBundle`
- faster-whisper transcription worker with word-level timestamps
- SQLAlchemy models replacing the in-memory track store
- Constrained-essay generation stage (telemetry-locked prompting)
- Dashboard: carousels, dual player, waveform timeline, synced lyric viewer
