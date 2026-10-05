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
frontend/       Dashboard: library carousels, dual-deck player, structural
              waveform timeline, time-synced lyrics (vanilla JS + Web Audio;
              demo audio is synthesized live from measured telemetry)
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

## Live demo (GitHub Pages, free)

The dashboard is a static site, so it deploys to GitHub Pages for free from
the `gh-pages` branch (which mirrors `frontend/` at the branch root).

One-time setup:

1. Open the repo on GitHub → **Settings** → **Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Select branch **gh-pages** and folder **/ (root)**, then **Save**.
4. The site goes live at `https://<your-username>.github.io/Project-Pythagora/`
   within a minute or two.

To update the live site later, push fresh copies of the `frontend/` files to
the `gh-pages` branch.

Note: Pages hosts static files only, so the FastAPI backend doesn't run
there. The dashboard detects this and runs in **demo mode** — the full
interactive UI works, with audio synthesized live from the demo track's
telemetry. To go fully live later, host the API somewhere (e.g. Render's free
tier) and point the page at it with:

```html
<script>window.PYTHAGORAS_API_BASE = 'https://your-api-host';</script>
```
