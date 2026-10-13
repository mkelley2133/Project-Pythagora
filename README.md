# Project Pythagoras

The **Zero-Hallucination Deep Musicology & Distribution Agent** — an AI system
for independent artists and musicologists that bridges deterministic signal
processing with constrained generative AI. Mathematical signal processing is
completely separated from generative text, so musical attributes are never
guessed by an AI: ground-truth telemetry is computed and locked down before
any narrative interpretation occurs.

## What it does

- **Ingestion** — hardened uploads (content-sniffed, size/duration capped)
  feeding deterministic analysis of the full mix. *V1 scope note: stem
  separation and automatic lyric transcription are deliberately excluded
  from the customer version.*
- **Deterministic harmonic & structural analysis** — exact key, mode, tempo,
  and structural segments (intro/verse/chorus/bridge), chord recognition,
  functional harmony, and Roman-numeral progressions.
- **Supplied lyric sheets & rhyme mapping** — time-synced display of lyric
  sheets supplied with a track, with programmatic rhyme-scheme tags.
  (Lyrics are never auto-transcribed in v1.)
- **Measured mix suggestions** — deterministic rules over band energy,
  levels, dynamics, stereo, and harmony; every suggestion quotes the
  measured value that fired it.
- **Projects & organization** — group tracks into projects, sort by
  name/date/tempo/key/duration, keep project and per-track notes
  (persisted across restarts).
- **Pitch & speed tools** — independent semitone/speed rendered exports
  (librosa) with derived tempo/key readouts, plus a 6-band parametric
  playback EQ in the dashboard.
- **Emotional & mood profiling** — hard numerical confidence scores for
  multi-label emotional classifications.
- **Vocal inflection & production analysis** — F0 variation, vibrato rate,
  pitch-bend range, pitch-correction artifacts, room acoustics.
- **Genre classification (trained model)** — PANNs Cnn10 pretrained on
  AudioSet reports the primary genre with its real output probabilities;
  below a 0.35 top score the genre reads "Mixed / uncertain" rather than
  forcing a label. Deterministic texture/style cues (e.g. sustained-pad
  texture) are reported separately and are never presented as genre.
- **Subgenre matching (zero-shot)** — LAION CLAP compares the audio
  against text descriptions of ~46 candidate subgenres (trap, melodic
  trap, emo rap, drill, shoegaze, ...) and reports the top 5 with
  similarity-share percents. These are similarity rankings, not trained
  probabilities, and are labeled as such everywhere they appear.
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
  models/       Pydantic schemas (track + telemetry bundles)
  routers/      API routes (/tracks, /tracks/{id}/analysis)
workers/        Background pipeline + Celery task wrapper
  tasks.py      healthcheck + analyze_track (same run_pipeline as the API)
  analysis/     deterministic measurement: chroma chord recognition +
                Roman numerals, vocal forensics (F0, vibrato, pitch-
                correction heuristic), waveform peak envelopes, mix
                measurements + rule-fired suggestions
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

### Genre model weights (one-time download)

Genre classification uses the PANNs Cnn10 checkpoint (AudioSet, mAP 0.380,
CC BY 4.0 — Kong et al., arXiv:1912.10211). Download it once into `models/`
(the directory is git-ignored; `PYTHAGORA_GENRE_MODEL` overrides the path):

```bash
mkdir -p models
curl -L -o "models/Cnn10_mAP=0.380.pth" \
  "https://zenodo.org/api/records/3987831/files/Cnn10_mAP=0.380.pth/content"
md5sum "models/Cnn10_mAP=0.380.pth"   # expect bfb1f1f9968938fa8ef4012b8471f5f6
```

Without the file, analyses still complete — the genre fields report the
model as unavailable instead of guessing.

### Subgenre model weights (one-time download)

Subgenre matching is zero-shot: LAION CLAP (`laion/clap-htsat-unfused`,
Apache-2.0, ~618 MB) compares the audio against a text description of
each candidate subgenre and reports similarity shares (not trained
probabilities — the bundle, dashboard, and report all label them as
such). Download it once into `models/clap-htsat/` (git-ignored;
`PYTHAGORA_SUBGENRE_MODEL` overrides the path):

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('laion/clap-htsat-unfused', local_dir='models/clap-htsat')"
sha256sum models/clap-htsat/pytorch_model.bin
# expect 1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb
```

After this one-time download, inference loads strictly from the local
directory (`local_files_only`) — no analysis ever touches the network.
Without the directory, analyses still complete: the subgenre fields
report the model as unavailable. (Do not substitute
`laion/larger_clap_music`: its Hugging Face port is broken — see
LAION-AI/CLAP issue #126 — and produces ~zero similarity for every
prompt.)

Background workers (needs Redis running):

```bash
celery -A workers.tasks.celery_app worker --loglevel=info
```

Frontend shell: serve `frontend/` with any static server and point it at the
API with the `?api=` query parameter (persisted to `localStorage` as
`pythagora_api_base`), e.g. `https://your-static-host/?api=http://127.0.0.1:8000`.
Remember the API must allow that origin via `PYTHAGORA_CORS_ORIGINS`.

## Run in GitHub Codespaces

The dashboard and API are one app — in Codespaces you run the backend and
open its forwarded URL. Do **not** use the GitHub Pages link for real work;
that one is the demo by design.

```bash
cd ~/workspace/Project-Pythagora   # or wherever the repo is checked out
~/workspace/.venvs/pg/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# (in a fresh Codespace without the venv: pip install -r requirements.txt first,
#  then: uvicorn app.main:app --host 0.0.0.0 --port 8000)
```

1. Codespaces will offer to forward port **8000** — accept, then open the
   forwarded `https://<codespace>-8000.app.github.dev` URL (Ports tab →
   right-click port 8000 → Open in Browser). Port visibility can stay
   **Private**; you're signed in, so it just works.
2. The badge at the top right must say **● LIVE** (and the footer reads
   "● LIVE — connected to the Pythagoras API"). First load can sit on
   **● CONNECTING** for a while on a cold Codespace — that's expected:
   a backend-served page retries the API until it answers and will never
   drop into demo on its own.
3. Upload a track. If it analyzes, you're live — demo mode cannot analyze.

**Troubleshooting — if the badge says DEMO:**

- You're probably on the wrong URL (GitHub Pages ends in `github.io` —
  that's always demo). Open the forwarded port-8000 URL instead.
- Check the server terminal: uvicorn must be running with `--host 0.0.0.0`.
- From the Codespace terminal: `curl http://127.0.0.1:8000/health` should
  return `{"status":"ok",...}` and `curl http://127.0.0.1:8000/config`
  should return `"served_by":"backend"`. If both answer but the badge still
  says DEMO after ~30s, hard-refresh (Ctrl/Cmd+Shift+R) — an old cached
  page is the usual culprit.
- `?demo=1` in the URL forces demo deliberately; remove it for live work.
  Conversely, `?api=<url>` can point a statically hosted page at a separate
  API origin (that origin must be allowed in `PYTHAGORA_CORS_ORIGINS`).

## API

| Method | Route                     | Description                        |
| ------ | ------------------------- | ---------------------------------- |
| GET    | `/`                       | Service banner                     |
| GET    | `/health`                 | Dependency availability report     |
| GET    | `/tracks`                 | List tracks                        |
| POST   | `/tracks`                 | Register a track (201, id assigned) |
| GET    | `/tracks/{id}`            | Get one track                      |
| GET    | `/tracks/{id}/analysis`   | Telemetry bundle for a track       |
| POST   | `/tracks/upload`          | Upload audio (+ optional cover art), start analysis (202 + `job_id`) |
| POST   | `/tracks/{id}/artwork`    | Upload/replace cover art           |
| GET    | `/tracks/{id}/artwork`    | Serve cover art                    |
| GET    | `/tracks/jobs/{job_id}`   | Job status: state / named stages / stage-weighted progress / elapsed |
| GET    | `/tracks/{id}/report`     | Branded printable analysis report (HTML) |
| GET    | `/tracks/compare?a=&b=`   | Side-by-side measured comparison of two tracks (JSON) |
| GET    | `/tracks/compare/report?a=&b=` | Branded printable comparison report |
| PATCH  | `/tracks/{id}`            | Assign/unassign project, edit track notes |
| GET    | `/tracks/{id}/adjusted?semitones=&speed=` | Pitch/speed-adjusted render (cached) |
| GET    | `/tracks/{id}/adjusted/meta` | Derived tempo/key readout for an adjustment |
| GET    | `/projects`               | List projects (with track counts) |
| POST   | `/projects`               | Create a project (201) |
| GET    | `/projects/{id}`          | Get one project |
| PATCH  | `/projects/{id}`          | Rename a project / edit its notes |
| DELETE | `/projects/{id}`          | Delete a project (tracks are unassigned, never deleted) |

**The upload loop:** `POST /tracks/upload` (multipart: `title`, `artist`, `audio`,
optional `artwork`) saves files under `storage/`, returns immediately with a
`job_id`, and runs the analysis pipeline in the background. Poll
`GET /tracks/jobs/{job_id}` for live stage/progress — the dashboard shows a
progress panel and opens the track when the job completes.

The pipeline (`workers/pipeline.py`) runs every deterministic measurement
in order — tempo, key, chords, sections/texture, waveform, vocal forensics
(on the full mix; it refuses to report when the F0 track looks like bass
<80 Hz rather than voice), then mix measurements + rule-fired suggestions —
and writes a `TelemetryBundle` plus a template-built, fully grounded
summary essay. Jobs run on a capped in-process executor: uploads beyond
the concurrency cap wait in a visible "queued" state, which is what makes
batch uploads work. Job status reports named stages with honest,
stage-weighted progress (it moves only when a stage completes). The same
pipeline is exposed as the Celery task `workers.tasks.analyze_track` for
multi-worker deployments.

The local analyzer is ported under `workers/analysis/`: `audio.py`
(tempo/key/sections/texture/mood/style), `measure.py` (band energy,
dBFS levels, stereo), `suggestions.py` (deterministic mix rules).

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

- SQLAlchemy models replacing the in-memory track store (projects, track
  metadata, and notes already persist to a JSON library file)
- Constrained-essay generation stage (telemetry-locked prompting)
- Dashboard: carousels, dual player, waveform timeline, synced lyric viewer
- *Beyond v1 (excluded from the customer version by decision):* stem
  separation and automatic lyric transcription — both require external
  model downloads and are not part of the v1 feature set

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
