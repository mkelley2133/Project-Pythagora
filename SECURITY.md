# Project Pythagoras — Security

Security posture for running Pythagoras as a small commercial audio-upload
analysis service. Controls are based on the OWASP File Upload Cheat Sheet
and the OWASP API Security Top 10, scoped to what this codebase actually is:
a single-process FastAPI service with an in-memory store, a browser
dashboard, and a heavyweight analysis pipeline.

_Last reviewed: 2026-10-12. Dependency audit (pip-audit) clean — see
"Dependency hygiene" below._

## Implemented

### File upload (OWASP File Upload Cheat Sheet)
- [x] **Extension allowlist** for audio (`.mp3 .wav .ogg .flac .m4a .aac`)
      and artwork (`.jpg .jpeg .png .webp`) — first gate only.
- [x] **Content sniffing**: audio is identified by magic bytes (RIFF/WAVE,
      fLaC, OggS, ID3/MPEG sync, ADTS, MP4 `ftyp` brands) in
      `backend/security.py::sniff_audio`. The client filename, extension,
      and Content-Type header are never trusted; the file is stored under
      the *sniffed* format's canonical extension.
- [x] **Artwork content sniffing**: cover art is likewise verified by
      magic bytes (`backend/security.py::sniff_image` — JPEG/PNG/WebP
      signatures) and must match the extension it arrived with, on both
      artwork routes. Magic bytes only: the project carries no imaging
      library, so artwork gets no decode proof (audio does — see next).
- [x] **Decode proof + duration cap**: uploads must actually decode
      (soundfile, falling back to ffprobe — subprocess with list args,
      never `shell=True`) and must not exceed
      `PYTHAGORA_MAX_DURATION_SECONDS` (default 900s).
- [x] **Size caps enforced while streaming** (not after): audio
      `PYTHAGORA_MAX_UPLOAD_BYTES` (default 50 MiB) → HTTP 413; artwork
      10 MiB → 413. A middleware also rejects oversized requests early
      from the Content-Length header.
- [x] **Server-generated storage names**: files are stored as
      `{track_id}{canonical_ext}` under a dedicated directory
      (`PYTHAGORA_STORAGE_DIR`, default `./storage`, split into `audio/`
      and `artwork/`). User filenames never reach the filesystem, which
      kills path traversal and overwrite attacks at the source.
- [x] **Served only via the API, only from inside storage**: the
      audio/artwork endpoints resolve the stored path and refuse anything
      outside the storage directory (`resolve_within_storage`), so a track
      registered through `POST /tracks` with `original_file_path` pointing
      at e.g. `/etc/passwd` returns 404, never file contents. Responses
      carry an explicit safe media type + `X-Content-Type-Options: nosniff`.

### API hygiene (OWASP API Security Top 10)
- [x] **Input validation everywhere**: resource IDs must match
      `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` (malformed → 400); Pydantic
      schemas bound string lengths and numeric ranges; upload form fields
      are length-capped.
- [x] **Rate limiting** (per-IP sliding window, `backend/security.py`):
      uploads — `PYTHAGORA_UPLOAD_RATE_LIMIT` per
      `PYTHAGORA_RATE_WINDOW_SECONDS` (default 30/60s); all `/tracks`
      routes — `PYTHAGORA_API_RATE_LIMIT` (default 1000/60s). Excess →
      429 with `Retry-After`.
- [x] **Concurrency cap**: at most `PYTHAGORA_MAX_CONCURRENT_JOBS`
      (default 4) analyses run at once on the in-process executor; uploads
      beyond the cap wait in a visible "queued" state (batch uploads rely
      on this) so a burst can't exhaust CPU; the finished-job registry is
      pruned so it can't grow without bound.
- [x] **Optional API key**: when `PYTHAGORA_API_KEY` is set, every
      `/tracks/*` and `/projects/*` route requires `X-API-Key`
      (constant-time compare).
      `/health` and the static dashboard stay public so the demo and
      uptime checks keep working. Off when unset (dev/demo).
- [x] **CORS from env**: `PYTHAGORA_CORS_ORIGINS` (comma-separated);
      default is localhost dev origins only. No wildcards. Allowed methods
      and headers are enumerated (`X-API-Key`, `Content-Type`).
- [x] **Security headers on every response**: `X-Content-Type-Options:
      nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`,
      `Content-Security-Policy` (`default-src 'self'`, no inline scripts —
      the dashboard was refactored to comply), `Cross-Origin-Resource-Policy:
      same-origin`, `Permissions-Policy`. HSTS is available via
      `PYTHAGORA_ENABLE_HSTS=1` once the service is behind HTTPS.
- [x] **Errors don't leak**: FastAPI `debug=False`; a catch-all handler
      logs the exception server-side and returns a generic JSON
      `{"detail": "Internal server error."}` — no stack traces to clients.
- [x] **Secrets via env only**: all credentials/limits come from the
      environment (`.env` is git-ignored); `.env.example` documents every
      variable with placeholder values only.

### Dependency hygiene
- [x] `pip-audit -r requirements.txt` (2026-10-12): **no known
      vulnerabilities**.
- [x] `pip-audit` on the dev venv (2026-10-12): the only findings were in
      the venv's bundled `pip` itself (24.0 → upgraded to 26.2.1); re-audit
      clean. `pip` is not an application dependency.
- [x] `requirements.txt` uses bounded ranges (`>=` with `<` upper bounds on
      core packages), so patched releases install without unpinning.

### Tests
`tests/test_security.py` + `tests/test_analysis_upgrades.py` pin the
controls: oversize upload → 413, non-audio/disguised content → 400,
undecodable audio → 400, duration cap → 400, traversal filename
neutralized, out-of-storage path not served, malformed IDs → 400,
upload rate limit → 429 with `Retry-After`, API key enforced when set
(and off by default), security headers present, JSON errors without
stack traces, report endpoint renders.

### Wave 2 surfaces

- [x] **Adjusted-audio renders** (`GET /tracks/{id}/adjusted`): cache
      filenames are built only from the server-validated track id and
      server-formatted clamped floats — no user input reaches a path;
      renders live under `storage/adjusted/` and are rate-limited like
      uploads. The original file is never overwritten.
- [x] **Library persistence**: projects / track metadata / notes persist
      to `<storage_dir>/library.json` (atomic tmp-file replace). It holds
      no secrets and no audio; treat it with the same care as the storage
      dir itself.
- [x] **V1 exclusions**: stem separation and automatic lyric transcription
      are out of the customer v1 — no model weights are downloaded or
      executed anywhere in the request path, and no endpoint serves stems.

## Deferred (deliberate, with rationale)

- [ ] **Malware scanning (ClamAV/CDR)** — OWASP recommends it for public
      upload services. Not wired: uploaded audio is only ever *decoded* by
      soundfile/ffmpeg/librosa and *served* back as audio; nothing is
      executed. Add ClamAV before accepting uploads at any real scale.
- [ ] **Full user accounts / per-customer authorization** — the API-key
      middleware is the seam: today one shared key gates the whole API.
      A SaaS launch needs accounts, per-user track ownership (BOLA/API1),
      and key rotation per customer.
- [ ] **Distributed rate limiting** — limiters are in-process, correct for
      the single-node deployment. Multi-worker hosting needs the counters
      in Redis (the Celery broker is already Redis when configured).
- [ ] **Signed media URLs** — with an API key set, the dashboard's
      `<audio>`/`<img>` tags can't send the key header, so real-upload
      playback/artwork degrade (API reads via fetch still work). The SaaS
      fix is short-lived signed URLs for media, issued after auth.
- [ ] **CSP tightening** — `style-src` keeps `'unsafe-inline'` because the
      dashboard uses inline `style` attributes for cover hues/meter
      widths. Script sources are already `'self'`-only. Removing the
      style exception means moving those to CSS custom properties set via
      `element.style` (already partially done) — cosmetic, low risk.
- [ ] **Upload CSRF tokens** — the API is token/header-authenticated (no
      cookies), so classic CSRF doesn't apply to the API itself; revisit
      if cookie sessions are ever added.
- [ ] **Audit logging / abuse alerts** — security events (401/413/429)
      are not yet shipped to a log aggregator; uvicorn logs are the only
      record. Wire structured logging before taking paying customers.

## Operator quick reference

| Goal | Setting |
| --- | --- |
| Require an API key | `PYTHAGORA_API_KEY=<long random value>` |
| Allow a production origin | `PYTHAGORA_CORS_ORIGINS=https://your-domain` |
| Smaller upload cap | `PYTHAGORA_MAX_UPLOAD_BYTES=20971520` |
| Shorter max track length | `PYTHAGORA_MAX_DURATION_SECONDS=600` |
| Behind HTTPS | `PYTHAGORA_ENABLE_HSTS=1` |
| More/fewer parallel analyses | `PYTHAGORA_MAX_CONCURRENT_JOBS=8` |
