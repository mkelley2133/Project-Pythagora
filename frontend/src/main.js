import { SynthEngine } from "./synth.js";
import {
  clampAdjust,
  coupledPlaybackRate,
  effectiveBpm,
  transposeKey,
} from "./adjust.js";
import {
  EQ_BANDS,
  EQ_PRESETS,
  EqChain,
  combinedResponseDb,
  loadEqSettings,
  presetBandState,
  saveEqSettings,
} from "./eq.js";
import { sortTracks } from "./sort.js";

/* ------------------------------------------------------------------ */
/* Mode detection — LIVE vs DEMO                                       */
/*                                                                     */
/* The backend injects <meta name="pythagoras-served-by"> into the     */
/* dashboard it serves (see app/main.py). Marker present  => this page */
/* came from the real API, so demo fallback is FORBIDDEN: the boot     */
/* sequence retries /health with backoff until it answers (Codespaces */
/* cold starts are slow) and the badge shows LIVE once connected.      */
/* Demo engages ONLY when the page was not backend-served AND one of:  */
/*   - hosted on *.github.io (the GitHub Pages demo) with no API base  */
/*     deliberately configured — a saved/?api= backend address makes  */
/*     even the hosted page go live so uploads work there,            */
/*   - explicit ?demo=1 in the URL,                                    */
/*   - no API answered after the full ~20s retry window.               */
/* API base resolution order: ?api= (persisted) > localStorage        */
/* pythagora_api_base > window.PYTHAGORAS_API_BASE > page origin.      */
/* ------------------------------------------------------------------ */
const _qs = new URLSearchParams(window.location.search);
const SERVED_BY_BACKEND = Boolean(
  document.querySelector('meta[name="pythagoras-served-by"]')
);
const DEMO_BY_QUERY = _qs.get("demo") === "1";
const DEMO_BY_HOST = /(^|\.)github\.io$/i.test(window.location.hostname);
const EXPLICIT_DEMO = DEMO_BY_QUERY || DEMO_BY_HOST;
/* An API base the user deliberately configured (?api= or the saved
   pythagora_api_base from the upload modal's Connect form) overrides
   the hosted-demo default: the page must still try to go LIVE against
   that backend, otherwise the public demo could never accept uploads. */
const HAS_CONFIGURED_API =
  Boolean(_qs.get("api")) ||
  (() => {
    try {
      return Boolean(window.localStorage.getItem("pythagora_api_base"));
    } catch {
      return false;
    }
  })();

function resolveApiBase() {
  const fromQuery = _qs.get("api");
  if (fromQuery) {
    try {
      localStorage.setItem("pythagora_api_base", fromQuery);
    } catch { /* storage unavailable */ }
    return fromQuery;
  }
  try {
    const stored = localStorage.getItem("pythagora_api_base");
    if (stored) return stored;
  } catch { /* storage unavailable */ }
  return window.PYTHAGORAS_API_BASE || window.location.origin;
}

const API_BASE = String(resolveApiBase()).replace(/\/+$/, "");
/* "connecting" | "live" | "demo" — the badge in the topbar reflects this. */
let appMode = "connecting";

/* ------------------------------------------------------------------ */
/* Demo fallback data (mirrors the backend demo bundle) so the UI      */
/* works even when the API isn't running.                             */
/* ------------------------------------------------------------------ */
const DEMO_TRACK = {
  id: "demo-track-001",
  title: "Sample Signal",
  artist: "Pythagoras",
  original_file_path: "/storage/demo.mp3",
  project_id: "demo-project-001",
  notes: "Placeholder telemetry until a real upload is analyzed.",
  created_at: "2026-01-01T00:00:00Z",
};

const DEMO_PROJECTS = [
  {
    id: "demo-project-001",
    name: "Sample Sessions",
    notes: "Demo project — groups the sample track so projects, sorting, and notes have something to show before your first upload.",
    created_at: "2026-01-01T00:00:00Z",
    track_count: 1,
    track_ids: ["demo-track-001"],
  },
];

const DEMO_TELEMETRY = {
  bpm: 92.4,
  duration: 83.12,
  musical_key: "D",
  mode: "minor",
  primary_genre: "Pop music",
  genre_predictions: [
    { label: "Pop music", probability: 0.412 },
    { label: "Independent music", probability: 0.183 },
    { label: "Ambient music", probability: 0.107 },
    { label: "Electronic music", probability: 0.088 },
    { label: "Rock music", probability: 0.051 },
  ],
  genre_model: "PANNs Cnn10 (AudioSet) — demo data",
  genre_note: "Demo bundle: sample genre output for layout preview.",
  primary_subgenre: "bedroom pop",
  subgenre_predictions: [
    { label: "bedroom pop", family: "Pop", percent: 24.6 },
    { label: "indie pop", family: "Pop", percent: 16.8 },
    { label: "shoegaze", family: "Rock / metal", percent: 11.9 },
    { label: "ambient electronic", family: "Electronic", percent: 8.7 },
    { label: "alternative R&B", family: "R&B", percent: 6.4 },
  ],
  subgenre_model: "LAION CLAP (clap-htsat-unfused) — demo data",
  subgenre_note: "Demo bundle: sample subgenre matches. Subgenre matches are zero-shot similarity shares from CLAP — they rank how close the audio sounds to each subgenre description and are NOT trained classifier probabilities.",
  texture_cues: [{ name: "dream pop", note: "Demo texture cue — placeholder until a real upload is analyzed." }],
  emotional_profile: { hope: 0.7, melancholy: 0.8 },
  chords: ["Dm", "Bb", "F", "C"],
  segments: [
    { name: "Intro", start: 0.0, end: 10.39 },
    { name: "Verse", start: 10.39, end: 31.18 },
    { name: "Chorus", start: 31.18, end: 51.95 },
    { name: "Verse", start: 51.95, end: 62.34 },
    { name: "Chorus", start: 62.34, end: 83.12 },
  ],
  rhyme_scheme_summary: "ABCB",
  mix_measurements: {
    sub_pct: 4.2, low_pct: 21.5, mid_pct: 46.8, high_pct: 21.9, air_pct: 5.6,
    peak_dbfs: -1.2, rms_dbfs: -16.4, crest_db: 15.2,
    is_stereo: true, stereo_correlation: 0.62, stereo_width: 0.58,
  },
  suggestions: [
    {
      area: "Harmony",
      suggestion: "The progression never leaves its 4-chord loop — consider a bridge or pre-chorus variation to reset the ear.",
      why: "Only 4 unique chords (Dm, Bb, F, C) across the full 83s demo track.",
      severity: "low",
    },
  ],
  timestamped_lyrics: [
    { start: 10.5, end: 15.5, text: "Neon hums on an empty street", rhyme: "A" },
    { start: 15.7, end: 20.7, text: "I chase the echo of your heartbeat", rhyme: "B" },
    { start: 20.9, end: 25.9, text: "Midnight folds the sky in two", rhyme: "C" },
    { start: 26.1, end: 31.0, text: "Still every road leads back to you", rhyme: "B" },
    { start: 31.3, end: 36.3, text: "Hold the light, don't let it fade", rhyme: "A" },
    { start: 36.5, end: 41.5, text: "We are embers in the snow", rhyme: "B" },
    { start: 41.7, end: 46.7, text: "If the morning takes your hand", rhyme: "C" },
    { start: 46.9, end: 51.8, text: "I'll be waiting in the snow", rhyme: "B" },
    { start: 52.1, end: 57.1, text: "Static blooms on the radio", rhyme: "A" },
    { start: 57.3, end: 62.2, text: "Take me anywhere but slow", rhyme: "B" },
    { start: 62.5, end: 67.5, text: "Hold the light, don't let it fade", rhyme: "A" },
    { start: 67.7, end: 72.7, text: "We are embers in the snow", rhyme: "B" },
    { start: 72.9, end: 77.9, text: "If the morning takes your hand", rhyme: "C" },
    { start: 78.1, end: 83.0, text: "I'll be waiting in the snow", rhyme: "B" },
  ],
  musicological_essay:
    "A concise placeholder analysis for the starter pipeline. The full essay stage " +
    "synthesizes locked telemetry — harmonic movement, rhyme architecture, and " +
    "emotional contour — into a grounded narrative of the track's journey from intro to outro.",
};

/* ------------------------------------------------------------------ */
/* Utils                                                              */
/* ------------------------------------------------------------------ */
function fmtTime(s) {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}

function hashStr(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function coverHues(id) {
  // Night-sky reds/crimsons/embers — never rainbow.
  const h1 = 346 + (hashStr(id) % 26);
  return { h1, h2: (h1 + 16) % 360 };
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 2600);
}

/* ------------------------------------------------------------------ */
/* API                                                                */
/* ------------------------------------------------------------------ */
let apiLive = false;

/* When the API runs with PYTHAGORA_API_KEY set, store the key in this
   browser via:  localStorage.setItem("pythagora_api_key", "<key>")       */
function apiHeaders(extra = {}) {
  let key = null;
  try {
    key = localStorage.getItem("pythagora_api_key");
  } catch { /* storage unavailable */ }
  return key ? { ...extra, "X-API-Key": key } : extra;
}

/* Build an Error from a failed response, surfacing the server's own
   JSON `detail` (e.g. "Audio could not be decoded.") instead of a bare
   "HTTP 400" — callers toast err.message, so this one helper improves
   every surface. The status rides along for callers that branch on it
   (loadAnalysis treats 404 as "not analyzed yet", not an error). */
async function apiError(res) {
  let detail = `HTTP ${res.status}`;
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string" && body.detail) detail = body.detail;
  } catch { /* non-JSON body: keep the status line */ }
  const err = new Error(detail);
  err.status = res.status;
  return err;
}

async function apiGet(path) {
  const res = await fetch(API_BASE + path, { headers: apiHeaders() });
  if (!res.ok) throw await apiError(res);
  return res.json();
}

async function apiSend(method, path, body) {
  const res = await fetch(API_BASE + path, {
    method,
    headers: apiHeaders({ "Content-Type": "application/json" }),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw await apiError(res);
  if (res.status === 204) return null;
  return res.json();
}

/* ------------------------------------------------------------------ */
/* Local library (no backend connected)                                */
/*                                                                     */
/* The app still works as an organizer with no backend: projects,      */
/* track groupings, and notes persist in localStorage, and locally     */
/* added tracks keep their audio in IndexedDB. Analysis (BPM, key,     */
/* chords) is the only thing that needs the backend.                   */
/* ------------------------------------------------------------------ */
const LOCAL_LIB_KEY = "pythagora_local_library_v1";
const _localObjectUrls = new Map();

function readLocalLibrary() {
  try {
    const raw = window.localStorage.getItem(LOCAL_LIB_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.projects) && Array.isArray(parsed.localTracks)) return parsed;
    }
  } catch { /* storage unavailable or corrupt — start fresh */ }
  return {
    projects: DEMO_PROJECTS.map((p) => ({ ...p })),
    localTracks: [],
    demoTrack: { project_id: DEMO_TRACK.project_id ?? null, notes: DEMO_TRACK.notes || "" },
  };
}

let localLib = readLocalLibrary();

/* Local mode keeps one source of truth while running: state. Persist
   it after every mutation so grouping survives a reload. */
function persistLocalState() {
  if (apiLive) return;
  try {
    localLib.projects = state.projects.map((p) => ({ ...p }));
    localLib.localTracks = state.tracks.filter((t) => t.local).map((t) => ({ ...t }));
    const demo = state.tracks.find((t) => t.id === DEMO_TRACK.id);
    if (demo) {
      localLib.demoTrack = { project_id: demo.project_id ?? null, notes: demo.notes || "" };
    }
    window.localStorage.setItem(LOCAL_LIB_KEY, JSON.stringify(localLib));
  } catch { /* storage unavailable — grouping stays session-only */ }
}

function idbOpen() {
  return new Promise((resolve, reject) => {
    const rq = window.indexedDB.open("pythagora-local", 1);
    rq.onupgradeneeded = () => rq.result.createObjectStore("audio");
    rq.onsuccess = () => resolve(rq.result);
    rq.onerror = () => reject(rq.error);
  });
}

async function idbPutAudio(id, blob) {
  if (!window.indexedDB) return false;
  try {
    const db = await idbOpen();
    return await new Promise((resolve) => {
      const tx = db.transaction("audio", "readwrite");
      tx.objectStore("audio").put(blob, id);
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => resolve(false);
    });
  } catch {
    return false;
  }
}

async function idbGetAudio(id) {
  if (!window.indexedDB) return null;
  try {
    const db = await idbOpen();
    return await new Promise((resolve) => {
      const rq = db.transaction("audio").objectStore("audio").get(id);
      rq.onsuccess = () => resolve(rq.result || null);
      rq.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

async function localAudioUrl(track) {
  if (_localObjectUrls.has(track.id)) return _localObjectUrls.get(track.id);
  const blob = await idbGetAudio(track.id);
  if (!blob) return null;
  let url;
  try {
    url = URL.createObjectURL(blob);
  } catch {
    return null; // no object-URL support — grouping still works, playback doesn't
  }
  _localObjectUrls.set(track.id, url);
  return url;
}

async function loadProjects() {
  if (apiLive) {
    try {
      return await apiGet("/projects");
    } catch (err) {
      /* Live mode never substitutes demo data silently. */
      toast("Couldn't load projects from the API: " + err.message);
      return [];
    }
  }
  return localLib.projects.map((p) => ({ ...p }));
}

/* CSP forbids inline onerror handlers, so broken artwork hides itself
   through this listener instead — call after injecting markup. */
function wireImgFallback(scope) {
  scope.querySelectorAll("img.cover-img, img.billboard-art").forEach((img) => {
    img.addEventListener("error", () => img.remove());
  });
}

async function openReport(trackId) {
  return openApiPage(`${API_BASE}/tracks/${trackId}/report`);
}

async function openApiPage(url) {
  let key = null;
  try {
    key = localStorage.getItem("pythagora_api_key");
  } catch { /* storage unavailable */ }
  if (!key) {
    window.open(url, "_blank", "noopener");
    return;
  }
  try {
    const res = await fetch(url, { headers: apiHeaders() });
    if (!res.ok) throw await apiError(res);
    const blob = await res.blob();
    window.open(URL.createObjectURL(blob), "_blank", "noopener");
  } catch (err) {
    toast("Report failed: " + err.message);
  }
}

function setModeBadge(mode, label) {
  const el = document.getElementById("api-status");
  if (el) {
    el.classList.toggle("ok", mode === "live");
    el.classList.toggle("demo", mode === "demo");
    el.classList.toggle("connecting", mode === "connecting");
    const labelEl = el.querySelector(".label");
    if (labelEl) labelEl.textContent = label;
  }
  const footer = document.getElementById("footer-mode");
  if (footer) {
    footer.textContent =
      mode === "live"
        ? "● LIVE — connected to the Pythagoras API"
        : mode === "demo"
          ? "● DEMO — sample telemetry, no backend connected"
          : "● CONNECTING — waiting for the Pythagoras API…";
  }
}

function enterDemoMode() {
  apiLive = false;
  appMode = "demo";
  setModeBadge("demo", "DEMO");
}

const _sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function checkApi() {
  /* Explicit demo is a deliberate choice, not a fallback — honor it
     immediately. One exception: on the hosted demo (GitHub Pages), a
     deliberately configured API base (?api= or the saved address from
     the upload modal's Connect form) means the user wants THIS page
     live against their backend, so fall through to the health check. */
  if (DEMO_BY_QUERY || (DEMO_BY_HOST && !HAS_CONFIGURED_API)) {
    enterDemoMode();
    return;
  }
  /* Retry with backoff. Backend-served pages retry indefinitely: the API
     is on this same origin, so a failure can only be a cold start that
     hasn't finished yet — demo data would be a lie. Static-hosted pages
     get a ~20s window (Codespaces-class cold starts) before demo. */
  const delays = [600, 1200, 2000, 3000, 4000, 5000, 5000];
  const started = Date.now();
  let attempt = 0;
  for (;;) {
    attempt += 1;
    setModeBadge("connecting", `connecting… (${attempt})`);
    try {
      await apiGet("/health");
      apiLive = true;
      appMode = "live";
      setModeBadge("live", "LIVE");
      return;
    } catch {
      const elapsed = Date.now() - started;
      if (!SERVED_BY_BACKEND && elapsed > 20000) {
        enterDemoMode();
        return;
      }
      await _sleep(delays[Math.min(attempt - 1, delays.length - 1)]);
    }
  }
}

async function loadTracks() {
  if (apiLive) {
    try {
      return await apiGet("/tracks");
    } catch (err) {
      toast("Couldn't load tracks from the API: " + err.message);
      return [];
    }
  }
  return [
    {
      ...DEMO_TRACK,
      project_id: localLib.demoTrack.project_id,
      notes: localLib.demoTrack.notes,
    },
    ...localLib.localTracks.map((t) => ({ ...t })),
  ];
}

async function loadAnalysis(trackId, { reportErrors = false } = {}) {
  if (apiLive) {
    try {
      const rec = await apiGet(`/tracks/${trackId}/analysis`);
      return rec.telemetry_json || null;
    } catch (err) {
      // 404 means "not analyzed yet" — a normal state, kept silent
      // (the library loader calls this for every track). Anything else
      // is a real failure worth surfacing when the caller asks.
      if (reportErrors && err.status !== 404) {
        toast("Couldn't load this track's analysis: " + err.message);
      }
      return null;
    }
  }
  return trackId === DEMO_TRACK.id ? DEMO_TELEMETRY : null;
}

/* ------------------------------------------------------------------ */
/* State                                                              */
/* ------------------------------------------------------------------ */
const state = {
  tracks: [],
  telemetry: {}, // trackId -> TelemetryBundle
  projects: [],
  projectFilter: "all", // "all" | "unsorted" | projectId
  sortKey: "date",
  sortDir: "desc",
  detail: null, // { track, telemetry, engine, peaks, raf, activeLyric }
  lastTrackId: null,
  comparePair: null, // { a, b } track ids while the compare modal is open
};

/* Player FX (Phases H + I): one EQ chain per live audio context, settings
   persisted per browser via localStorage. */
const fx = {
  settings: loadEqSettings(),
  chain: null,
  ctx: null,
  adjust: { semitones: 0, speed: 1 },
};

function attachFx(ctx, sourceNode) {
  try {
    const chain = new EqChain(ctx, fx.settings);
    try { sourceNode.disconnect(); } catch { /* not wired yet */ }
    sourceNode.connect(chain.input);
    chain.output.connect(ctx.destination);
    fx.chain = chain;
    fx.ctx = ctx;
    drawEqCurve();
  } catch { /* FX is enhancement-only; raw playback continues */ }
}

/* ------------------------------------------------------------------ */
/* Library view                                                       */
/* ------------------------------------------------------------------ */
function trackCard(track) {
  const { h1, h2 } = coverHues(track.id);
  const tel = state.telemetry[track.id];
  const art = artworkUrl(track);
  const project = track.project_id
    ? state.projects.find((p) => p.id === track.project_id)
    : null;
  const card = document.createElement("div");
  card.className = "track-card";
  card.innerHTML = `
    <div class="cover" style="--h1:${h1};--h2:${h2}">
      ${art ? `<img class="cover-img" src="${art}" alt="" loading="lazy">` : ""}
      <span class="cover-glyph">♪</span>
      <div class="play-hover"><button class="pp" aria-label="Open">▶</button></div>
    </div>
    <div class="track-meta">
      <h3>${esc(track.title)}</h3>
      <p class="artist">${esc(track.artist)}</p>
      <div class="chip-row">
        ${
          tel
            ? `<span class="chip hot">${tel.bpm} BPM</span><span class="chip">${esc(tel.musical_key)} ${esc(tel.mode)}</span>`
            : `<span class="chip">analysis pending</span>`
        }
        ${project ? `<span class="chip proj-chip">▸ ${esc(project.name)}</span>` : ""}
      </div>
    </div>`;
  card.addEventListener("click", () => openDetail(track.id));
  wireImgFallback(card);
  return card;
}

function moodOf(telemetry) {
  const entries = Object.entries(telemetry.emotional_profile || {});
  if (!entries.length) return "unclassified";
  return entries.sort((a, b) => b[1] - a[1])[0][0];
}

function artworkUrl(track) {
  // Local tracks may carry a small data-URL cover; live tracks are
  // served by the API; otherwise the gradient cover shows.
  if (track.artwork_data) return track.artwork_data;
  return apiLive ? `${API_BASE}/tracks/${track.id}/artwork` : null;
}

function energyOf(telemetry) {
  if (!telemetry) return null;
  const p = telemetry.emotional_profile || {};
  const nums = [p.arousal, p.energy].filter((v) => typeof v === "number");
  if (nums.length) return Math.max(...nums);
  const m = moodOf(telemetry).toLowerCase();
  if (/upbeat|intense|euphoric|aggressive/.test(m)) return 0.8;
  if (/mellow|dark|brooding|reflective/.test(m)) return 0.25;
  return 0.5;
}

function featuredTrack() {
  const withTel = state.tracks.filter((t) => state.telemetry[t.id]);
  if (withTel.length) {
    return withTel.sort(
      (a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0)
    )[0];
  }
  return state.tracks[0] || null;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function makeRow(title, sub, list) {
  const block = document.createElement("div");
  block.className = "carousel-block";
  block.innerHTML = `<div class="carousel-head"><h2>${esc(title)}</h2><span>${esc(sub)}</span></div>`;
  const car = document.createElement("div");
  car.className = "carousel";
  if (!list.length) {
    car.innerHTML = `<div class="empty-note">Nothing here yet.</div>`;
  } else {
    for (const t of list) car.appendChild(trackCard(t));
  }
  block.appendChild(car);
  return block;
}

function renderBillboard() {
  const host = document.getElementById("billboard");
  const feat = featuredTrack();
  if (!feat) {
    host.innerHTML = `
      <div class="billboard billboard-empty">
        <div class="billboard-copy">
          <p class="eyebrow">Zero-hallucination music analysis</p>
          <h1>Every note, <em>measured</em>.</h1>
          <p class="lede">Upload a track to lock down its telemetry — key, tempo, structure, rhyme — before any story is told.</p>
        </div>
      </div>`;
    return;
  }
  const tel = state.telemetry[feat.id];
  const { h1, h2 } = coverHues(feat.id);
  const art = artworkUrl(feat);
  const mood = tel ? moodOf(tel) : "—";
  host.innerHTML = `
    <div class="billboard" style="--h1:${h1};--h2:${h2}">
      ${art ? `<img class="billboard-art" src="${art}" alt="">` : ""}
      <div class="billboard-shade"></div>
      <div class="billboard-copy">
        <p class="eyebrow">Featured analysis · ${esc(mood)}</p>
        <h1>${esc(feat.title)}</h1>
        <p class="artist">${esc(feat.artist)}</p>
        <div class="chip-row">
          ${
            tel
              ? `<span class="chip hot">${tel.bpm} BPM</span>
                 <span class="chip">${esc(tel.musical_key)} ${esc(tel.mode)}</span>
                 <span class="chip">${esc(tel.rhyme_scheme_summary || "—")} rhyme</span>
                 <span class="chip">${tel.segments ? tel.segments.length : 0} sections</span>`
              : `<span class="chip">analysis pending</span>`
          }
        </div>
        <div class="billboard-actions">
          <button class="cta-btn" id="bb-play">▶ Play</button>
          <button class="ghost-btn" id="bb-info">More info</button>
        </div>
      </div>
    </div>`;
  wireImgFallback(host);
  document.getElementById("bb-play").addEventListener("click", () => openDetail(feat.id));
  document.getElementById("bb-info").addEventListener("click", () => openDetail(feat.id));
}

function renderLibraryControls() {
  const bar = document.createElement("div");
  bar.className = "lib-controls";
  const tab = (value, label) =>
    `<button class="proj-tab${state.projectFilter === value ? " active" : ""}" data-filter="${esc(value)}">${esc(label)}</button>`;
  bar.innerHTML = `
    <div class="proj-tabs">
      ${tab("all", "All tracks")}
      ${tab("unsorted", "Unsorted")}
      ${state.projects.map((p) => tab(p.id, p.name)).join("")}
      <button class="proj-tab proj-new" id="proj-new-btn">+ Project</button>
      <span class="proj-new-form hidden" id="proj-new-form">
        <input id="proj-new-name" type="text" placeholder="Project name" maxlength="200" />
        <button class="cta-btn" id="proj-new-create">Create</button>
      </span>
    </div>
    <div class="sort-controls">
      <label>Sort
        <select id="sort-key">
          <option value="date"${state.sortKey === "date" ? " selected" : ""}>Date added</option>
          <option value="name"${state.sortKey === "name" ? " selected" : ""}>Name</option>
          <option value="tempo"${state.sortKey === "tempo" ? " selected" : ""}>Tempo</option>
          <option value="key"${state.sortKey === "key" ? " selected" : ""}>Key</option>
          <option value="duration"${state.sortKey === "duration" ? " selected" : ""}>Duration</option>
        </select>
      </label>
      <button class="ghost-btn sort-dir" id="sort-dir" title="Toggle sort direction">${state.sortDir === "asc" ? "↑ Asc" : "↓ Desc"}</button>
    </div>`;
  bar.querySelectorAll(".proj-tab[data-filter]").forEach((b) =>
    b.addEventListener("click", () => {
      state.projectFilter = b.dataset.filter;
      renderLibrary();
    })
  );
  bar.querySelector("#sort-key").addEventListener("change", (e) => {
    state.sortKey = e.target.value;
    renderLibrary();
  });
  bar.querySelector("#sort-dir").addEventListener("click", () => {
    state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
    renderLibrary();
  });
  bar.querySelector("#proj-new-btn").addEventListener("click", () => {
    bar.querySelector("#proj-new-form").classList.toggle("hidden");
    bar.querySelector("#proj-new-name").focus();
  });
  bar.querySelector("#proj-new-create").addEventListener("click", async () => {
    const name = bar.querySelector("#proj-new-name").value.trim();
    if (!name) return;
    await createProject(name);
  });
  return bar;
}

async function createProject(name) {
  let created = null;
  if (apiLive) {
    try {
      created = await apiSend("POST", "/projects", { name, notes: "" });
      state.projects = await loadProjects();
      state.projectFilter = created.id;
    } catch (err) {
      toast("Project failed: " + err.message);
      return null;
    }
  } else {
    created = {
      id: `demo-project-${Date.now()}`,
      name,
      notes: "",
      created_at: new Date().toISOString(),
      track_count: 0,
      track_ids: [],
    };
    state.projects.push(created);
    state.projectFilter = created.id;
    toast("Project created — saved in this browser.");
  }
  persistLocalState();
  renderLibrary();
  return created;
}

function renderProjectHeader(project) {
  const host = document.createElement("div");
  host.className = "project-head panel";
  const count = state.tracks.filter((t) => t.project_id === project.id).length;
  host.innerHTML = `
    <div class="project-head-top">
      <input id="proj-name" class="project-name" type="text" value="${esc(project.name)}" maxlength="200" />
      <span class="chip">${count} track${count === 1 ? "" : "s"}</span>
      <button class="ghost-btn" id="proj-delete">Delete project</button>
    </div>
    <label class="project-notes-label">Project notes
      <textarea id="proj-notes" rows="3" placeholder="What is this project? What is left to do?">${esc(project.notes || "")}</textarea>
    </label>
    <div class="project-head-actions">
      <button class="cta-btn" id="proj-save">Save project</button>
      <span class="player-note">Deleting a project keeps its tracks — they go back to Unsorted.</span>
    </div>`;
  host.querySelector("#proj-save").addEventListener("click", async () => {
    const name = host.querySelector("#proj-name").value.trim() || project.name;
    const notes = host.querySelector("#proj-notes").value;
    if (apiLive) {
      try {
        await apiSend("PATCH", `/projects/${project.id}`, { name, notes });
        state.projects = await loadProjects();
        toast("Project saved.");
      } catch (err) {
        toast("Save failed: " + err.message);
        return;
      }
    } else {
      project.name = name;
      project.notes = notes;
      persistLocalState();
      toast("Project saved in this browser.");
    }
    renderLibrary();
  });
  host.querySelector("#proj-delete").addEventListener("click", async () => {
    if (apiLive) {
      try {
        await apiSend("DELETE", `/projects/${project.id}`);
        state.projects = await loadProjects();
      } catch (err) {
        toast("Delete failed: " + err.message);
        return;
      }
    } else {
      state.projects = state.projects.filter((p) => p.id !== project.id);
      for (const t of state.tracks) if (t.project_id === project.id) t.project_id = null;
      persistLocalState();
    }
    state.projectFilter = "all";
    toast("Project deleted — its tracks are now Unsorted.");
    renderLibrary();
  });
  return host;
}

async function renderLibrary() {
  const carousels = document.getElementById("carousels");
  carousels.innerHTML = "";

  // Load telemetry for every track (parallel) so rows can classify by mood/energy.
  state.telemetry = {};
  await Promise.all(
    state.tracks.map(async (t) => {
      const tel = await loadAnalysis(t.id);
      if (tel) state.telemetry[t.id] = tel;
    })
  );

  renderBillboard();
  carousels.appendChild(renderLibraryControls());

  const sorted = (list) => sortTracks(list, state.telemetry, state.sortKey, state.sortDir);

  if (state.projectFilter === "unsorted") {
    const unsorted = state.tracks.filter((t) => !t.project_id);
    carousels.appendChild(makeRow("Unsorted", "tracks with no project", sorted(unsorted)));
    return;
  }
  if (state.projectFilter !== "all") {
    const project = state.projects.find((p) => p.id === state.projectFilter);
    if (project) {
      carousels.appendChild(renderProjectHeader(project));
      const mine = state.tracks.filter((t) => t.project_id === project.id);
      carousels.appendChild(makeRow(project.name, "project tracks", sorted(mine)));
      return;
    }
    state.projectFilter = "all"; // stale filter — fall through to all
  }

  const energy = (t) => energyOf(state.telemetry[t.id]);
  const recent = [...state.tracks].sort(
    (a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0)
  );

  carousels.appendChild(
    makeRow("Your library", `${state.tracks.length} track${state.tracks.length === 1 ? "" : "s"}`, sorted(state.tracks))
  );
  const high = state.tracks.filter((t) => (energy(t) ?? 0) >= 0.6);
  if (high.length) carousels.appendChild(makeRow("High energy", "arousal ≥ 60%", high));
  const chill = state.tracks.filter((t) => {
    const e = energy(t);
    return e !== null && e < 0.4;
  });
  if (chill.length) carousels.appendChild(makeRow("Chill & mellow", "low arousal", chill));
  if (recent.length > 1) carousels.appendChild(makeRow("Recently added", "latest first", recent.slice(0, 10)));

  // Browse by mood — groups tracks under their dominant measured emotion.
  const byMood = {};
  for (const t of state.tracks) {
    const m = state.telemetry[t.id] ? moodOf(state.telemetry[t.id]) : "unclassified";
    (byMood[m] = byMood[m] || []).push(t);
  }
  const moodNames = Object.keys(byMood).sort();
  if (moodNames.length) {
    const block = document.createElement("div");
    block.className = "carousel-block";
    block.innerHTML = `<div class="carousel-head"><h2>Browse by mood</h2><span>dominant measured emotion per track</span></div>`;
    const car = document.createElement("div");
    car.className = "carousel";
    for (const m of moodNames) {
      for (const t of byMood[m]) {
        const wrap = document.createElement("div");
        wrap.style.minWidth = "220px";
        if (t === byMood[m][0]) {
          wrap.innerHTML = `<div class="chip" style="margin-bottom:8px;display:inline-block">${esc(m)}</div>`;
        }
        wrap.appendChild(trackCard(t));
        car.appendChild(wrap);
      }
    }
    block.appendChild(car);
    carousels.appendChild(block);
  }
}

/* ------------------------------------------------------------------ */
/* Waveform (deterministic placeholder peaks until the analysis       */
/* workers ship real peak data)                                       */
/* ------------------------------------------------------------------ */
function makePeaks(trackId, n = 520) {
  const rand = mulberry32(hashStr(trackId));
  const peaks = [];
  let v = 0.5;
  for (let i = 0; i < n; i++) {
    v = v * 0.82 + rand() * 0.35;
    // swells near "chorus" regions for a musical feel
    const swell = 0.55 + 0.45 * Math.abs(Math.sin((i / n) * Math.PI * 5));
    peaks.push(Math.min(1, v * swell + rand() * 0.12));
  }
  return peaks;
}

const SEG_COLORS = ["#e50914", "#f5b544", "#d97b2f", "#ff6b5e", "#8f1d22"];

function drawWave(canvas, peaks, segments, duration, playheadT) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== Math.round(w * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // Segment bands.
  segments.forEach((seg, i) => {
    const x0 = (seg.start / duration) * w;
    const x1 = (seg.end / duration) * w;
    ctx.fillStyle = SEG_COLORS[i % SEG_COLORS.length] + "22";
    ctx.fillRect(x0, 0, x1 - x0, h);
  });

  // Peaks.
  const n = peaks.length;
  const bw = w / n;
  const playedX = (playheadT / duration) * w;
  for (let i = 0; i < n; i++) {
    const x = i * bw;
    const ph = peaks[i] * (h * 0.86);
    const y = (h - ph) / 2;
    ctx.fillStyle = x <= playedX ? "rgba(229,9,20,0.92)" : "rgba(154,148,168,0.5)";
    ctx.fillRect(x, y, Math.max(1, bw - 0.6), ph);
  }

  // Playhead.
  ctx.fillStyle = "#fff";
  ctx.fillRect(playedX - 1, 0, 2, h);
  ctx.fillStyle = "#e5484d";
  ctx.beginPath();
  ctx.moveTo(playedX - 6, 0);
  ctx.lineTo(playedX + 6, 0);
  ctx.lineTo(playedX, 10);
  ctx.closePath();
  ctx.fill();
}

/* ------------------------------------------------------------------ */
/* Detail view                                                        */
/* ------------------------------------------------------------------ */
function showView(name) {
  document.getElementById("view-library").classList.toggle("hidden", name !== "library");
  document.getElementById("view-detail").classList.toggle("hidden", name !== "detail");
  document.querySelectorAll(".nav-link").forEach((b) =>
    b.classList.toggle("active", b.dataset.nav === name || (b.dataset.nav === "detail-demo" && name === "detail"))
  );
  if (name === "library") stopDetail();
  window.scrollTo({ top: 0 });
}

function stopDetail() {
  const d = state.detail;
  if (!d) return;
  if (d.engine) d.engine.pause();
  if (d.raf) cancelAnimationFrame(d.raf);
  state.detail = null;
  fx.chain = null;
  fx.ctx = null;
}

async function openDetail(trackId) {
  stopDetail();
  state.lastTrackId = trackId;
  const track =
    state.tracks.find((t) => t.id === trackId) ||
    (appMode === "demo" ? DEMO_TRACK : null);
  if (!track) {
    toast("Track not found in the live library.");
    return;
  }
  const telemetry = await loadAnalysis(trackId, { reportErrors: true });
  // Prefer the real uploaded recording when the API can serve it;
  // otherwise fall back to the synthesized stand-in. Unanalyzed tracks
  // still get their recording (the pending view plays it directly).
  let audioUrl = null;
  if (apiLive) {
    try {
      const r = await fetch(`${API_BASE}/tracks/${trackId}/audio`, { method: "HEAD", headers: apiHeaders() });
      if (r.ok) audioUrl = `${API_BASE}/tracks/${trackId}/audio`;
    } catch {
      /* fall through to synth */
    }
  } else if (track.local) {
    audioUrl = await localAudioUrl(track);
  }
  showView("detail");
  renderDetail(track, telemetry, audioUrl);
}

/* Plays the actual uploaded recording, synced to measured telemetry.
   Same interface as SynthEngine so the detail view works unchanged. */
class RealAudioEngine {
  constructor(url, telemetry) {
    this.el = new Audio(url);
    this.telemetry = telemetry;
    this._lastChord = null;
    this.el.preload = "auto";
    this.el.addEventListener("ended", () => this.onended && this.onended());
    this.ctx = null;
    this.onfxready = null; // (ctx, sourceNode) hook for the Phase I EQ chain
    this.onchord = null;
    this.onended = null;
  }
  get playing() {
    return !this.el.paused;
  }
  _ensureGraph() {
    if (this.ctx) return;
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
      const src = this.ctx.createMediaElementSource(this.el);
      this._fxIn = this.ctx.createGain();
      src.connect(this._fxIn);
      this._fxIn.connect(this.ctx.destination);
      if (this.onfxready) this.onfxready(this.ctx, this._fxIn);
    } catch {
      this.ctx = null; // element keeps playing direct; FX is enhancement-only
    }
  }
  /* Phase H preview: playbackRate resampling is a COUPLED preview (pitch
     and speed move together) — the rendered export applies them
     independently. */
  setAdjust({ semitones = 0, speed = 1 } = {}) {
    try {
      this.el.preservesPitch = false;
    } catch { /* older engines */ }
    this.el.playbackRate = coupledPlaybackRate(semitones, speed);
  }
  play() {
    this._ensureGraph();
    if (this.ctx && this.ctx.state === "suspended") this.ctx.resume();
    return this.el.play();
  }
  pause() {
    this.el.pause();
  }
  seek(t) {
    this.el.currentTime = Math.max(0, Math.min(t, this.telemetry.duration || t));
  }
  getTime() {
    this._syncChord();
    return this.el.currentTime;
  }
  setDeck() {
    /* no-op: single-source playback of the recording */
  }
  _syncChord() {
    const t = this.el.currentTime;
    const tl = this.telemetry.chord_timeline || [];
    let chord = null;
    if (tl.length) {
      const ev = tl.find((e) => t >= e.start && t < e.end);
      chord = ev ? ev.chord : null;
    }
    if (chord !== this._lastChord) {
      this._lastChord = chord;
      if (this.onchord) this.onchord(chord);
    }
  }
}

/* The Organization panel is rendered for EVERY track — analyzed or
   not — because grouping tracks must never depend on analysis state.
   (After a backend restart every track is unanalyzed; hiding this
   panel then made grouping impossible.) */
function orgPanelHtml(track) {
  return `
        <div class="panel">
          <h2>Organization</h2>
          <p class="panel-sub">Project &amp; notes — saved with the track</p>
          <label class="org-label">Project
            <select id="org-project">
              <option value="">No project (Unsorted)</option>
              ${state.projects.map((p) => `<option value="${esc(p.id)}"${track.project_id === p.id ? " selected" : ""}>${esc(p.name)}</option>`).join("")}
              <option value="__new__">＋ New project…</option>
            </select>
          </label>
          <div id="org-new-row" class="org-new-row hidden">
            <input id="org-new-name" type="text" placeholder="New project name" maxlength="200" />
            <button class="ghost-btn" id="org-new-create">Create</button>
          </div>
          <label class="org-label">Track notes
            <textarea id="org-notes" rows="3" placeholder="Mix notes, version context, TODOs…">${esc(track.notes || "")}</textarea>
          </label>
          <button class="cta-btn" id="org-save">Save</button>
        </div>`;
}

function wireOrgPanel(host, track) {
  /* Quick-create without leaving the track: the dropdown's
     "＋ New project…" entry opens an inline name row; creating rebuilds
     the dropdown with the new project selected, ready for Save. */
  const orgProjectSel = host.querySelector("#org-project");
  const orgNewRow = host.querySelector("#org-new-row");
  if (orgProjectSel && orgNewRow) {
    orgProjectSel.addEventListener("change", () => {
      const isNew = orgProjectSel.value === "__new__";
      orgNewRow.classList.toggle("hidden", !isNew);
      if (isNew) host.querySelector("#org-new-name").focus();
    });
    host.querySelector("#org-new-create").addEventListener("click", async () => {
      const name = host.querySelector("#org-new-name").value.trim();
      if (!name) return;
      const created = await createProject(name);
      if (!created) return; // createProject already toasted the failure
      orgProjectSel.innerHTML =
        `<option value="">No project (Unsorted)</option>` +
        state.projects
          .map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`)
          .join("") +
        `<option value="__new__">＋ New project…</option>`;
      orgProjectSel.value = created.id;
      orgNewRow.classList.add("hidden");
      toast(`Project "${created.name}" created — hit Save to put this track in it.`);
    });
  }
  const orgSave = host.querySelector("#org-save");
  if (orgSave) {
    orgSave.addEventListener("click", async () => {
      const rawProject = host.querySelector("#org-project").value;
      if (rawProject === "__new__") {
        toast("Name the new project and hit Create first — or pick an existing one.");
        return;
      }
      const projectId = rawProject || null;
      const notes = host.querySelector("#org-notes").value;
      if (apiLive) {
        try {
          const updated = await apiSend("PATCH", `/tracks/${track.id}`, {
            project_id: projectId,
            notes,
          });
          Object.assign(track, updated);
          state.projects = await loadProjects();
          toast("Track saved.");
        } catch (err) {
          toast("Save failed: " + err.message);
          return;
        }
      } else {
        track.project_id = projectId;
        track.notes = notes;
        persistLocalState();
        toast("Track saved in this browser.");
      }
      // Rebuild the library so the new project chip / filter row is
      // already correct when the user goes back — without this the
      // save looked like it did nothing.
      await renderLibrary();
    });
  }
}

function renderDetail(track, telemetry, audioUrl) {
  const host = document.getElementById("detail-content");
  const { h1, h2 } = coverHues(track.id);

  if (!telemetry) {
    const art0 = artworkUrl(track);
    host.innerHTML = `
      <div class="detail-head">
        <div class="detail-cover" style="--h1:${h1};--h2:${h2}">${art0 ? `<img class="cover-img" src="${art0}" alt="">` : ""}<span class="cover-glyph">♪</span></div>
        <div><h1>${esc(track.title)}</h1><p class="artist">${esc(track.artist)}</p></div>
      </div>
      <div class="panel"><h2>Analysis pending</h2>
        <p class="panel-sub">This track hasn't been analyzed yet.</p>
        <p style="color:var(--muted);font-size:14px">Run the analysis pipeline to generate its telemetry bundle — BPM, key, segments, chords, rhyme map, and essay will appear here. You can still organize it into a project below.</p>
      </div>
      ${audioUrl ? `<div class="panel"><h2>Recording</h2><audio controls preload="metadata" src="${audioUrl}" style="width:100%"></audio></div>` : ""}
      ${orgPanelHtml(track)}`;
    wireImgFallback(host);
    wireOrgPanel(host, track);
    return;
  }

  const moods = Object.entries(telemetry.emotional_profile || {}).sort((a, b) => b[1] - a[1]);
  const genreChip = telemetry.primary_genre
    ? `<span class="chip hot">Genre: ${esc(telemetry.primary_genre)}${
        telemetry.genre_predictions && telemetry.genre_predictions.length
          ? ` · ${(telemetry.genre_predictions[0].probability * 100).toFixed(1)}%`
          : ""
      }${telemetry.primary_subgenre ? ` → ${esc(telemetry.primary_subgenre)}` : ""}</span>`
    : "";
  const textureChips = (telemetry.texture_cues || [])
    .map((c) => `<span class="chip" title="${esc(c.note || "")}">${esc(c.name)}</span>`)
    .join("");
  const textureRow = textureChips
    ? `<div class="chip-row"><span class="chip-label">Texture cues</span>${textureChips}</div>`
    : "";

  const genrePanel = (() => {
    const preds = telemetry.genre_predictions || [];
    const predRows = preds
      .map(
        (p) => `
      <div class="mood-row">
        <div class="mood-top"><b>${esc(p.label)}</b><span>${(p.probability * 100).toFixed(1)}%</span></div>
        <div class="mood-bar"><div class="mood-fill" style="width:${(p.probability * 100).toFixed(1)}%"></div></div>
      </div>`
      )
      .join("");
    const head = telemetry.primary_genre
      ? `<p class="essay"><strong>${esc(telemetry.primary_genre)}${
          telemetry.primary_subgenre ? ` → ${esc(telemetry.primary_subgenre)}` : ""
        }</strong></p>`
      : `<p class="essay">No genre classification for this track.</p>`;
    const note = telemetry.genre_note ? `<p class="player-note">${esc(telemetry.genre_note)}</p>` : "";
    const model = telemetry.genre_model ? `<p class="player-note">Model: ${esc(telemetry.genre_model)}</p>` : "";
    const subPreds = telemetry.subgenre_predictions || [];
    const subRows = subPreds
      .map(
        (p) => `
      <div class="mood-row">
        <div class="mood-top"><b>${esc(p.label)}</b><span>${esc(p.family)} · ${Number(p.percent).toFixed(1)}%</span></div>
        <div class="mood-bar"><div class="mood-fill" style="width:${Number(p.percent).toFixed(1)}%"></div></div>
      </div>`
      )
      .join("");
    const subBlock = (subPreds.length || telemetry.subgenre_note)
      ? `<div class="subgenre-block">
          <h3>Subgenre matches · zero-shot similarity</h3>
          ${subRows}
          ${telemetry.subgenre_note ? `<p class="player-note">${esc(telemetry.subgenre_note)}</p>` : ""}
          ${telemetry.subgenre_model ? `<p class="player-note">Model: ${esc(telemetry.subgenre_model)}</p>` : ""}
        </div>`
      : "";
    return `${head}${predRows}${note}${model}${subBlock}`;
  })();
  const texturePanel = (telemetry.texture_cues || []).length
    ? (telemetry.texture_cues || [])
        .map(
          (c) => `<div class="mood-row"><div class="mood-top"><b>${esc(c.name)}</b></div><p class="player-note">${esc(c.note || "")}</p></div>`
        )
        .join("")
    : `<p class="essay">No texture cues recorded.</p>`;
  const vp = telemetry.vocal_profile;
  const vocalHtml = vp
    ? `<div class="tele-grid">
         <div class="tele-cell"><div class="k">Mean pitch</div><div class="v">${vp.f0_mean_hz} <small>Hz</small></div></div>
         <div class="tele-cell"><div class="k">Pitch range</div><div class="v">${vp.f0_range_semitones} <small>st</small></div></div>
         <div class="tele-cell"><div class="k">Vibrato</div><div class="v">${vp.vibrato_rate_hz} <small>Hz</small></div></div>
         <div class="tele-cell"><div class="k">Autotune · heuristic</div><div class="v">${Math.round(vp.autotune_likelihood * 100)}<small>%</small></div></div>
       </div>
       <p class="player-note">Autotune likelihood is a heuristic score, not a verdict — see the full report for method notes.</p>`
    : `<p class="essay">${esc(telemetry.vocal_note || "No vocal profile was measured for this track.")}</p>`;
  const keyConfCell =
    typeof telemetry.key_confidence === "number"
      ? `<div class="tele-cell"><div class="k">Key confidence</div><div class="v">${Math.round(telemetry.key_confidence * 100)}<small>%</small></div></div>`
      : "";
  const mm = telemetry.mix_measurements;
  const mixCells = mm
    ? `<div class="tele-cell"><div class="k">Peak</div><div class="v">${mm.peak_dbfs} <small>dBFS</small></div></div>
       <div class="tele-cell"><div class="k">Crest</div><div class="v">${mm.crest_db} <small>dB</small></div></div>
       <div class="tele-cell"><div class="k">Sub share</div><div class="v">${mm.sub_pct}<small>%</small></div></div>`
    : "";

  host.innerHTML = `
    <div class="detail-head">
      <div class="detail-cover" style="--h1:${h1};--h2:${h2}">${artworkUrl(track) ? `<img class="cover-img" src="${artworkUrl(track)}" alt="">` : ""}<span class="cover-glyph">♪</span></div>
      <div>
        <h1>${esc(track.title)}</h1>
        <p class="artist">${esc(track.artist)}</p>
        <div class="chip-row">
          <span class="chip hot">${telemetry.bpm} BPM</span>
          <span class="chip">${esc(telemetry.musical_key)} ${esc(telemetry.mode)}</span>
          <span class="chip">${esc(telemetry.rhyme_scheme_summary)} rhyme</span>
          ${genreChip}
        </div>
        ${textureRow}
        ${apiLive ? `<div class="detail-actions"><button class="ghost-btn report-btn" id="report-btn">⭳ Analysis report (print / PDF)</button></div>` : ""}
        ${(() => {
          const others = state.tracks.filter((t) => t.id !== track.id && state.telemetry[t.id]);
          if (!others.length) return "";
          return `<div class="detail-actions cmp-actions">
            <label>Compare with
              <select id="cmp-select">${others.map((t) => `<option value="${esc(t.id)}">${esc(t.title)}</option>`).join("")}</select>
            </label>
            <button class="ghost-btn" id="cmp-open">⇄ Compare</button>
          </div>`;
        })()}
      </div>
    </div>

    <div class="panel">
      <h2>Player</h2>
      <p class="panel-sub">Dual deck · structural timeline · synthesized from measured telemetry</p>
      <div class="transport">
        <button class="big-play" id="play-btn" aria-label="Play">▶</button>
        <div class="time-read"><b id="t-cur">0:00</b> / <span id="t-dur">${fmtTime(telemetry.duration)}</span></div>
        ${audioUrl
          ? `<span class="chip">full recording</span>`
          : `<div class="deck-toggle" role="tablist" aria-label="Deck">
              <button id="deck-a" class="active">Deck A · Full mix</button>
              <button id="deck-b">Deck B · Stripped</button>
            </div>`}
      </div>
      <div id="wave-wrap"><canvas id="wave"></canvas></div>
      <div class="seg-legend" id="seg-legend"></div>
      <div class="chord-strip" id="chord-strip"></div>
      <p class="player-note">${audioUrl
        ? "Playing your uploaded recording — waveform, chords, and lyrics stay synced to its measured telemetry."
        : "Audio is synthesized live from the track's measured chords &amp; tempo — a stand-in when no recording is available."} Click the waveform or any lyric line to seek.</p>
    </div>

    <div class="detail-grid">
      <div>
        <div class="panel">
          <h2>Pitch &amp; speed</h2>
          <p class="panel-sub">Preview is a coupled resample; the rendered export applies pitch and speed independently</p>
          <div class="fx-row">
            <label>Semitones <b id="adj-semi-val">+0.0 st</b>
              <input id="adj-semi" type="range" min="-12" max="12" step="0.5" value="0" />
            </label>
            <label>Speed <b id="adj-speed-val">1.00×</b>
              <input id="adj-speed" type="range" min="0.5" max="2" step="0.05" value="1" />
            </label>
            <button class="ghost-btn" id="adj-reset">Reset</button>
          </div>
          <p class="player-note" id="adj-derived"></p>
          ${apiLive ? `<div class="fx-row">
            <button class="cta-btn" id="adj-render">Render adjusted audio</button>
            <audio id="adj-audio" class="hidden" controls preload="none"></audio>
          </div>
          <p class="player-note">Rendered with librosa (pitch shift + time stretch, independently) and cached on the server. The original file and its analysis are never modified.</p>` : ""}
        </div>
        <div class="panel">
          <h2>EQ · playback only</h2>
          <p class="panel-sub">Six-band parametric EQ on what you hear — never on the file or its analysis</p>
          <div class="eq-presets" id="eq-presets">
            ${Object.keys(EQ_PRESETS).map((name) => `<button class="chip eq-preset${fx.settings.preset === name ? " active" : ""}" data-preset="${esc(name)}">${esc(name)}</button>`).join("")}
            <label class="eq-bypass"><input type="checkbox" id="eq-bypass" ${fx.settings.bypass ? "checked" : ""} /> Bypass</label>
          </div>
          <canvas id="eq-curve" width="640" height="150"></canvas>
          <div id="eq-bands"></div>
        </div>
      </div>
      <div>
        <div class="panel">
          <h2>Mix suggestions</h2>
          <p class="panel-sub">Fired by deterministic rules over measured values — each quotes its evidence</p>
          <div id="sugg-list"></div>
        </div>
        ${orgPanelHtml(track)}
      </div>
    </div>

    <div class="detail-grid">
      <div>
        <div class="panel">
          <h2>Lyrics · time-synced</h2>
          <p class="panel-sub">${(telemetry.timestamped_lyrics || []).length
            ? "Lyric sheet supplied with the track · click a line to seek"
            : "No lyric sheet supplied with this track"}</p>
          <div class="lyrics" id="lyrics"></div>
        </div>
        <div class="panel">
          <h2>Musicological essay</h2>
          <p class="panel-sub">Grounded in locked telemetry — no generative guessing</p>
          <p class="essay">${esc(telemetry.musicological_essay)}</p>
        </div>
      </div>
      <div>
        <div class="panel">
          <h2>Telemetry</h2>
          <p class="panel-sub">Deterministic measurements</p>
          <div class="tele-grid">
            <div class="tele-cell"><div class="k">Tempo</div><div class="v">${telemetry.bpm} <small>BPM</small></div></div>
            <div class="tele-cell"><div class="k">Key</div><div class="v">${esc(telemetry.musical_key)} <small>${esc(telemetry.mode)}</small></div></div>
            <div class="tele-cell"><div class="k">Length</div><div class="v">${fmtTime(telemetry.duration)}</div></div>
            <div class="tele-cell"><div class="k">Sections</div><div class="v">${telemetry.segments.length}</div></div>
            ${keyConfCell}
            ${mixCells}
          </div>
          <div id="moods"></div>
        </div>
        <div class="panel">
          <h2>Genre · trained model + subgenre</h2>
          <p class="panel-sub">PANNs Cnn10 (AudioSet) — real model probabilities, not rules</p>
          ${genrePanel}
        </div>
        <div class="panel">
          <h2>Texture cues</h2>
          <p class="panel-sub">Deterministic style rules over texture — texture is not genre</p>
          ${texturePanel}
        </div>
        <div class="panel">
          <h2>Vocal forensics</h2>
          <p class="panel-sub">Measured pitch behavior — heuristics labeled as heuristics</p>
          ${vocalHtml}
        </div>
      </div>
    </div>`;

  wireImgFallback(host);
  const reportBtn = host.querySelector("#report-btn");
  if (reportBtn) reportBtn.addEventListener("click", () => openReport(track.id));

  // Moods
  const moodsEl = host.querySelector("#moods");
  moodsEl.innerHTML = moods
    .map(
      ([name, v]) => `
      <div class="mood-row">
        <div class="mood-top"><b>${esc(name)}</b><span>${Math.round(v * 100)}%</span></div>
        <div class="mood-bar"><div class="mood-fill" style="width:${Math.round(v * 100)}%"></div></div>
      </div>`
    )
    .join("");

  // Lyrics
  const lyricsEl = host.querySelector("#lyrics");
  const lines = telemetry.timestamped_lyrics || [];
  lyricsEl.innerHTML = lines.length
    ? lines
    .map((l, i) => {
      const conf = typeof l.confidence === "number" ? l.confidence : null;
      const low = conf !== null && conf < 0.5;
      return `
      <div class="lyric-line${low ? " lowconf" : ""}" data-i="${i}"${
        conf !== null ? ` title="line confidence ${Math.round(conf * 100)}%"` : ""
      }>
        <span class="ts">${fmtTime(l.start)} → ${fmtTime(l.end)}</span>
        <span class="txt">${esc(l.text)}</span>
        ${low ? `<span class="conf-badge">${Math.round(conf * 100)}%</span>` : ""}
        <span class="rhyme">${esc(l.rhyme)}</span>
      </div>`;
    })
    .join("")
    : `<p class="essay">${esc(telemetry.lyrics_note || "No lyric sheet was supplied for this track.")}</p>`;
  const lineEls = [...lyricsEl.querySelectorAll(".lyric-line")];

  // Segments legend
  const legend = host.querySelector("#seg-legend");
  legend.innerHTML = "";
  const segPills = telemetry.segments.map((seg, i) => {
    const b = document.createElement("button");
    b.className = "seg-pill";
    b.style.setProperty("--c", SEG_COLORS[i % SEG_COLORS.length]);
    b.innerHTML = `<span style="color:${SEG_COLORS[i % SEG_COLORS.length]}">●</span> ${esc(seg.name)} · ${fmtTime(seg.start)}`;
    b.addEventListener("click", () => state.detail.engine.seek(seg.start));
    legend.appendChild(b);
    return b;
  });

  // Chord strip
  const strip = host.querySelector("#chord-strip");
  strip.innerHTML = "";
  const chordChips = telemetry.chords.map((c) => {
    const s = document.createElement("div");
    s.className = "chord-chip";
    s.textContent = c;
    strip.appendChild(s);
    return s;
  });

  // Engine — real recording when available, synth stand-in otherwise.
  const engine = audioUrl
    ? new RealAudioEngine(audioUrl, telemetry)
    : new SynthEngine({
        bpm: telemetry.bpm,
        chords: telemetry.chords,
        duration: telemetry.duration,
      });
  engine.onchord = (chord) => {
    const idx = telemetry.chords.indexOf(chord);
    chordChips.forEach((el, i) => el.classList.toggle("active", i === idx));
  };
  engine.onended = () => {
    document.getElementById("play-btn").textContent = "▶";
    document.getElementById("play-btn").disabled = false;
  };

  // Real measured peaks when the analysis pipeline has produced them;
  // otherwise the deterministic seeded placeholder.
  const peaks =
    telemetry.waveform_peaks && telemetry.waveform_peaks.length
      ? telemetry.waveform_peaks
      : makePeaks(track.id);
  const wave = host.querySelector("#wave");
  const tCur = host.querySelector("#t-cur");
  const playBtn = host.querySelector("#play-btn");

  wave.addEventListener("click", (e) => {
    const rect = wave.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    engine.seek(ratio * telemetry.duration);
  });

  lineEls.forEach((el, i) => {
    el.addEventListener("click", () => engine.seek(lines[i].start + 0.01));
  });

  playBtn.addEventListener("click", () => {
    try {
      if (engine.playing) {
        engine.pause();
        playBtn.textContent = "▶";
      } else {
        const p = engine.play();
        if (p && typeof p.catch === "function") {
          p.catch((err) => toast("Audio couldn't start: " + err.message));
        }
        playBtn.textContent = "⏸";
      }
    } catch (err) {
      toast("Audio couldn't start: " + err.message);
    }
  });

  const deckA = host.querySelector("#deck-a");
  const deckB = host.querySelector("#deck-b");
  if (deckA && deckB) {
    deckA.addEventListener("click", (e) => setDeck("A", e));
    deckB.addEventListener("click", (e) => setDeck("B", e));
  }
  function setDeck(d, e) {
    engine.setDeck(d);
    host.querySelector("#deck-a").classList.toggle("active", d === "A");
    host.querySelector("#deck-b").classList.toggle("active", d === "B");
  }

  state.detail = { track, telemetry, engine, peaks, activeLyric: -1, activeSeg: -1 };

  /* ---- Phase E: compare entry ---- */
  const cmpOpen = host.querySelector("#cmp-open");
  if (cmpOpen) {
    cmpOpen.addEventListener("click", () => {
      const other = host.querySelector("#cmp-select").value;
      if (other) openCompare(track.id, other);
    });
  }

  /* ---- Phase G: mix suggestions ---- */
  const suggHost = host.querySelector("#sugg-list");
  if (suggHost) {
    const suggs = telemetry.suggestions || [];
    suggHost.innerHTML = suggs.length
      ? suggs
          .map(
            (s) => `<div class="sugg sugg-${esc(s.severity)}">
              <div class="sugg-top"><span class="chip sev-chip sev-${esc(s.severity)}">${esc(s.severity)}</span><b>${esc(s.area)}</b></div>
              <p class="sugg-text">${esc(s.suggestion)}</p>
              <p class="sugg-why">Why: ${esc(s.why)}</p>
            </div>`
          )
          .join("")
      : `<p class="essay">${telemetry.mix_measurements ? "No rules fired — every measured axis sits within its threshold." : "Mix measurements weren't recorded for this analysis."}</p>`;
  }

  /* ---- Phase F: organization (project + notes) ---- */
  wireOrgPanel(host, track);

  /* ---- Phase H: pitch & speed ---- */
  fx.adjust = { semitones: 0, speed: 1 };
  engine.onfxready = attachFx;
  const semiEl = host.querySelector("#adj-semi");
  const speedEl = host.querySelector("#adj-speed");
  if (semiEl && speedEl) {
    const semiVal = host.querySelector("#adj-semi-val");
    const speedVal = host.querySelector("#adj-speed-val");
    const derived = host.querySelector("#adj-derived");
    const applyAdjust = () => {
      fx.adjust = clampAdjust(parseFloat(semiEl.value), parseFloat(speedEl.value));
      semiVal.textContent = `${fx.adjust.semitones >= 0 ? "+" : ""}${fx.adjust.semitones.toFixed(1)} st`;
      speedVal.textContent = `${fx.adjust.speed.toFixed(2)}×`;
      derived.textContent =
        `Derived from the measured analysis (not re-measured): ` +
        `${effectiveBpm(telemetry.bpm, fx.adjust.speed)} BPM · ` +
        `${transposeKey(telemetry.musical_key, fx.adjust.semitones)} ${telemetry.mode}`;
      if (engine.setAdjust) engine.setAdjust(fx.adjust);
    };
    semiEl.addEventListener("input", applyAdjust);
    speedEl.addEventListener("input", applyAdjust);
    host.querySelector("#adj-reset").addEventListener("click", () => {
      semiEl.value = "0";
      speedEl.value = "1";
      applyAdjust();
    });
    applyAdjust();
    const renderBtn = host.querySelector("#adj-render");
    if (renderBtn) {
      renderBtn.addEventListener("click", async () => {
        const { semitones, speed } = fx.adjust;
        const url = `${API_BASE}/tracks/${track.id}/adjusted?semitones=${semitones}&speed=${speed}`;
        const audioEl = host.querySelector("#adj-audio");
        renderBtn.disabled = true;
        renderBtn.textContent = "Rendering…";
        try {
          let key = null;
          try { key = localStorage.getItem("pythagora_api_key"); } catch { /* ignore */ }
          if (key) {
            const res = await fetch(url, { headers: apiHeaders() });
            if (!res.ok) throw await apiError(res);
            audioEl.src = URL.createObjectURL(await res.blob());
          } else {
            audioEl.src = url;
          }
          audioEl.classList.remove("hidden");
          await audioEl.play().catch(() => {});
        } catch (err) {
          toast("Render failed: " + err.message);
        }
        renderBtn.disabled = false;
        renderBtn.textContent = "Render adjusted audio";
      });
    }
  }

  /* ---- Phase I: EQ wiring ---- */
  renderEqControls();
  drawEqCurve();
  host.querySelectorAll(".eq-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      fx.settings.preset = btn.dataset.preset;
      fx.settings.bands = presetBandState(btn.dataset.preset);
      if (fx.chain) fx.chain.applySettings(fx.settings);
      saveEqSettings(fx.settings);
      host.querySelectorAll(".eq-preset").forEach((b) =>
        b.classList.toggle("active", b === btn)
      );
      renderEqControls();
      drawEqCurve();
    });
  });
  const bypassEl = host.querySelector("#eq-bypass");
  if (bypassEl) {
    bypassEl.addEventListener("change", () => {
      fx.settings.bypass = bypassEl.checked;
      if (fx.chain) fx.chain.setBypass(bypassEl.checked);
      saveEqSettings(fx.settings);
      drawEqCurve();
    });
  }


  const loop = () => {
    const d = state.detail;
    if (!d) return;
    const t = engine.getTime();
    tCur.textContent = fmtTime(t);
    drawWave(wave, peaks, telemetry.segments, telemetry.duration, t);

    // Lyrics sync
    let li = lines.findIndex((l) => t >= l.start && t < l.end);
    if (li !== d.activeLyric) {
      d.activeLyric = li;
      lineEls.forEach((el, i) => {
        el.classList.toggle("active", i === li);
        el.classList.toggle("past", li >= 0 && i < li);
      });
      if (li >= 0) lineEls[li].scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    // Segment sync
    const si = telemetry.segments.findIndex((s) => t >= s.start && t < s.end);
    if (si !== d.activeSeg) {
      d.activeSeg = si;
      segPills.forEach((el, i) => {
        el.classList.toggle("active", i === si);
        if (i === si) el.style.background = SEG_COLORS[i % SEG_COLORS.length] + "55";
        else el.style.background = "";
      });
    }
    d.raf = requestAnimationFrame(loop);
  };
  loop();
}

/* ------------------------------------------------------------------ */
/* EQ controls + curve (Phase I)                                      */
/* ------------------------------------------------------------------ */
function drawEqCurve() {
  const canvas = document.getElementById("eq-curve");
  if (!canvas) return;
  const g = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  g.clearRect(0, 0, w, h);
  const N = 96;
  const freqs = [];
  for (let i = 0; i < N; i++) freqs.push(20 * Math.pow(16000 / 20, i / (N - 1)));
  let db;
  if (fx.settings.bypass) {
    db = freqs.map(() => 0);
  } else if (fx.chain && fx.ctx) {
    db = fx.chain.getResponseDb(new Float32Array(freqs));
  } else {
    db = freqs.map((f) => combinedResponseDb(fx.settings.bands, f, 44100));
  }
  const range = 18;
  const yOf = (v) => h / 2 - (Math.max(-range, Math.min(range, v)) / range) * (h / 2 - 8);
  g.strokeStyle = "rgba(255,255,255,0.16)";
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(0, yOf(0));
  g.lineTo(w, yOf(0));
  g.stroke();
  g.strokeStyle = "#e50914";
  g.lineWidth = 2;
  g.beginPath();
  freqs.forEach((f, i) => {
    const x = (i / (N - 1)) * w;
    const y = yOf(db[i]);
    if (i) g.lineTo(x, y);
    else g.moveTo(x, y);
  });
  g.stroke();
  g.fillStyle = "rgba(255,255,255,0.45)";
  g.font = "10px Inter, sans-serif";
  g.fillText("20 Hz", 4, h - 6);
  g.fillText("16 kHz", w - 38, h - 6);
  g.fillText("+18 dB", 4, 12);
}

function renderEqControls() {
  const host = document.getElementById("eq-bands");
  if (!host) return;
  host.innerHTML = EQ_BANDS.map((def, i) => {
    const s = fx.settings.bands[i];
    return `<div class="eq-band">
      <div class="eq-band-head"><b>${esc(def.label)}</b><span>${esc(def.type)}</span></div>
      <label>Freq <input type="range" class="eq-f" data-i="${i}" min="${def.fMin}" max="${def.fMax}" step="1" value="${s.f}" /><span class="eq-val eq-fval">${Math.round(s.f)} Hz</span></label>
      <label>Gain <input type="range" class="eq-g" data-i="${i}" min="-15" max="15" step="0.5" value="${s.g}" /><span class="eq-val eq-gval">${(+s.g).toFixed(1)} dB</span></label>
      ${def.type === "peaking" ? `<label>Q <input type="range" class="eq-q" data-i="${i}" min="0.3" max="4" step="0.1" value="${s.q}" /><span class="eq-val eq-qval">${(+s.q).toFixed(1)}</span></label>` : ""}
    </div>`;
  }).join("");
  host.querySelectorAll("input").forEach((inp) => {
    inp.addEventListener("input", () => {
      const i = +inp.dataset.i;
      const bandState = fx.settings.bands[i];
      if (inp.classList.contains("eq-f")) {
        bandState.f = +inp.value;
        inp.closest("label").querySelector(".eq-fval").textContent = `${Math.round(bandState.f)} Hz`;
      }
      if (inp.classList.contains("eq-g")) {
        bandState.g = +inp.value;
        inp.closest("label").querySelector(".eq-gval").textContent = `${bandState.g.toFixed(1)} dB`;
      }
      if (inp.classList.contains("eq-q")) {
        bandState.q = +inp.value;
        inp.closest("label").querySelector(".eq-qval").textContent = bandState.q.toFixed(1);
      }
      fx.settings.preset = "Custom";
      document.querySelectorAll(".eq-preset").forEach((b) => b.classList.remove("active"));
      if (fx.chain) fx.chain.setBand(i, bandState);
      saveEqSettings(fx.settings);
      drawEqCurve();
    });
  });
}

/* ------------------------------------------------------------------ */
/* Track comparison (Phase E)                                         */
/* ------------------------------------------------------------------ */
function localComparison(aId, bId) {
  const ta = state.tracks.find((t) => t.id === aId);
  const tb = state.tracks.find((t) => t.id === bId);
  const A = state.telemetry[aId];
  const B = state.telemetry[bId];
  if (!ta || !tb || !A || !B) return null;
  const lvl = (t, f) => (t.mix_measurements ? t.mix_measurements[f] : null);
  const vocal = (t) =>
    t.vocal_profile
      ? `F0 mean ${Math.round(t.vocal_profile.f0_mean_hz)} Hz · voiced ${Math.round(t.vocal_profile.voiced_fraction * 100)}%`
      : t.vocal_note || "Not measured";
  const rows = [];
  const row = (field, label, a, b, delta) =>
    rows.push({ field, label, a, b, delta: delta === undefined ? null : delta, different: a !== b });
  row("duration", "Duration (s)", A.duration, B.duration, Math.round((B.duration - A.duration) * 10) / 10);
  row("bpm", "Tempo (BPM)", A.bpm, B.bpm, Math.round((B.bpm - A.bpm) * 10) / 10);
  row("key", "Key", `${A.musical_key} ${A.mode}`, `${B.musical_key} ${B.mode}`);
  row("key_confidence", "Key confidence", A.key_confidence ?? null, B.key_confidence ?? null);
  row("peak_dbfs", "Peak (dBFS)", lvl(A, "peak_dbfs"), lvl(B, "peak_dbfs"));
  row("crest_db", "Crest / dynamics (dB)", lvl(A, "crest_db"), lvl(B, "crest_db"));
  row("sub_pct", "Sub-bass share (%)", lvl(A, "sub_pct"), lvl(B, "sub_pct"));
  row("segments", "Sections", (A.segments || []).length, (B.segments || []).length);
  row("unique_chords", "Unique chords", (A.chords || []).length, (B.chords || []).length);
  row("vocal", "Vocal forensics", vocal(A), vocal(B));
  row("suggestions", "Mix suggestions", (A.suggestions || []).length, (B.suggestions || []).length);
  const chordsA = A.chords || [];
  const chordsB = B.chords || [];
  return {
    a: { id: ta.id, title: ta.title, artist: ta.artist },
    b: { id: tb.id, title: tb.title, artist: tb.artist },
    rows,
    chords: {
      shared: chordsA.filter((c) => chordsB.includes(c)),
      only_a: chordsA.filter((c) => !chordsB.includes(c)),
      only_b: chordsB.filter((c) => !chordsA.includes(c)),
    },
    summary: {
      fields_compared: rows.length,
      fields_different: rows.filter((r) => r.different).length,
    },
  };
}

function renderComparisonTable(cmp) {
  const fmt = (v) => (v === null || v === undefined ? "—" : typeof v === "number" ? String(Math.round(v * 100) / 100) : esc(String(v)));
  const rows = cmp.rows
    .map(
      (r) => `<tr class="${r.different ? "diff" : ""}"><td>${esc(r.label)}</td><td>${fmt(r.a)}</td><td>${fmt(r.b)}</td></tr>`
    )
    .join("");
  return `
    <p class="panel-sub">${cmp.summary.fields_different} of ${cmp.summary.fields_compared} compared measurements differ.</p>
    <table class="cmp-table">
      <tr><th>Measurement</th><th>${esc(cmp.a.title)}</th><th>${esc(cmp.b.title)}</th></tr>
      ${rows}
    </table>
    <p class="player-note">Shared chords: ${esc(cmp.chords.shared.join(", ") || "none")} ·
    only ${esc(cmp.a.title)}: ${esc(cmp.chords.only_a.join(", ") || "—")} ·
    only ${esc(cmp.b.title)}: ${esc(cmp.chords.only_b.join(", ") || "—")}</p>`;
}

async function openCompare(aId, bId) {
  state.comparePair = { a: aId, b: bId };
  const modal = document.getElementById("compare-modal");
  const content = document.getElementById("cmp-content");
  modal.classList.remove("hidden");
  content.innerHTML = `<p class="panel-sub">Loading comparison…</p>`;
  let cmp = null;
  if (apiLive) {
    try {
      cmp = await apiGet(`/tracks/compare?a=${encodeURIComponent(aId)}&b=${encodeURIComponent(bId)}`);
    } catch { /* fall through to local */ }
  }
  if (!cmp) cmp = localComparison(aId, bId);
  if (!cmp) {
    content.innerHTML = `<p class="essay">Comparison needs both tracks to be analyzed.</p>`;
    return;
  }
  document.getElementById("cmp-title").textContent = `${cmp.a.title} vs ${cmp.b.title}`;
  content.innerHTML = renderComparisonTable(cmp);
  document.getElementById("cmp-report").classList.toggle("hidden", !apiLive);
}

function wireCompareModal() {
  const modal = document.getElementById("compare-modal");
  const close = () => modal.classList.add("hidden");
  document.getElementById("cmp-close").addEventListener("click", close);
  document.getElementById("cmp-done").addEventListener("click", close);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });
  document.getElementById("cmp-report").addEventListener("click", () => {
    if (!state.comparePair) return;
    const { a, b } = state.comparePair;
    openApiPage(`${API_BASE}/tracks/compare/report?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  });
}

/* ------------------------------------------------------------------ */
/* Upload + job tracking — drag & drop ingest with live previews      */
/* ------------------------------------------------------------------ */
const upState = { audioFiles: [], artFile: null, artUrl: null };

/* Fallback stage list — mirrors workers/pipeline.py STAGES. The live
   job payload's own `stages` array takes precedence when present. */
const PIPE_STAGES = [
  "Loading audio",
  "Measuring tempo & key",
  "Recognizing chords",
  "Mapping sections & texture",
  "Classifying genre",
  "Rendering waveform",
  "Vocal forensics",
  "Measuring mix & suggestions",
  "Finalizing",
];

const AUDIO_EXT_RE = /\.(mp3|wav|ogg|flac|m4a|aac)$/i;

function syncPreview() {
  const t = document.getElementById("up-title").value.trim() || "Untitled";
  const a = document.getElementById("up-artist").value.trim() || "Unknown Artist";
  document.getElementById("up-preview-title").textContent = t;
  document.getElementById("up-preview-artist").textContent = a;
  document.getElementById("up-prog-title").textContent = `Analyzing “${t}”`;
}

function drawMiniWave(canvas, peaks) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const bw = w / peaks.length;
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, "#e50914");
  grad.addColorStop(1, "#f5b544");
  ctx.fillStyle = grad;
  peaks.forEach((p, i) => {
    const ph = Math.max(2, p * h);
    ctx.fillRect(i * bw, (h - ph) / 2, Math.max(1, bw - 1), ph);
  });
}

function titleFromFilename(name) {
  return name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim() || "Untitled";
}

function renderFileList() {
  const host = document.getElementById("up-file-list");
  const files = upState.audioFiles;
  host.classList.toggle("hidden", files.length === 0);
  host.innerHTML = files
    .map(
      (f, i) => `<div class="up-file-row">
        <span class="up-file-name">${esc(f.name)}</span>
        <span class="up-file-size">${(f.size / 1048576).toFixed(1)} MB</span>
        <button class="dz-clear up-file-x" data-i="${i}" aria-label="Remove file">×</button>
      </div>`
    )
    .join("");
  host.querySelectorAll(".up-file-x").forEach((btn) =>
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      upState.audioFiles.splice(+btn.dataset.i, 1);
      if (!upState.audioFiles.length) clearAudio();
      else renderFileList();
    })
  );
}

async function previewFirstFile(file) {
  document.getElementById("up-audio-idle").classList.add("hidden");
  document.getElementById("up-audio-file").classList.remove("hidden");
  document.getElementById("up-audio-name").textContent =
    upState.audioFiles.length > 1 ? `${file.name} (+${upState.audioFiles.length - 1} more)` : file.name;
  document.getElementById("up-audio-sub").textContent = "decoding…";
  try {
    const ab = await file.arrayBuffer();
    const AC = window.AudioContext || window.webkitAudioContext;
    const ac = new AC();
    const buf = await ac.decodeAudioData(ab);
    const ch = buf.getChannelData(0);
    const n = 120;
    const peaks = [];
    const block = Math.max(1, Math.floor(ch.length / n));
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (let j = i * block; j < Math.min((i + 1) * block, ch.length); j += 9) {
        m = Math.max(m, Math.abs(ch[j]));
      }
      peaks.push(Math.min(1, m));
    }
    drawMiniWave(document.getElementById("up-wave"), peaks);
    document.getElementById("up-audio-sub").textContent =
      `${fmtTime(buf.duration)} · ${(file.size / 1048576).toFixed(1)} MB`;
    if (upState.audioFiles.length === 1 && !document.getElementById("up-title").value.trim()) {
      document.getElementById("up-title").value = titleFromFilename(file.name);
      syncPreview();
    }
    ac.close();
  } catch {
    document.getElementById("up-audio-sub").textContent = `${(file.size / 1048576).toFixed(1)} MB`;
  }
}

function handleAudioFiles(fileList) {
  const files = [...fileList].filter((f) => AUDIO_EXT_RE.test(f.name));
  if (!files.length) {
    toast("Unsupported audio type.");
    return;
  }
  if (files.length !== fileList.length) toast("Skipped non-audio files.");
  upState.audioFiles = files;
  renderFileList();
  previewFirstFile(files[0]);
}

function handleArtFile(file) {
  if (!file) return;
  if (!/\.(jpe?g|png|webp)$/i.test(file.name)) {
    toast("Unsupported image type.");
    return;
  }
  upState.artFile = file;
  if (upState.artUrl) URL.revokeObjectURL(upState.artUrl);
  upState.artUrl = URL.createObjectURL(file);
  const img = document.getElementById("up-art-preview");
  img.src = upState.artUrl;
  img.classList.remove("hidden");
  document.getElementById("up-art-idle").classList.add("hidden");
  document.getElementById("up-art-clear").classList.remove("hidden");
  const cover = document.getElementById("up-preview-cover");
  cover.querySelector("img")?.remove();
  const ci = document.createElement("img");
  ci.className = "cover-img";
  ci.src = upState.artUrl;
  ci.alt = "";
  cover.prepend(ci);
}

function clearAudio() {
  upState.audioFiles = [];
  document.getElementById("up-audio").value = "";
  document.getElementById("up-audio-idle").classList.remove("hidden");
  document.getElementById("up-audio-file").classList.add("hidden");
  renderFileList();
}

function clearArt() {
  upState.artFile = null;
  if (upState.artUrl) URL.revokeObjectURL(upState.artUrl);
  upState.artUrl = null;
  document.getElementById("up-artwork").value = "";
  document.getElementById("up-art-idle").classList.remove("hidden");
  document.getElementById("up-art-preview").classList.add("hidden");
  document.getElementById("up-art-clear").classList.add("hidden");
  document.querySelector("#up-preview-cover img")?.remove();
}

function resetUploadModal() {
  clearAudio();
  clearArt();
  document.getElementById("up-title").value = "";
  document.getElementById("up-artist").value = "";
  document.getElementById("up-form").classList.remove("hidden");
  document.getElementById("up-offline").classList.add("hidden");
  document.getElementById("up-local-note").classList.add("hidden");
  document.getElementById("up-progress").classList.add("hidden");
  document.getElementById("up-back").classList.add("hidden");
  document.getElementById("up-queue").innerHTML = "";
  const btn = document.getElementById("up-submit");
  btn.disabled = false;
  btn.textContent = "Upload & analyze";
  syncPreview();
}

function wireDropzone(dzId, inputId, handler) {
  const dz = document.getElementById(dzId);
  const input = document.getElementById(inputId);
  dz.addEventListener("click", (e) => {
    if (!e.target.closest(".dz-clear")) input.click();
  });
  input.addEventListener("change", () => {
    if (input.files.length) handler(input.files);
  });
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.remove("drag");
    })
  );
  dz.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) handler(e.dataTransfer.files);
  });
}

/* Render a stage stepper from a job's own stage list (statuses come from
   the API: pending | active | done | skipped). */
function stageListHtml(stages) {
  return stages
    .map((s) => `<li class="${esc(s.status)}"><span class="st-ic"></span><span>${esc(s.name)}</span></li>`)
    .join("");
}

function renderFocusedStages(job) {
  const stages =
    job && job.stages && job.stages.length
      ? job.stages
      : PIPE_STAGES.map((name) => ({ name, status: "pending" }));
  document.getElementById("up-stages").innerHTML = stageListHtml(stages);
  document.getElementById("up-prog-fill").style.width = `${job ? job.progress || 0 : 0}%`;
}

function queueCardHtml(entry) {
  const job = entry.job;
  const status = entry.error
    ? `<span class="q-status q-failed">failed — ${esc(entry.error)}</span>`
    : job
      ? job.state === "done"
        ? `<span class="q-status q-done">done</span>`
        : job.state === "failed"
          ? `<span class="q-status q-failed">failed — ${esc(job.error || "analysis error")}</span>`
          : `<span class="q-status">${esc(job.stage || job.state)} · ${job.progress || 0}%${typeof job.elapsed_seconds === "number" ? ` · ${job.elapsed_seconds.toFixed(0)}s` : ""}</span>`
      : `<span class="q-status">uploading…</span>`;
  const stages = job && job.stages && job.stages.length ? job.stages : null;
  return `
    <div class="q-top"><b>${esc(entry.title)}</b>${status}</div>
    <div class="q-bar"><div class="q-fill" style="width:${job ? job.progress || 0 : 0}%"></div></div>
    ${stages ? `<ol class="pipe-stages q-stages">${stageListHtml(stages)}</ol>` : ""}
    ${job && job.state === "done" ? `<button class="ghost-btn q-open" data-track="${esc(entry.trackId)}">Open track →</button>` : ""}`;
}

function renderQueue(entries) {
  const host = document.getElementById("up-queue");
  host.innerHTML = "";
  for (const entry of entries) {
    const card = document.createElement("div");
    card.className = "q-card";
    card.innerHTML = queueCardHtml(entry);
    const openBtn = card.querySelector(".q-open");
    if (openBtn) {
      openBtn.addEventListener("click", () => {
        document.getElementById("upload-modal").classList.add("hidden");
        openDetail(entry.trackId);
      });
    }
    host.appendChild(card);
  }
}

function wireUpload() {
  const modal = document.getElementById("upload-modal");
  const showConnectPanel = () => {
    document.getElementById("up-form").classList.add("hidden");
    document.getElementById("up-progress").classList.add("hidden");
    document.getElementById("up-offline").classList.remove("hidden");
    const input = document.getElementById("up-api-url");
    try {
      input.value = window.localStorage.getItem("pythagora_api_base") || "";
    } catch { /* storage unavailable */ }
  };
  const open = () => {
    resetUploadModal();
    if (!apiLive) {
      // Local mode: the familiar form, working entirely in the
      // browser — tracks can be added and grouped with no backend.
      // The connect panel stays one click away for full analysis.
      document.getElementById("up-local-note").classList.remove("hidden");
      document.getElementById("up-submit").textContent = "Add to library";
    }
    modal.classList.remove("hidden");
  };
  const close = () => modal.classList.add("hidden");
  document.getElementById("upload-btn").addEventListener("click", open);
  document.getElementById("up-show-connect").addEventListener("click", showConnectPanel);
  document.getElementById("up-connect-back").addEventListener("click", () => {
    document.getElementById("up-offline").classList.add("hidden");
    document.getElementById("up-form").classList.remove("hidden");
  });
  document.getElementById("up-connect").addEventListener("click", () => {
    const raw = document
      .getElementById("up-api-url")
      .value.trim()
      .replace(/\/+$/, "");
    if (!/^https?:\/\/.+/i.test(raw)) {
      toast("Enter the backend address, starting with http:// or https://");
      return;
    }
    try {
      window.localStorage.setItem("pythagora_api_base", raw);
    } catch {
      toast("This browser blocked saving the backend address.");
      return;
    }
    // Reload: boot re-resolves the API base, health-checks it, and goes
    // LIVE if it answers (demo returns if it never does).
    window.location.reload();
  });
  document.getElementById("up-cancel").addEventListener("click", close);
  document.getElementById("up-close").addEventListener("click", close);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });
  wireDropzone("up-audio-dz", "up-audio", handleAudioFiles);
  wireDropzone("up-art-dz", "up-artwork", (files) => handleArtFile(files[0]));
  document.getElementById("up-audio-clear").addEventListener("click", (e) => {
    e.stopPropagation();
    clearAudio();
  });
  document.getElementById("up-art-clear").addEventListener("click", (e) => {
    e.stopPropagation();
    clearArt();
  });
  document.getElementById("up-title").addEventListener("input", syncPreview);
  document.getElementById("up-artist").addEventListener("input", syncPreview);
  document.getElementById("up-submit").addEventListener("click", submitUpload);
  document.getElementById("up-back").addEventListener("click", () => {
    document.getElementById("up-progress").classList.add("hidden");
    document.getElementById("up-form").classList.remove("hidden");
    document.getElementById("up-back").classList.add("hidden");
    const btn = document.getElementById("up-submit");
    btn.disabled = false;
    btn.textContent = apiLive ? "Upload & analyze" : "Add to library";
    if (!apiLive) document.getElementById("up-local-note").classList.remove("hidden");
  });
  syncPreview();
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function probeDuration(file) {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return 0;
    const ac = new AC();
    const buf = await ac.decodeAudioData(await file.arrayBuffer());
    const duration = buf.duration || 0;
    ac.close();
    return duration;
  } catch {
    return 0;
  }
}

/* Local ingest: no backend, no analysis — the track is saved in the
   browser (audio in IndexedDB, metadata in localStorage) so it can be
   organized into projects immediately and survives reloads. */
async function submitLocalUpload(files, single, artist, formTitle) {
  const btn = document.getElementById("up-submit");
  btn.disabled = true;
  btn.textContent = "Adding…";
  document.getElementById("up-form").classList.add("hidden");
  document.getElementById("up-offline").classList.add("hidden");
  document.getElementById("up-progress").classList.remove("hidden");
  document.getElementById("up-prog-title").textContent =
    files.length > 1
      ? `Adding ${files.length} tracks locally…`
      : `Adding "${single && formTitle ? formTitle : titleFromFilename(files[0].name)}" locally`;
  document.getElementById("up-prog-sub").textContent =
    "Saved in this browser — group them from the library or the track page. Analysis needs the backend.";
  renderFocusedStages(null);

  let artworkData = null;
  if (single && upState.artFile && upState.artFile.size <= 400 * 1024) {
    try {
      artworkData = await fileToDataUrl(upState.artFile);
    } catch { /* artwork is optional */ }
  }

  const entries = [];
  let storageWarning = false;
  for (const file of files) {
    const entry = {
      file,
      title: single && formTitle ? formTitle : titleFromFilename(file.name),
      trackId: null,
      jobId: null,
      job: null,
      error: null,
    };
    entries.push(entry);
    renderQueue(entries);
    try {
      const id = `local-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
      const duration = await probeDuration(file);
      const stored = await idbPutAudio(id, file);
      state.tracks.push({
        id,
        title: entry.title,
        artist,
        original_file_path: null,
        artwork_path: null,
        artwork_data: artworkData,
        project_id: null,
        notes: "",
        created_at: new Date().toISOString(),
        local: true,
        duration,
        audio_stored: stored,
      });
      entry.trackId = id;
      entry.job = { state: "done", stage: "Saved locally", progress: 100, stages: [] };
      if (!stored) storageWarning = true;
    } catch (err) {
      entry.error = err.message;
    }
    renderQueue(entries);
  }
  persistLocalState();
  const doneCount = entries.filter((e) => e.trackId).length;
  document.getElementById("up-prog-sub").textContent =
    `${doneCount} of ${entries.length} track${entries.length === 1 ? "" : "s"} saved locally. ` +
    "Group them from the library or the track page — analysis needs the backend." +
    (storageWarning
      ? " Note: this browser refused audio storage, so added tracks won't survive a reload or play back."
      : "");
  document.getElementById("up-back").classList.remove("hidden");
  btn.disabled = false;
  btn.textContent = "Add to library";
  await renderLibrary();
  if (entries.length === 1 && doneCount === 1) {
    await sleep(900);
    document.getElementById("upload-modal").classList.add("hidden");
    openDetail(entries[0].trackId);
  }
}

async function submitUpload() {
  if (!upState.audioFiles.length) {
    toast("Drop an audio file first.");
    return;
  }
  const files = [...upState.audioFiles];
  const single = files.length === 1;
  const artist = document.getElementById("up-artist").value.trim() || "Unknown Artist";
  const formTitle = document.getElementById("up-title").value.trim();

  if (!apiLive) {
    await submitLocalUpload(files, single, artist, formTitle);
    return;
  }

  const btn = document.getElementById("up-submit");
  btn.disabled = true;
  btn.textContent = "Uploading…";

  document.getElementById("up-form").classList.add("hidden");
  document.getElementById("up-progress").classList.remove("hidden");
  document.getElementById("up-prog-title").textContent =
    files.length > 1 ? `Analyzing ${files.length} tracks…` : `Analyzing “${formTitle || titleFromFilename(files[0].name)}”`;
  document.getElementById("up-prog-sub").textContent =
    "Deterministic measurements, in order. Nothing is guessed.";
  renderFocusedStages(null);

  const pill = document.getElementById("analyzing-pill");
  const pillLabel = document.getElementById("analyzing-label");
  pill.classList.remove("hidden");

  // Upload sequentially: each accepted file becomes a queued server job;
  // a rejected file (bad content, oversize) fails its own card only.
  const entries = [];
  for (const file of files) {
    const entry = {
      file,
      title: single && formTitle ? formTitle : titleFromFilename(file.name),
      trackId: null,
      jobId: null,
      job: null,
      error: null,
    };
    entries.push(entry);
    renderQueue(entries);
    const fd = new FormData();
    fd.append("title", entry.title);
    fd.append("artist", artist);
    fd.append("audio", file);
    if (single && upState.artFile) fd.append("artwork", upState.artFile);
    try {
      const res = await fetch(API_BASE + "/tracks/upload", {
        method: "POST",
        body: fd,
        headers: apiHeaders(),
      });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const body = await res.json();
          if (body && body.detail) detail = body.detail;
        } catch { /* keep HTTP status */ }
        throw new Error(detail);
      }
      const { track, job_id } = await res.json();
      entry.trackId = track.id;
      entry.jobId = job_id;
      entry.title = track.title;
    } catch (err) {
      entry.error = err.message;
    }
    renderQueue(entries);
  }

  await pollBatch(entries, pill, pillLabel);
}

async function pollBatch(entries, pill, pillLabel) {
  const active = () => entries.filter((e) => e.jobId && (!e.job || !["done", "failed"].includes(e.job.state)));
  while (true) {
    const pending = active();
    if (!pending.length) break;
    await Promise.all(
      pending.map(async (entry) => {
        try {
          entry.job = await apiGet(`/tracks/jobs/${entry.jobId}`);
          entry.pollFailures = 0;
        } catch {
          /* Transient failures keep polling — but not forever. If the
             job record is gone (the server restarted mid-analysis,
             which drops its in-memory job registry), every poll 404s
             and the card would spin eternally. After a few consecutive
             failures, fail the card with an explanation instead. */
          entry.pollFailures = (entry.pollFailures || 0) + 1;
          if (entry.pollFailures >= 5) {
            entry.job = {
              state: "failed",
              error:
                "Lost contact with this analysis job — the server may " +
                "have restarted. Re-upload to try again.",
              progress: entry.job ? entry.job.progress || 0 : 0,
              stages: entry.job ? entry.job.stages || [] : [],
            };
          }
        }
      })
    );
    renderQueue(entries);
    const focused = pending.find((e) => e.job && e.job.state === "running") || pending[0];
    if (focused && focused.job) {
      renderFocusedStages(focused.job);
      pillLabel.textContent =
        entries.length > 1
          ? `Analyzing ${entries.filter((e) => e.job && e.job.state === "done").length}/${entries.length} tracks…`
          : focused.job.stage || "Analyzing…";
    }
    await sleep(1200);
  }
  pill.classList.add("hidden");
  renderQueue(entries);
  const lastJob = [...entries].reverse().find((e) => e.job);
  if (lastJob) renderFocusedStages(lastJob.job);
  const doneCount = entries.filter((e) => e.job && e.job.state === "done").length;
  const sub = document.getElementById("up-prog-sub");
  if (doneCount) {
    sub.textContent = `${doneCount} of ${entries.length} track${entries.length === 1 ? "" : "s"} measured and locked.`;
    state.tracks = await loadTracks();
    state.projects = await loadProjects();
    await renderLibrary();
  } else {
    sub.innerHTML = `<span class="pipe-error">Nothing completed.</span> Check the per-file errors above.`;
  }
  document.getElementById("up-back").classList.remove("hidden");
  const btn = document.getElementById("up-submit");
  btn.disabled = false;
  btn.textContent = "Upload & analyze";
  // Single-track flow keeps its old habit: jump straight into the track.
  if (entries.length === 1 && doneCount === 1) {
    await sleep(900);
    document.getElementById("upload-modal").classList.add("hidden");
    openDetail(entries[0].trackId);
  }
}
/* ------------------------------------------------------------------ */
/* Boot                                                              */
/* ------------------------------------------------------------------ */
document.getElementById("back-btn").addEventListener("click", () => {
  showView("library");
  renderLibrary();
});
document.querySelectorAll(".nav-link").forEach((b) => {
  b.addEventListener("click", () => {
    if (b.dataset.nav === "library") showView("library");
    else openDetail(state.lastTrackId || DEMO_TRACK.id);
  });
});

window.addEventListener(
  "scroll",
  () => {
    document.getElementById("topbar").classList.toggle("scrolled", window.scrollY > 40);
  },
  { passive: true }
);

(async function init() {
  await checkApi();
  wireUpload();
  wireCompareModal();
  state.tracks = await loadTracks();
  state.projects = await loadProjects();
  await renderLibrary();
})();
