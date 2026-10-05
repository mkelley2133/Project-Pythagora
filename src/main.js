import { SynthEngine } from "./synth.js";

const API_BASE = String(window.PYTHAGORAS_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/, "");

/* ------------------------------------------------------------------ */
/* Demo fallback data (mirrors the backend demo bundle) so the UI      */
/* works even when the API isn't running.                             */
/* ------------------------------------------------------------------ */
const DEMO_TRACK = {
  id: "demo-track-001",
  title: "Sample Signal",
  artist: "Pythagoras",
  original_file_path: "/storage/demo.mp3",
  vocal_stem_path: "/storage/demo_vocals.wav",
  instrumental_stem_path: "/storage/demo_inst.wav",
  created_at: "2026-01-01T00:00:00Z",
};

const DEMO_TELEMETRY = {
  bpm: 92.4,
  duration: 83.12,
  musical_key: "D",
  mode: "minor",
  sub_genres: [{ name: "dream pop", confidence: 0.82 }],
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
  const h1 = hashStr(id) % 360;
  return { h1, h2: (h1 + 50) % 360 };
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

async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function checkApi() {
  const el = document.getElementById("api-status");
  try {
    await apiGet("/health");
    apiLive = true;
    el.classList.add("ok");
    el.querySelector(".label").textContent = "api live";
  } catch {
    apiLive = false;
    el.querySelector(".label").textContent = "demo mode";
  }
  document.getElementById("footer-mode").textContent = apiLive ? "live api" : "demo telemetry";
}

async function loadTracks() {
  if (apiLive) {
    try {
      return await apiGet("/tracks");
    } catch { /* fall through to demo */ }
  }
  return [DEMO_TRACK];
}

async function loadAnalysis(trackId) {
  if (apiLive) {
    try {
      const rec = await apiGet(`/tracks/${trackId}/analysis`);
      return rec.telemetry_json || null;
    } catch { /* fall through */ }
  }
  return trackId === DEMO_TRACK.id ? DEMO_TELEMETRY : null;
}

/* ------------------------------------------------------------------ */
/* State                                                              */
/* ------------------------------------------------------------------ */
const state = {
  tracks: [],
  telemetry: {}, // trackId -> TelemetryBundle
  detail: null, // { track, telemetry, engine, peaks, raf, activeLyric }
  lastTrackId: null,
};

/* ------------------------------------------------------------------ */
/* Library view                                                       */
/* ------------------------------------------------------------------ */
function trackCard(track) {
  const { h1, h2 } = coverHues(track.id);
  const tel = state.telemetry[track.id];
  const art = artworkUrl(track);
  const card = document.createElement("div");
  card.className = "track-card";
  card.innerHTML = `
    <div class="cover" style="--h1:${h1};--h2:${h2}">
      ${art ? `<img class="cover-img" src="${art}" alt="" loading="lazy" onerror="this.remove()">` : ""}
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
      </div>
    </div>`;
  card.addEventListener("click", () => openDetail(track.id));
  return card;
}

function moodOf(telemetry) {
  const entries = Object.entries(telemetry.emotional_profile || {});
  if (!entries.length) return "unclassified";
  return entries.sort((a, b) => b[1] - a[1])[0][0];
}

function artworkUrl(track) {
  // Cover art only exists when the API is live; otherwise the gradient cover shows.
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
      ${art ? `<img class="billboard-art" src="${art}" alt="" onerror="this.remove()">` : ""}
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
  document.getElementById("bb-play").addEventListener("click", () => openDetail(feat.id));
  document.getElementById("bb-info").addEventListener("click", () => openDetail(feat.id));
}

async function renderLibrary() {
  const carousels = document.getElementById("carousels");
  carousels.innerHTML = "";

  // Load telemetry for every track (parallel) so rows can classify by mood/energy.
  state.telemetry = {};
  let analyses = 0;
  await Promise.all(
    state.tracks.map(async (t) => {
      const tel = await loadAnalysis(t.id);
      if (tel) {
        state.telemetry[t.id] = tel;
        analyses++;
      }
    })
  );

  renderBillboard();

  const energy = (t) => energyOf(state.telemetry[t.id]);
  const recent = [...state.tracks].sort(
    (a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0)
  );

  carousels.appendChild(
    makeRow("Your library", `${state.tracks.length} track${state.tracks.length === 1 ? "" : "s"}`, state.tracks)
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
      const wrap = document.createElement("div");
      wrap.style.minWidth = "220px";
      wrap.innerHTML = `<div class="chip" style="margin-bottom:8px;display:inline-block">${esc(m)}</div>`;
      wrap.appendChild(trackCard(byMood[m][0]));
      car.appendChild(wrap);
      for (const t of byMood[m].slice(1)) {
        const w2 = document.createElement("div");
        w2.style.minWidth = "220px";
        w2.appendChild(trackCard(t));
        car.appendChild(w2);
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

const SEG_COLORS = ["#e5484d", "#9b8cff", "#4fd1a5", "#f5b544", "#5aa9ff"];

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
    ctx.fillStyle = x <= playedX ? "rgba(245,181,68,0.95)" : "rgba(154,148,168,0.5)";
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
}

async function openDetail(trackId) {
  stopDetail();
  state.lastTrackId = trackId;
  const track = state.tracks.find((t) => t.id === trackId) || DEMO_TRACK;
  const telemetry = await loadAnalysis(trackId);
  // Prefer the real uploaded recording when the API can serve it;
  // otherwise fall back to the synthesized stand-in.
  let audioUrl = null;
  if (apiLive && telemetry) {
    try {
      const r = await fetch(`${API_BASE}/tracks/${trackId}/audio`, { method: "HEAD" });
      if (r.ok) audioUrl = `${API_BASE}/tracks/${trackId}/audio`;
    } catch {
      /* fall through to synth */
    }
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
  }
  get playing() {
    return !this.el.paused;
  }
  play() {
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
    /* no-op: the recording is the full mix; stems come with separation */
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

function renderDetail(track, telemetry, audioUrl) {
  const host = document.getElementById("detail-content");
  const { h1, h2 } = coverHues(track.id);

  if (!telemetry) {
    const art0 = artworkUrl(track);
    host.innerHTML = `
      <div class="detail-head">
        <div class="detail-cover" style="--h1:${h1};--h2:${h2}">${art0 ? `<img class="cover-img" src="${art0}" alt="" onerror="this.remove()">` : ""}<span class="cover-glyph">♪</span></div>
        <div><h1>${esc(track.title)}</h1><p class="artist">${esc(track.artist)}</p></div>
      </div>
      <div class="panel"><h2>Analysis pending</h2>
        <p class="panel-sub">This track hasn't been analyzed yet.</p>
        <p style="color:var(--muted);font-size:14px">Run the analysis pipeline to generate its telemetry bundle — BPM, key, segments, chords, rhyme map, and essay will appear here.</p>
      </div>`;
    return;
  }

  const moods = Object.entries(telemetry.emotional_profile || {}).sort((a, b) => b[1] - a[1]);
  const genres = (telemetry.sub_genres || [])
    .map((g) => `<span class="chip">${esc(g.name)} · ${Math.round(g.confidence * 100)}%</span>`)
    .join("");

  host.innerHTML = `
    <div class="detail-head">
      <div class="detail-cover" style="--h1:${h1};--h2:${h2}">${artworkUrl(track) ? `<img class="cover-img" src="${artworkUrl(track)}" alt="" onerror="this.remove()">` : ""}<span class="cover-glyph">♪</span></div>
      <div>
        <h1>${esc(track.title)}</h1>
        <p class="artist">${esc(track.artist)}</p>
        <div class="chip-row">
          <span class="chip hot">${telemetry.bpm} BPM</span>
          <span class="chip">${esc(telemetry.musical_key)} ${esc(telemetry.mode)}</span>
          <span class="chip">${esc(telemetry.rhyme_scheme_summary)} rhyme</span>
          ${genres}
        </div>
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
        : "Audio is synthesized live from the track's measured chords &amp; tempo — a stand-in until the ingestion pipeline ships real stems."} Click the waveform or any lyric line to seek.</p>
    </div>

    <div class="detail-grid">
      <div>
        <div class="panel">
          <h2>Lyrics · time-synced</h2>
          <p class="panel-sub">Click a line to seek · rhyme tags follow the ${esc(telemetry.rhyme_scheme_summary)} scheme</p>
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
          </div>
          <div id="moods"></div>
        </div>
      </div>
    </div>`;

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
  lyricsEl.innerHTML = lines
    .map(
      (l, i) => `
      <div class="lyric-line" data-i="${i}">
        <span class="ts">${fmtTime(l.start)} → ${fmtTime(l.end)}</span>
        <span class="txt">${esc(l.text)}</span>
        <span class="rhyme">${esc(l.rhyme)}</span>
      </div>`
    )
    .join("");
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
/* Upload + job tracking — drag & drop ingest with live previews      */
/* ------------------------------------------------------------------ */
const upState = { audioFile: null, artFile: null, artUrl: null };

const PIPE_STAGES = [
  "Loading audio",
  "Measuring tempo & key",
  "Recognizing chords",
  "Mapping sections & texture",
  "Rendering waveform",
  "Vocal forensics",
  "Transcribing lyrics",
  "Finalizing",
];

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
  grad.addColorStop(0, "#e5484d");
  grad.addColorStop(1, "#9b8cff");
  ctx.fillStyle = grad;
  peaks.forEach((p, i) => {
    const ph = Math.max(2, p * h);
    ctx.fillRect(i * bw, (h - ph) / 2, Math.max(1, bw - 1), ph);
  });
}

async function handleAudioFile(file) {
  if (!/\.(mp3|wav|ogg|flac|m4a|aac)$/i.test(file.name)) {
    toast("Unsupported audio type.");
    return;
  }
  upState.audioFile = file;
  document.getElementById("up-audio-idle").classList.add("hidden");
  document.getElementById("up-audio-file").classList.remove("hidden");
  document.getElementById("up-audio-name").textContent = file.name;
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
    if (!document.getElementById("up-title").value.trim()) {
      document.getElementById("up-title").value = file.name
        .replace(/\.[^.]+$/, "")
        .replace(/[_-]+/g, " ");
      syncPreview();
    }
    ac.close();
  } catch {
    document.getElementById("up-audio-sub").textContent = `${(file.size / 1048576).toFixed(1)} MB`;
  }
}

function handleArtFile(file) {
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
  upState.audioFile = null;
  document.getElementById("up-audio").value = "";
  document.getElementById("up-audio-idle").classList.remove("hidden");
  document.getElementById("up-audio-file").classList.add("hidden");
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
  document.getElementById("up-progress").classList.add("hidden");
  document.getElementById("up-back").classList.add("hidden");
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
    if (input.files[0]) handler(input.files[0]);
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
    const f = e.dataTransfer.files[0];
    if (f) handler(f);
  });
}

function renderPipeStages() {
  document.getElementById("up-stages").innerHTML = PIPE_STAGES.map(
    (s) => `<li><span class="st-ic"></span><span>${s}</span></li>`
  ).join("");
}

function markPipeStages(stageLabel, allDone) {
  const idx = allDone
    ? PIPE_STAGES.length
    : PIPE_STAGES.findIndex((s) => stageLabel.startsWith(s));
  document.querySelectorAll("#up-stages li").forEach((li, i) => {
    li.classList.toggle("done", i < idx);
    li.classList.toggle("active", i === idx);
  });
}

function wireUpload() {
  const modal = document.getElementById("upload-modal");
  const open = () => {
    if (!apiLive) {
      toast("Upload needs the API — run it locally to ingest tracks.");
      return;
    }
    resetUploadModal();
    modal.classList.remove("hidden");
  };
  const close = () => modal.classList.add("hidden");
  document.getElementById("upload-btn").addEventListener("click", open);
  document.getElementById("up-cancel").addEventListener("click", close);
  document.getElementById("up-close").addEventListener("click", close);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });
  wireDropzone("up-audio-dz", "up-audio", handleAudioFile);
  wireDropzone("up-art-dz", "up-artwork", handleArtFile);
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
    btn.textContent = "Upload & analyze";
  });
  syncPreview();
}

async function submitUpload() {
  if (!upState.audioFile) {
    toast("Drop an audio file first.");
    return;
  }
  const fd = new FormData();
  fd.append("title", document.getElementById("up-title").value.trim() || upState.audioFile.name);
  fd.append("artist", document.getElementById("up-artist").value.trim() || "Unknown Artist");
  fd.append("audio", upState.audioFile);
  if (upState.artFile) fd.append("artwork", upState.artFile);

  const btn = document.getElementById("up-submit");
  btn.disabled = true;
  btn.textContent = "Uploading…";
  try {
    const res = await fetch(API_BASE + "/tracks/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { track, job_id } = await res.json();
    pollJob(job_id, track);
  } catch (err) {
    toast("Upload failed: " + err.message);
    btn.disabled = false;
    btn.textContent = "Upload & analyze";
  }
}

async function pollJob(jobId, track) {
  document.getElementById("up-form").classList.add("hidden");
  document.getElementById("up-progress").classList.remove("hidden");
  renderPipeStages();
  const fill = document.getElementById("up-prog-fill");
  const sub = document.getElementById("up-prog-sub");
  const pill = document.getElementById("analyzing-pill");
  const pillLabel = document.getElementById("analyzing-label");
  // Unmistakable global indicator: visible from any view, even if the modal closes.
  pill.classList.remove("hidden");

  while (true) {
    try {
      const job = await apiGet(`/tracks/jobs/${jobId}`);
      fill.style.width = `${job.progress || 0}%`;
      if (job.state === "done") {
        markPipeStages("", true);
        sub.textContent = "Measurements locked. Opening your track…";
        pill.classList.add("hidden");
        state.tracks = await loadTracks();
        await renderLibrary();
        await sleep(900);
        document.getElementById("upload-modal").classList.add("hidden");
        openDetail(track.id);
        return;
      }
      if (job.state === "failed") {
        const err = job.error || "unknown error";
        const decodeIssue = /format|decod|codec|ffmpeg|audioread|not recognised/i.test(err);
        sub.innerHTML =
          `<span class="pipe-error">Analysis failed.</span> ${esc(err)}` +
          (decodeIssue
            ? `<br><br>WAV files always decode. For MP3/M4A the server needs <code>ffmpeg</code> installed (<code>sudo apt-get install -y ffmpeg</code>).`
            : "");
        document.getElementById("up-back").classList.remove("hidden");
        pill.classList.add("hidden");
        return;
      }
      sub.textContent = job.stage || job.state;
      pillLabel.textContent = job.stage || "Analyzing…";
      markPipeStages(job.stage || "", false);
    } catch {
      /* transient — keep polling */
    }
    await sleep(1500);
  }
}
/* ------------------------------------------------------------------ */
/* Boot                                                              */
/* ------------------------------------------------------------------ */
document.getElementById("back-btn").addEventListener("click", () => showView("library"));
document.querySelectorAll(".nav-link").forEach((b) => {
  b.addEventListener("click", () => {
    if (b.dataset.nav === "library") showView("library");
    else openDetail(state.lastTrackId || DEMO_TRACK.id);
  });
});

(async function init() {
  await checkApi();
  wireUpload();
  state.tracks = await loadTracks();
  await renderLibrary();
})();
