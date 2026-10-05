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
  detail: null, // { track, telemetry, engine, peaks, raf, activeLyric }
  lastTrackId: null,
};

/* ------------------------------------------------------------------ */
/* Library view                                                       */
/* ------------------------------------------------------------------ */
function trackCard(track) {
  const { h1, h2 } = coverHues(track.id);
  const card = document.createElement("div");
  card.className = "track-card";
  card.innerHTML = `
    <div class="cover" style="--h1:${h1};--h2:${h2}">
      <span class="cover-glyph">♪</span>
      <div class="play-hover"><button class="pp" aria-label="Open">▶</button></div>
    </div>
    <div class="track-meta">
      <h3>${esc(track.title)}</h3>
      <p class="artist">${esc(track.artist)}</p>
      <div class="chip-row">
        <span class="chip hot">${esc(DEMO_TELEMETRY.bpm)} BPM</span>
        <span class="chip">${esc(DEMO_TELEMETRY.musical_key)} ${esc(DEMO_TELEMETRY.mode)}</span>
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

async function renderLibrary() {
  const carousels = document.getElementById("carousels");
  carousels.innerHTML = "";
  document.getElementById("stat-tracks").textContent = state.tracks.length;

  let analyses = 0;
  try {
    if (state.tracks.some((t) => t.id === DEMO_TRACK.id)) {
      const tel = await loadAnalysis(DEMO_TRACK.id);
      if (tel) analyses++;
      state.tracks._telemetry = { [DEMO_TRACK.id]: tel };
    }
  } catch { /* ignore */ }
  document.getElementById("stat-analyses").textContent = analyses;

  const libBlock = document.createElement("div");
  libBlock.className = "carousel-block";
  libBlock.innerHTML = `<div class="carousel-head"><h2>Your library</h2><span>${state.tracks.length} track${state.tracks.length === 1 ? "" : "s"}</span></div>`;
  const libCar = document.createElement("div");
  libCar.className = "carousel";
  if (!state.tracks.length) {
    libCar.innerHTML = `<div class="empty-note">No tracks yet. Register one via <code>POST /tracks</code>.</div>`;
  } else {
    for (const t of state.tracks) libCar.appendChild(trackCard(t));
  }
  libBlock.appendChild(libCar);
  carousels.appendChild(libBlock);

  // Browse by mood — groups tracks under their dominant measured emotion.
  const moodBlock = document.createElement("div");
  moodBlock.className = "carousel-block";
  const tel = (state.tracks._telemetry || {})[DEMO_TRACK.id];
  const mood = tel ? moodOf(tel) : "—";
  moodBlock.innerHTML = `<div class="carousel-head"><h2>Browse by mood</h2><span>dominant emotion per track</span></div>`;
  const moodCar = document.createElement("div");
  moodCar.className = "carousel";
  for (const t of state.tracks) {
    const wrap = document.createElement("div");
    wrap.style.minWidth = "220px";
    wrap.innerHTML = `<div class="chip" style="margin-bottom:8px;display:inline-block">${esc(mood)}</div>`;
    wrap.appendChild(trackCard(t));
    moodCar.appendChild(wrap);
  }
  moodBlock.appendChild(moodCar);
  carousels.appendChild(moodBlock);

  drawHeroWave();
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

function drawHeroWave() {
  const canvas = document.getElementById("hero-wave");
  if (!canvas) return;
  const peaks = makePeaks("hero", 220);
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 520;
  const h = 220;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const bw = w / peaks.length;
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, "#e5484d");
  grad.addColorStop(0.5, "#9b8cff");
  grad.addColorStop(1, "#4fd1a5");
  ctx.fillStyle = grad;
  peaks.forEach((p, i) => {
    const ph = p * h * 0.8;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(i * bw, (h - ph) / 2, Math.max(1, bw - 1), ph);
  });
  ctx.globalAlpha = 1;
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
  showView("detail");
  renderDetail(track, telemetry);
}

function renderDetail(track, telemetry) {
  const host = document.getElementById("detail-content");
  const { h1, h2 } = coverHues(track.id);

  if (!telemetry) {
    host.innerHTML = `
      <div class="detail-head">
        <div class="detail-cover" style="--h1:${h1};--h2:${h2}"><span class="cover-glyph">♪</span></div>
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
      <div class="detail-cover" style="--h1:${h1};--h2:${h2}"><span class="cover-glyph">♪</span></div>
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
        <div class="deck-toggle" role="tablist" aria-label="Deck">
          <button id="deck-a" class="active">Deck A · Full mix</button>
          <button id="deck-b">Deck B · Stripped</button>
        </div>
      </div>
      <div id="wave-wrap"><canvas id="wave"></canvas></div>
      <div class="seg-legend" id="seg-legend"></div>
      <div class="chord-strip" id="chord-strip"></div>
      <p class="player-note">Audio is synthesized live from the track's measured chords &amp; tempo — a stand-in until the ingestion pipeline ships real stems. Click the waveform or any lyric line to seek.</p>
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

  // Engine
  const engine = new SynthEngine({
    bpm: telemetry.bpm,
    chords: telemetry.chords,
    duration: telemetry.duration,
  });
  const barLen = engine.barLen;
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
        engine.play();
        playBtn.textContent = "⏸";
      }
    } catch (err) {
      toast("Audio couldn't start: " + err.message);
    }
  });

  host.querySelector("#deck-a").addEventListener("click", (e) => setDeck("A", e));
  host.querySelector("#deck-b").addEventListener("click", (e) => setDeck("B", e));
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
/* Boot                                                              */
/* ------------------------------------------------------------------ */
document.getElementById("back-btn").addEventListener("click", () => showView("library"));
document.querySelectorAll(".nav-link").forEach((b) => {
  b.addEventListener("click", () => {
    if (b.dataset.nav === "library") showView("library");
    else openDetail(state.lastTrackId || DEMO_TRACK.id);
  });
});
window.addEventListener("resize", () => {
  if (state.detail) drawHeroWave();
});

(async function init() {
  await checkApi();
  state.tracks = await loadTracks();
  await renderLibrary();
})();
