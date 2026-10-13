/* Built-in parametric EQ (Phase I) — Web Audio BiquadFilters.
 *
 * Six bands: low shelf, four peaking, high shelf. Each band has frequency
 * and gain (±15 dB); peaking bands also have Q. Playback only — the EQ
 * colors what you hear in the dashboard player, never the stored audio
 * or its analysis.
 *
 * The pure RBJ-cookbook response math below mirrors what the browser's
 * BiquadFilterNodes compute; it powers the drawn curve before an audio
 * context exists and is unit-tested in Node (tests/test_frontend.mjs).
 * When a live chain exists, the UI prefers the nodes' own
 * getFrequencyResponse (EqChain.getResponseDb).
 */

export const EQ_STORAGE_KEY = "pythagora_eq";

export const EQ_BANDS = [
  { id: "ls", label: "Low shelf", type: "lowshelf", f: 120, fMin: 40, fMax: 400, q: null },
  { id: "p1", label: "Low-mid", type: "peaking", f: 300, fMin: 100, fMax: 900, q: 1.0 },
  { id: "p2", label: "Mid", type: "peaking", f: 900, fMin: 300, fMax: 2400, q: 1.0 },
  { id: "p3", label: "Presence", type: "peaking", f: 2500, fMin: 900, fMax: 6000, q: 1.1 },
  { id: "p4", label: "Upper", type: "peaking", f: 6000, fMin: 2400, fMax: 12000, q: 1.0 },
  { id: "hs", label: "High shelf", type: "highshelf", f: 10000, fMin: 5000, fMax: 16000, q: null },
];

export const GAIN_MIN = -15;
export const GAIN_MAX = 15;

function band(f, g, q) {
  const out = { g };
  if (f !== undefined) out.f = f;
  if (q !== undefined) out.q = q;
  return out;
}

/* Presets are partial per-band settings merged over the band defaults. */
export const EQ_PRESETS = {
  Flat: [{}, {}, {}, {}, {}, {}],
  "Vocal Presence": [
    band(undefined, 0),
    band(300, -1.5),
    band(900, 1.0),
    band(2800, 3.5),
    band(5500, 2.0),
    band(undefined, 1.0),
  ],
  "Bass Trim": [
    band(120, -6),
    band(250, -2),
    band(undefined, 0),
    band(undefined, 0),
    band(undefined, 0),
    band(undefined, 0),
  ],
  "Air / Top Lift": [
    band(undefined, 0),
    band(undefined, 0),
    band(undefined, 0),
    band(undefined, 0),
    band(6000, 2.5),
    band(10000, 5),
  ],
  "Low-End Focus": [
    band(110, 5),
    band(300, 1.5),
    band(undefined, 0),
    band(2500, -2),
    band(undefined, 0),
    band(undefined, -2),
  ],
  "De-Mud": [
    band(undefined, -1.5),
    band(300, -4.5, 1.2),
    band(500, -2.5),
    band(undefined, 0),
    band(undefined, 0),
    band(undefined, 0),
  ],
};

export function defaultBandState() {
  return EQ_BANDS.map((b) => ({ f: b.f, g: 0, q: b.q === null ? 0.9 : b.q }));
}

export function presetBandState(name) {
  const preset = EQ_PRESETS[name] || EQ_PRESETS.Flat;
  return EQ_BANDS.map((b, i) => {
    const base = { f: b.f, g: 0, q: b.q === null ? 0.9 : b.q };
    return { ...base, ...preset[i] };
  });
}

export function defaultEqSettings() {
  return { bypass: false, preset: "Flat", bands: defaultBandState() };
}

export function loadEqSettings(storage) {
  const store = storage || (typeof localStorage !== "undefined" ? localStorage : null);
  if (!store) return defaultEqSettings();
  try {
    const raw = store.getItem(EQ_STORAGE_KEY);
    if (!raw) return defaultEqSettings();
    const parsed = JSON.parse(raw);
    const settings = defaultEqSettings();
    if (typeof parsed.bypass === "boolean") settings.bypass = parsed.bypass;
    if (typeof parsed.preset === "string") settings.preset = parsed.preset;
    if (Array.isArray(parsed.bands) && parsed.bands.length === EQ_BANDS.length) {
      settings.bands = parsed.bands.map((b, i) => ({
        f: clampNum(b.f, EQ_BANDS[i].fMin, EQ_BANDS[i].fMax, EQ_BANDS[i].f),
        g: clampNum(b.g, GAIN_MIN, GAIN_MAX, 0),
        q: clampNum(b.q, 0.3, 6, 1),
      }));
    }
    return settings;
  } catch {
    return defaultEqSettings();
  }
}

export function saveEqSettings(settings, storage) {
  const store = storage || (typeof localStorage !== "undefined" ? localStorage : null);
  if (!store) return;
  try {
    store.setItem(EQ_STORAGE_KEY, JSON.stringify(settings));
  } catch { /* storage unavailable — session-only EQ */ }
}

function clampNum(v, lo, hi, fallback) {
  const n = +v;
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
}

/* ------------------------------------------------ RBJ response math -- */

function biquadCoeffs(def, state, sampleRate) {
  const f = state.f;
  const gainDb = state.g;
  const A = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * f) / sampleRate;
  const cosw = Math.cos(w0);
  const sinw = Math.sin(w0);
  let b0, b1, b2, a0, a1, a2;
  if (def.type === "peaking") {
    const alpha = sinw / (2 * (state.q || 1));
    b0 = 1 + alpha * A;
    b1 = -2 * cosw;
    b2 = 1 - alpha * A;
    a0 = 1 + alpha / A;
    a1 = -2 * cosw;
    a2 = 1 - alpha / A;
  } else {
    // Shelves: RBJ cookbook with slope S = 1 (alpha = sin(w0)/2 * sqrt(2)).
    const alpha = (sinw / 2) * Math.SQRT2;
    const twoSqrtAalpha = 2 * Math.sqrt(A) * alpha;
    if (def.type === "lowshelf") {
      b0 = A * ((A + 1) - (A - 1) * cosw + twoSqrtAalpha);
      b1 = 2 * A * ((A - 1) - (A + 1) * cosw);
      b2 = A * ((A + 1) - (A - 1) * cosw - twoSqrtAalpha);
      a0 = (A + 1) + (A - 1) * cosw + twoSqrtAalpha;
      a1 = -2 * ((A - 1) + (A + 1) * cosw);
      a2 = (A + 1) + (A - 1) * cosw - twoSqrtAalpha;
    } else {
      b0 = A * ((A + 1) + (A - 1) * cosw + twoSqrtAalpha);
      b1 = -2 * A * ((A - 1) + (A + 1) * cosw);
      b2 = A * ((A + 1) + (A - 1) * cosw - twoSqrtAalpha);
      a0 = (A + 1) - (A - 1) * cosw + twoSqrtAalpha;
      a1 = 2 * ((A - 1) - (A + 1) * cosw);
      a2 = (A + 1) - (A - 1) * cosw - twoSqrtAalpha;
    }
  }
  return { b: [b0 / a0, b1 / a0, b2 / a0], a: [1, a1 / a0, a2 / a0] };
}

function responseDb(coeffs, freq, sampleRate) {
  const w = (2 * Math.PI * freq) / sampleRate;
  const cosw = Math.cos(w);
  const cos2w = Math.cos(2 * w);
  const [b0, b1, b2] = coeffs.b;
  const [, a1, a2] = coeffs.a;
  const num = b0 * b0 + b1 * b1 + b2 * b2 + 2 * (b0 * b1 + b1 * b2) * cosw + 2 * b0 * b2 * cos2w;
  const den = 1 + a1 * a1 + a2 * a2 + 2 * (a1 + a1 * a2) * cosw + 2 * a2 * cos2w;
  return 10 * Math.log10(Math.max(num / Math.max(den, 1e-12), 1e-12));
}

/** Combined magnitude response (dB) of a band-state array at one frequency. */
export function combinedResponseDb(bandStates, freq, sampleRate = 44100) {
  let db = 0;
  for (let i = 0; i < EQ_BANDS.length; i++) {
    db += responseDb(biquadCoeffs(EQ_BANDS[i], bandStates[i], sampleRate), freq, sampleRate);
  }
  return db;
}

/* ------------------------------------------------------- live chain -- */

export class EqChain {
  /** A chain of BiquadFilters between input and output GainNodes. */
  constructor(ctx, settings) {
    this.ctx = ctx;
    this.input = ctx.createGain();
    this.output = ctx.createGain();
    this.filters = EQ_BANDS.map((def, i) => {
      const node = ctx.createBiquadFilter();
      node.type = def.type;
      this._applyBand(node, def, settings.bands[i]);
      return node;
    });
    this.bypass = false;
    this._wire(false);
  }

  _applyBand(node, def, state) {
    node.frequency.value = state.f;
    node.gain.value = state.g;
    if (def.type === "peaking") node.Q.value = state.q || 1;
  }

  _wire(bypass) {
    this.bypass = bypass;
    try { this.input.disconnect(); } catch { /* not connected yet */ }
    this.filters.forEach((f) => { try { f.disconnect(); } catch { /* noop */ } });
    if (bypass) {
      this.input.connect(this.output);
      return;
    }
    let prev = this.input;
    for (const f of this.filters) {
      prev.connect(f);
      prev = f;
    }
    prev.connect(this.output);
  }

  applySettings(settings) {
    this.filters.forEach((node, i) => this._applyBand(node, EQ_BANDS[i], settings.bands[i]));
    if (settings.bypass !== this.bypass) this._wire(settings.bypass);
  }

  setBand(i, state) {
    this._applyBand(this.filters[i], EQ_BANDS[i], state);
  }

  setBypass(bypass) {
    if (bypass !== this.bypass) this._wire(bypass);
  }

  /** Combined response using the real nodes' getFrequencyResponse. */
  getResponseDb(freqs) {
    const sr = this.ctx.sampleRate;
    const mags = new Float32Array(freqs.length);
    const phases = new Float32Array(freqs.length);
    const total = new Float64Array(freqs.length).fill(1);
    for (const node of this.filters) {
      node.getFrequencyResponse(freqs, mags, phases);
      for (let i = 0; i < freqs.length; i++) total[i] *= mags[i];
    }
    void sr;
    return Array.from(total, (m) => 20 * Math.log10(Math.max(m, 1e-6)));
  }
}
