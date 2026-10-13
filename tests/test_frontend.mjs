/* Frontend unit tests — run with: node tests/test_frontend.mjs
 * (also driven from pytest via tests/test_frontend_js.py).
 *
 * Covers Phase H math (adjust.js), Phase I EQ definitions + curve math
 * (eq.js), and Phase F sorting (sort.js). The live Web Audio behavior of
 * EqChain (real BiquadFilterNodes) can't run in Node — its settings are
 * applied through the same band-state objects tested here; see the wave-2
 * report for the manual verification steps for the audio path.
 */
import assert from "node:assert/strict";

import {
  clampAdjust,
  coupledPlaybackRate,
  effectiveBpm,
  transposeKey,
} from "../frontend/src/adjust.js";
import {
  EQ_BANDS,
  EQ_PRESETS,
  combinedResponseDb,
  defaultBandState,
  loadEqSettings,
  presetBandState,
  saveEqSettings,
} from "../frontend/src/eq.js";
import { sortTracks } from "../frontend/src/sort.js";

/* ------------------------------------------------------------- Phase H */
assert.equal(transposeKey("C", 2), "D");
assert.equal(transposeKey("B", 1), "C");
assert.equal(transposeKey("A", -3), "F#");
assert.equal(transposeKey("C", 12), "C");
assert.equal(transposeKey("C", -1), "B");
assert.equal(effectiveBpm(100, 1.1), 110);
assert.equal(effectiveBpm(92.4, 0.5), 46.2);
assert.equal(coupledPlaybackRate(12, 1), 2);
assert.equal(coupledPlaybackRate(0, 1), 1);
assert.equal(coupledPlaybackRate(-12, 2), 1);
assert.deepEqual(clampAdjust(99, 5), { semitones: 12, speed: 2 });
assert.deepEqual(clampAdjust(-99, 0.1), { semitones: -12, speed: 0.5 });
assert.deepEqual(clampAdjust(3, 1.25), { semitones: 3, speed: 1.25 });

/* ------------------------------------------------------------- Phase I */
assert.equal(EQ_BANDS.length, 6);
assert.deepEqual(
  EQ_BANDS.map((b) => b.type),
  ["lowshelf", "peaking", "peaking", "peaking", "peaking", "highshelf"]
);
for (const name of ["Flat", "Vocal Presence", "Bass Trim", "Air / Top Lift", "Low-End Focus", "De-Mud"]) {
  assert.ok(EQ_PRESETS[name], `preset ${name} exists`);
  assert.equal(EQ_PRESETS[name].length, 6);
}

// Flat preset: unity response across the spectrum.
const flat = presetBandState("Flat");
for (const f of [40, 120, 440, 1000, 4000, 12000]) {
  assert.ok(Math.abs(combinedResponseDb(flat, f, 44100)) < 1e-6, `flat @ ${f} Hz`);
}

// Bass Trim: ~-6 dB deep in the shelf region, ~unity at the top.
const bassTrim = presetBandState("Bass Trim");
const lowResp = combinedResponseDb(bassTrim, 60, 44100);
assert.ok(lowResp < -4.5 && lowResp > -6.6, `bass trim @60Hz = ${lowResp}`);
assert.ok(Math.abs(combinedResponseDb(bassTrim, 12000, 44100)) < 0.3);

// Vocal Presence: a clear lift around 2.8 kHz.
const vocal = presetBandState("Vocal Presence");
const presResp = combinedResponseDb(vocal, 2800, 44100);
assert.ok(presResp > 2.5 && presResp < 5.5, `presence @2.8kHz = ${presResp}`);

// Air / Top Lift: positive at 12 kHz, neutral at 100 Hz.
const air = presetBandState("Air / Top Lift");
assert.ok(combinedResponseDb(air, 12000, 44100) > 4);
assert.ok(Math.abs(combinedResponseDb(air, 100, 44100)) < 0.3);

// De-Mud: cuts at 300 Hz.
const demud = presetBandState("De-Mud");
assert.ok(combinedResponseDb(demud, 300, 44100) < -3.5);

// Settings persistence roundtrip with a fake storage.
const mem = new Map();
const fakeStorage = {
  getItem: (k) => (mem.has(k) ? mem.get(k) : null),
  setItem: (k, v) => mem.set(k, String(v)),
};
const settings = { bypass: true, preset: "Bass Trim", bands: presetBandState("Bass Trim") };
saveEqSettings(settings, fakeStorage);
const loaded = loadEqSettings(fakeStorage);
assert.equal(loaded.bypass, true);
assert.equal(loaded.preset, "Bass Trim");
assert.equal(loaded.bands[0].g, -6);
assert.deepEqual(loadEqSettings({ getItem: () => null, setItem: () => {} }).bands, defaultBandState());

/* ------------------------------------------------------------- Phase F */
const tracks = [
  { id: "t1", title: "Bravo", created_at: "2026-01-03T00:00:00Z" },
  { id: "t2", title: "Alpha", created_at: "2026-01-01T00:00:00Z" },
  { id: "t3", title: "Charlie", created_at: "2026-01-02T00:00:00Z" },
];
const telemetry = {
  t1: { bpm: 120, duration: 200, musical_key: "D", mode: "minor" },
  t2: { bpm: 90, duration: 120, musical_key: "C", mode: "major" },
  // t3 unanalyzed on purpose
};
assert.deepEqual(sortTracks(tracks, telemetry, "name", "asc").map((t) => t.id), ["t2", "t1", "t3"]);
assert.deepEqual(sortTracks(tracks, telemetry, "date", "desc").map((t) => t.id), ["t1", "t3", "t2"]);
assert.deepEqual(sortTracks(tracks, telemetry, "tempo", "asc").map((t) => t.id), ["t2", "t1", "t3"]);
assert.deepEqual(sortTracks(tracks, telemetry, "tempo", "desc").map((t) => t.id), ["t1", "t2", "t3"]);
assert.deepEqual(sortTracks(tracks, telemetry, "key", "asc").map((t) => t.id), ["t2", "t1", "t3"]);
assert.deepEqual(sortTracks(tracks, telemetry, "duration", "desc").map((t) => t.id), ["t1", "t2", "t3"]);

console.log("frontend unit tests: OK");
