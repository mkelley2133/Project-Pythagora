/* Pitch/speed adjustment math (Phase H) — pure functions, no DOM.
 *
 * The server render (backend/adjust.py) applies pitch and speed
 * independently. The live preview on an <audio> element can only resample
 * (coupled), so the preview rate combines both: rate = speed x 2^(st/12).
 * Derived readouts (effective BPM / transposed key) are computed from the
 * original measured values and labeled "derived" in the UI.
 */

export const PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export const SEMITONE_MIN = -12;
export const SEMITONE_MAX = 12;
export const SPEED_MIN = 0.5;
export const SPEED_MAX = 2;

export function clampAdjust(semitones, speed) {
  let st = Number.isFinite(+semitones) ? +semitones : 0;
  let sp = Number.isFinite(+speed) ? +speed : 1;
  st = Math.min(SEMITONE_MAX, Math.max(SEMITONE_MIN, st));
  sp = Math.min(SPEED_MAX, Math.max(SPEED_MIN, sp));
  return { semitones: Math.round(st * 100) / 100, speed: Math.round(sp * 1000) / 1000 };
}

export function transposeKey(key, semitones) {
  const idx = PITCH_NAMES.indexOf(key);
  if (idx < 0) return key;
  const steps = Math.round(+semitones || 0);
  return PITCH_NAMES[(((idx + steps) % 12) + 12) % 12];
}

export function effectiveBpm(bpm, speed) {
  return Math.round(bpm * speed * 10) / 10;
}

export function coupledPlaybackRate(semitones, speed) {
  return speed * Math.pow(2, semitones / 12);
}
