/* Track sorting (Phase F) — pure comparator logic, no DOM.
 *
 * Sort keys: name | date | tempo | key | duration. Tracks without an
 * analysis (no tempo/key/duration) always sort last, in both directions —
 * "unknown" is not zero.
 */

const PITCH_INDEX = { C: 0, "C#": 1, D: 2, "D#": 3, E: 4, F: 5, "F#": 6, G: 7, "G#": 8, A: 9, "A#": 10, B: 11 };

function valueOf(track, telemetry, key) {
  const tel = telemetry ? telemetry[track.id] : null;
  switch (key) {
    case "name":
      return (track.title || "").toLowerCase();
    case "date":
      return track.created_at ? Date.parse(track.created_at) || 0 : 0;
    case "tempo":
      return tel && typeof tel.bpm === "number" ? tel.bpm : null;
    case "duration":
      return tel && typeof tel.duration === "number" ? tel.duration : null;
    case "key": {
      if (!tel || !(tel.musical_key in PITCH_INDEX)) return null;
      return PITCH_INDEX[tel.musical_key] * 2 + (tel.mode === "minor" ? 1 : 0);
    }
    default:
      return null;
  }
}

export function sortTracks(tracks, telemetry, key = "date", dir = "desc") {
  const sign = dir === "asc" ? 1 : -1;
  return [...tracks].sort((a, b) => {
    const va = valueOf(a, telemetry, key);
    const vb = valueOf(b, telemetry, key);
    if (va === null && vb === null) return 0;
    if (va === null) return 1; // missing always last
    if (vb === null) return -1;
    if (va < vb) return -1 * sign;
    if (va > vb) return 1 * sign;
    return (a.title || "").localeCompare(b.title || "");
  });
}
