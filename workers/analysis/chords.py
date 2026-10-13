"""Deterministic chord recognition + Roman numeral analysis.

Pipeline: HPSS harmonic component -> CQT chromagram -> beat-synchronous
segments (fixed windows when beat tracking is unreliable) -> cosine
similarity against 24 major/minor triad templates -> harmonic-bleed
correction (see _BLEED_BETA) -> Viterbi temporal smoothing (chord changes
prefer beat boundaries; single-segment flips are suppressed) -> short-event
absorption -> merged chord events -> Roman numerals relative to the track's
key.

Everything here is deterministic signal processing (no generative models),
in keeping with the project's zero-hallucination design: a chord label is a
measurement, never a guess.
"""
from __future__ import annotations

import numpy as np

try:
    import librosa

    _HAVE_LIBROSA = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_LIBROSA = False

# Pitch-class indices: C=0, C#=1, ..., B=11.
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_MAJOR_INTERVALS = (0, 4, 7)
_MINOR_INTERVALS = (0, 3, 7)


def _build_templates() -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for root in range(12):
        for suffix, intervals in (("", _MAJOR_INTERVALS), ("m", _MINOR_INTERVALS)):
            vec = np.zeros(12)
            # Slightly weight the root so root-position voicings win ties.
            for i, iv in enumerate(intervals):
                vec[(root + iv) % 12] = 1.0 if i else 1.15
            name = f"{PITCH_CLASSES[root]}{suffix}"
            templates[name] = vec / np.linalg.norm(vec)
    return templates


_TEMPLATES = _build_templates()
_CHORD_NAMES = list(_TEMPLATES.keys())
_TEMPLATE_INDEX = {name: i for i, name in enumerate(_CHORD_NAMES)}

# Harmonic-bleed correction for major/minor quality decisions.
#
# A root tone's 5th harmonic IS its major third (two octaves up), so the
# major-third pitch class carries a bleed floor proportional to root
# energy that the minor-third pitch class does not (no low harmonic of
# the root or fifth lands on the minor third). The template decision
# between a root's major and minor forms rides entirely on the two third
# pitch classes, so when a chord's real third is weak, that bleed
# decides the label instead, and minor chords flip to major (measured on
# a human-verified C-minor track: the tonic slot of an identical loop
# alternated Cm/C across repetitions, 25s of C major in a C-minor song).
#
# The correction discounts each major template's segment similarity by
# _BLEED_BETA * (the segment's root chroma energy). Dose selection
# (validated 2026-10-13; do not retune without re-running both checks):
# - A synthetic weak-third battery (full-harmonic-series triads, exact
#   ground truth) flips Dm->D / Em->E at the weakest third gain on the
#   uncorrected recognizer; every dose in 0.0075-0.03 fixes the battery
#   with no regression at stronger gains, and 0.05 over-corrects.
# - On real tracks with human-verified keys, the tonic-quality share
#   (tonic read in the key's quality) rises with dose — This Side
#   0.73 -> 0.89 at 0.0075, 0.95 at 0.03 — but so does collateral drift
#   on non-tonic major chords whose thirds are also weak (Drowning's
#   IV chord erodes from 0.0075 up, and is gone at 0.03). 0.0075 is the
#   smallest dose that fixes the battery and captures most of the
#   verified tonic win, so it is the shipped dose.
# A per-track calibrated dose was tried and rejected: in real mixes the
# major-third pitch class also carries melody and neighboring-chord
# energy, so the calibrated value saturates and destroys verified major
# chords (Drowning tonic share 0.89 -> 0.21).
_BLEED_BETA = 0.0075

_MAJOR_ROMAN = {
    0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV",
    6: "bV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII",
}
_MINOR_ROMAN = {
    0: "i", 1: "bi", 2: "iio", 3: "III", 4: "III+", 5: "iv",
    6: "bV", 7: "v", 8: "VI", 9: "VI+", 10: "VII", 11: "VII+",
}


def _normalize_note(name: str) -> str:
    flats = {"BB": "A#", "DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#"}
    up = name.strip().upper().replace("♯", "#").replace("♭", "B")
    return flats.get(up, up)


def roman_numeral(chord: str, key: str = "C", mode: str = "major") -> str:
    """Map a chord symbol to a Roman numeral relative to key/mode.

    Simplified: natural-minor degrees for minor keys; chromatic roots get
    flat-prefixed labels. Documented simplification, not a guess.
    """
    key = _normalize_note(key)
    if key not in PITCH_CLASSES:
        return ""
    root_name = _normalize_note(chord[:-1] if chord.endswith("m") else chord)
    if root_name not in PITCH_CLASSES:
        return ""
    offset = (PITCH_CLASSES.index(root_name) - PITCH_CLASSES.index(key)) % 12
    table = _MAJOR_ROMAN if mode.lower() == "major" else _MINOR_ROMAN
    numeral = table[offset]
    # Match case to chord quality for diatonic numerals.
    if chord.endswith("m") and numeral.isupper() and "b" not in numeral:
        numeral = numeral.lower()
    elif not chord.endswith("m") and numeral.islower() and numeral not in ("iio",):
        numeral = numeral.upper()
    return numeral


def _require_librosa() -> None:
    if not _HAVE_LIBROSA:
        raise RuntimeError("librosa is required for chord recognition")


def _viterbi_smooth(sims: np.ndarray, transition_penalty: float = 0.12) -> np.ndarray:
    """Best label path through per-segment template similarities.

    sims: (n_segments, n_labels) cosine similarities. Switching labels costs
    `transition_penalty`, so a chord change must earn its way in with
    sustained evidence — isolated one-segment flips are suppressed.
    Returns an int array of label indices, one per segment.
    """
    n_segments, n_labels = sims.shape
    if n_segments == 0:
        return np.array([], dtype=int)
    dp = sims[0].astype(float).copy()
    back = np.zeros((n_segments, n_labels), dtype=int)
    switch_cost = transition_penalty * (1.0 - np.eye(n_labels))
    for t in range(1, n_segments):
        trans = dp[:, None] - switch_cost
        back[t] = np.argmax(trans, axis=0)
        dp = sims[t] + trans[back[t], np.arange(n_labels)]
    path = np.zeros(n_segments, dtype=int)
    path[-1] = int(np.argmax(dp))
    for t in range(n_segments - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def recognize_chords(
    audio_path: str,
    key: str = "C",
    mode: str = "major",
    window_seconds: float = 0.5,
    sr: int = 22050,
    beat_sync: bool = True,
    transition_penalty: float = 0.12,
    min_event_seconds: float = 0.4,
) -> list[dict]:
    """Recognize chords over beat-synchronous segments.

    Beat tracking anchors chord changes to beats (the fix for over-segmented
    trap harmony); when beats can't be tracked reliably the recognizer falls
    back to fixed windows. Either way, labels pass through Viterbi smoothing
    and sub-`min_event_seconds` events are absorbed into a neighbor.

    Returns a list of merged events:
    {"chord": "Dm", "roman": "iv", "start": 10.5, "end": 13.0, "confidence": 0.87}
    """
    _require_librosa()
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    if y.size == 0:
        return []

    hop = 512
    # Chroma from the harmonic component: drums/808 transients bleed less
    # into pitch-class estimates than they do from the raw mix.
    try:
        from workers.analysis.audio import hpss_waveform

        y_harm, _ = hpss_waveform(y)
    except Exception:
        y_harm = y
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop)
    n_frames = chroma.shape[1]
    if n_frames == 0:
        return []
    duration = y.size / sr

    # --- Segment boundaries (in chroma frames) ---
    seg_bounds: np.ndarray | None = None
    if beat_sync and duration >= 2.0:
        try:
            _, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=hop, units="frames"
            )
            beats = np.unique(
                np.clip(np.asarray(beat_frames, dtype=int), 0, n_frames)
            )
            if beats.size >= 4:
                bounds = np.unique(np.concatenate([[0], beats, [n_frames]]))
                if bounds.size >= 5:
                    seg_bounds = bounds
        except Exception:
            seg_bounds = None
    if seg_bounds is None:
        win_frames = max(1, int(window_seconds * sr / hop))
        seg_bounds = np.unique(
            np.concatenate([np.arange(0, n_frames, win_frames), [n_frames]])
        )
    seg_times = librosa.frames_to_time(seg_bounds, sr=sr, hop_length=hop)
    n_segs = len(seg_bounds) - 1

    # --- Per-segment template similarities ---
    template_matrix = np.stack([_TEMPLATES[name] for name in _CHORD_NAMES])
    seg_sims = np.zeros((n_segs, len(_CHORD_NAMES)))
    seg_vecs: list[np.ndarray | None] = [None] * n_segs
    silent = np.zeros(n_segs, dtype=bool)
    for i in range(n_segs):
        seg = chroma[:, seg_bounds[i] : seg_bounds[i + 1]]
        if seg.shape[1] == 0:
            silent[i] = True
            continue
        vec = np.median(seg, axis=1)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            silent[i] = True
            continue
        seg_vecs[i] = vec / norm
        seg_sims[i] = template_matrix @ seg_vecs[i]
    # --- Harmonic-bleed correction (see _BLEED_BETA) ---
    # Discount each major template by the bleed its root's own harmonics
    # are expected to have injected into the major-third pitch class.
    # Runs before the silent-segment carry below so carried evidence is
    # the corrected evidence.
    for i in range(n_segs):
        vec = seg_vecs[i]
        if vec is None:
            continue
        for root in range(12):
            j_major = _TEMPLATE_INDEX[PITCH_CLASSES[root]]
            seg_sims[i, j_major] -= _BLEED_BETA * float(vec[root])

    for i in range(n_segs):
        if silent[i]:
            # Carry the previous segment's evidence so silence doesn't
            # invent a chord change.
            seg_sims[i] = seg_sims[i - 1] if i > 0 else -1.0

    label_idx = _viterbi_smooth(seg_sims, transition_penalty)
    labels = [_CHORD_NAMES[j] for j in label_idx]
    if silent[0]:
        labels[0] = "N.C."
    seg_conf = np.array(
        [
            0.0 if silent[i] else float(seg_sims[i, label_idx[i]])
            for i in range(n_segs)
        ]
    )

    # --- Merge consecutive identical labels into events ---
    events: list[dict] = []
    start = 0
    for i in range(1, n_segs + 1):
        if i == n_segs or labels[i] != labels[start]:
            events.append(
                {
                    "chord": labels[start],
                    "start": float(seg_times[start]),
                    "end": float(seg_times[i]),
                    "confidence": float(np.mean(seg_conf[start:i])) if i > start else 0.0,
                    "_n": i - start,
                }
            )
            start = i

    # --- Absorb implausibly short events into a neighbor ---
    changed = True
    while changed and len(events) > 1:
        changed = False
        for idx, event in enumerate(events):
            if event["end"] - event["start"] >= min_event_seconds:
                continue
            if idx == 0:
                target = 1
                events[target]["start"] = event["start"]
            else:
                target = idx - 1
                events[target]["end"] = event["end"]
            total_n = events[target]["_n"] + event["_n"]
            events[target]["confidence"] = (
                events[target]["confidence"] * events[target]["_n"]
                + event["confidence"] * event["_n"]
            ) / total_n
            events[target]["_n"] = total_n
            del events[idx]
            changed = True
            break

    return [
        {
            "chord": event["chord"],
            "roman": roman_numeral(event["chord"], key, mode),
            "start": round(event["start"], 2),
            "end": round(event["end"], 2),
            "confidence": round(event["confidence"], 3),
        }
        for event in events
    ]
