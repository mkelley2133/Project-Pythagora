"""Deterministic chord recognition + Roman numeral analysis.

Pipeline: CQT chromagram -> fixed time windows -> cosine similarity against
24 major/minor triad templates -> temporal smoothing -> merged chord events ->
Roman numerals relative to the track's key.

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


def recognize_chords(
    audio_path: str,
    key: str = "C",
    mode: str = "major",
    window_seconds: float = 0.5,
    sr: int = 22050,
) -> list[dict]:
    """Recognize chords over fixed time windows.

    Returns a list of merged events:
    {"chord": "Dm", "roman": "iv", "start": 10.5, "end": 13.0, "confidence": 0.87}
    """
    _require_librosa()
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    if y.size == 0:
        return []

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    if chroma.shape[1] == 0:
        return []

    # Aggregate chroma into fixed windows (robust when beat tracking fails,
    # e.g. on ambient/pad-heavy material).
    win_frames = max(1, int(window_seconds * sr / hop))
    n_windows = int(np.ceil(chroma.shape[1] / win_frames))
    labels: list[str] = []
    confidences: list[float] = []
    for w in range(n_windows):
        seg = chroma[:, w * win_frames : (w + 1) * win_frames]
        vec = np.median(seg, axis=1)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            labels.append(labels[-1] if labels else "N.C.")
            confidences.append(0.0)
            continue
        vec = vec / norm
        sims = {name: float(vec @ tmpl) for name, tmpl in _TEMPLATES.items()}
        best = max(sims, key=sims.get)
        labels.append(best)
        confidences.append(round(sims[best], 3))

    # Temporal smoothing: majority vote over a 3-window neighborhood.
    smoothed: list[str] = []
    for i, lab in enumerate(labels):
        window = labels[max(0, i - 1) : i + 2]
        smoothed.append(max(set(window), key=window.count))

    # Merge consecutive identical labels into events.
    events: list[dict] = []
    start_w = 0
    for i in range(1, len(smoothed) + 1):
        if i == len(smoothed) or smoothed[i] != smoothed[start_w]:
            chord = smoothed[start_w]
            conf = round(float(np.mean(confidences[start_w:i])), 3)
            events.append(
                {
                    "chord": chord,
                    "roman": roman_numeral(chord, key, mode),
                    "start": round(start_w * window_seconds, 2),
                    "end": round(i * window_seconds, 2),
                    "confidence": conf,
                }
            )
            start_w = i
    return events
