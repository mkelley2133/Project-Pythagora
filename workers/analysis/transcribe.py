"""Speech-to-text for lyric transcription (optional dependency)."""
from __future__ import annotations

import math
import os

DEFAULT_MODEL = os.environ.get("PYTHAGORAS_WHISPER_MODEL", "large-v3")


def confidence_of(avg_logprob: float, no_speech_prob: float) -> float:
    """Map Whisper segment scores to a 0-1 confidence.

    avg_logprob is typically in [-2, 0]; exp() maps that to (0, 1].
    Multiplied by (1 - no_speech_prob) so segments that are probably
    not speech score low even when the log-prob looks fine.
    """
    try:
        base = math.exp(float(avg_logprob))
        if math.isnan(base):
            base = 0.0
    except (TypeError, ValueError, OverflowError):
        base = 0.0
    try:
        speech = 1.0 - float(no_speech_prob)
    except (TypeError, ValueError):
        speech = 1.0
    return round(max(0.0, min(1.0, base * max(0.0, speech))), 3)


def transcribe(path, model_name=None, source="full mix"):
    """Transcribe audio to segments with word timestamps and confidence.

    model_name defaults to PYTHAGORAS_WHISPER_MODEL or "large-v3".
    Returns {"available": False, ...} when faster-whisper isn't installed.
    """
    model_name = model_name or DEFAULT_MODEL
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {
            "available": False,
            "model": model_name,
            "source": source,
            "error": (
                "faster-whisper is not installed. Install it (`pip install faster-whisper`) "
                "or paste lyrics manually on the song page — manual lyrics still sync live."
            ),
        }
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, word_timestamps=True)
    out = []
    result = {"available": True, "model": model_name, "source": source, "lines": out}
    for seg in segments:
        words = []
        for w in seg.words or []:
            words.append({"w": w.word, "start": round(w.start, 2), "end": round(w.end, 2)})
        out.append(
            {
                "text": seg.text.strip(),
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "words": words,
                "confidence": confidence_of(
                    getattr(seg, "avg_logprob", -1.0),
                    getattr(seg, "no_speech_prob", 0.0),
                ),
            }
        )
    result["language"] = info.language
    return result
