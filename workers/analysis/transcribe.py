"""Speech-to-text for lyric transcription (optional dependency)."""
from __future__ import annotations


def transcribe(path, model_name="base"):
    """Transcribe audio to segments with word timestamps.

    Returns {"available": False, ...} when faster-whisper isn't installed.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {
            "available": False,
            "error": (
                "faster-whisper is not installed. Install it (`pip install faster-whisper`) "
                "or paste lyrics manually on the song page — manual lyrics still sync live."
            ),
        }
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, word_timestamps=True)
    out = []
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
            }
        )
    return {"available": True, "language": info.language, "lines": out}
