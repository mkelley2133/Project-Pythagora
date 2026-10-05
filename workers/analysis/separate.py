"""Vocal-stem separation via Demucs (optional dependency).

Isolating the vocal stem before transcription (and vocal forensics) is the
single biggest accuracy win on full mixes: Whisper no longer has to guess
which parts of a trap instrumental are voice.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def separate_vocals(wav_path: str, out_dir: str | None = None) -> dict:
    """Separate the vocal stem from a wav file with Demucs.

    Returns {"available": bool, "vocals_path": str | None, "error": str | None}.
    Skips cleanly when demucs isn't installed — callers fall back to the mix.
    """
    try:
        import demucs  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "vocals_path": None,
            "error": "demucs is not installed (`pip install demucs`) — using the full mix instead.",
        }
    src = Path(wav_path)
    work = Path(out_dir) if out_dir else src.parent / f".demucs-{src.stem}"
    work.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-o",
        str(work),
        str(src),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        return {"available": False, "vocals_path": None, "error": "demucs timed out"}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        return {"available": False, "vocals_path": None, "error": f"demucs failed: {tail}"}
    matches = sorted(work.rglob("vocals.wav")) + sorted(work.rglob("vocals.mp3"))
    if not matches:
        return {
            "available": False,
            "vocals_path": None,
            "error": "demucs finished but produced no vocal stem",
        }
    return {"available": True, "vocals_path": str(matches[0]), "error": None}
