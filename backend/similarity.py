"""Similar-tracks retrieval over CLAP audio embeddings.

The retrieval half of the embeddings story: the subgenre stage
(workers/analysis/subgenre.py) embeds audio with LAION CLAP to rank it
against text descriptions; this module embeds each track the same way
and ranks the OTHER tracks by cosine similarity of their mean audio
embeddings — "what else in the library sounds like this track."

Honesty contract, same as the rest of the system:
- A similarity score is a cosine between two embedding vectors. It is
  a closeness ranking, not a genre claim, not a probability, and it is
  labeled as similarity everywhere it is returned.
- Embeddings are computed with the subgenre model's own loader,
  windowing, and normalization, and cached in-process per track id
  (analyses themselves are in-memory in v1 — see backend/store.py).
  The first similar-tracks call for a library pays one embedding pass
  per track; later calls reuse the cache.
- If the CLAP weights are absent the feature degrades exactly like the
  subgenre stage: an explicit unavailable note and an empty ranking —
  never an exception, never a fabricated score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from backend import store
from backend.models.schemas import TrackRecord
from workers.analysis import subgenre as subgenre_mod

_REPO_ROOT = Path(__file__).resolve().parents[1]

# In-process embedding cache: track_id -> L2-normalized mean CLAP audio
# embedding. Mirrors the in-memory analysis store's lifetime.
_EMBEDDING_CACHE: dict[str, np.ndarray] = {}

SIMILARITY_NOTE = (
    "Similarity is the cosine between mean CLAP audio embeddings "
    "(the same embeddings the subgenre stage matches against text). "
    "It ranks sonic closeness inside this library only — it is not a "
    "genre classification and not a probability."
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two vectors; 0.0 when either has zero norm."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_by_similarity(
    query: np.ndarray,
    candidates: dict[str, np.ndarray],
    top_n: int = 5,
    exclude_id: str | None = None,
) -> list[dict]:
    """Rank candidate embeddings against the query, best first.

    Pure NumPy, deterministic (ties break by track id), so the ranking
    is unit-testable with synthetic vectors. Returns
    [{"track_id": ..., "similarity": ...}] with the query's own id
    excluded when `exclude_id` is given.
    """
    scored = [
        {"track_id": track_id, "similarity": round(cosine_similarity(query, vec), 4)}
        for track_id, vec in candidates.items()
        if track_id != exclude_id
    ]
    scored.sort(key=lambda item: (-item["similarity"], item["track_id"]))
    return scored[: max(0, top_n)]


def _resolve_audio_path(track: TrackRecord) -> Path | None:
    path = Path(track.original_file_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path if path.is_file() else None


def compute_track_embedding(audio_path: str | Path) -> np.ndarray | None:
    """Mean CLAP audio embedding for one file, or None when the model
    is unavailable. Mirrors predict_subgenre's embedding half exactly:
    same sample rate, window size/count, per-window L2 normalization;
    the per-window embeddings are then averaged and renormalized."""
    state, _error = subgenre_mod._load_model()
    if state is None:
        return None
    model, processor, _text_emb, _logit_scale = state
    try:
        import librosa
        import torch

        y, _sr = librosa.load(str(audio_path), sr=subgenre_mod.SAMPLE_RATE, mono=True)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return None
        window = int(subgenre_mod.WINDOW_SECONDS * subgenre_mod.SAMPLE_RATE)
        starts = subgenre_mod._window_starts(y.size, window)
        clips = []
        for start in starts:
            clip = y[start : start + window]
            if clip.size < window:
                clip = np.pad(clip, (0, window - clip.size))
            clips.append(clip)
        audio_inputs = processor(
            audio=clips, sampling_rate=subgenre_mod.SAMPLE_RATE, return_tensors="pt"
        )
        with torch.no_grad():
            aud_emb = model.get_audio_features(**audio_inputs)
        aud_emb = aud_emb / aud_emb.norm(p=2, dim=-1, keepdim=True)
        mean_emb = aud_emb.mean(axis=0)
        norm = float(mean_emb.norm())
        if norm == 0.0:
            return None
        return (mean_emb / norm).cpu().numpy().astype(np.float64)
    except Exception:  # noqa: BLE001 - degradation is the contract
        return None


def get_or_compute_embedding(track: TrackRecord) -> np.ndarray | None:
    """Cached embedding for a track; None when it cannot be embedded
    (missing file or unavailable model)."""
    cached = _EMBEDDING_CACHE.get(track.id)
    if cached is not None:
        return cached
    path = _resolve_audio_path(track)
    if path is None:
        return None
    embedding = compute_track_embedding(path)
    if embedding is not None:
        _EMBEDDING_CACHE[track.id] = embedding
    return embedding


def similar_tracks(track_id: str, top_n: int = 5) -> dict | None:
    """Rank the library against one track. None when the track does not
    exist (the router turns that into a 404); a degraded dict with an
    explicit note when embeddings are unavailable."""
    track = store.get_track(track_id)
    if track is None:
        return None
    base = {
        "track": {"id": track.id, "title": track.title, "artist": track.artist},
        "model": subgenre_mod.MODEL_ID,
        "similar": [],
        "note": SIMILARITY_NOTE,
    }
    query = get_or_compute_embedding(track)
    if query is None:
        base["note"] = (
            "Similar tracks unavailable: the CLAP model could not embed "
            "this track (weights absent or audio unreadable). " + SIMILARITY_NOTE
        )
        return base
    candidates: dict[str, np.ndarray] = {}
    titles: dict[str, TrackRecord] = {}
    for other in store.list_tracks():
        if other.id == track_id:
            continue
        embedding = get_or_compute_embedding(other)
        if embedding is not None:
            candidates[other.id] = embedding
            titles[other.id] = other
    ranked = rank_by_similarity(query, candidates, top_n=top_n, exclude_id=track_id)
    base["similar"] = [
        {
            "id": item["track_id"],
            "title": titles[item["track_id"]].title,
            "artist": titles[item["track_id"]].artist,
            "similarity": item["similarity"],
        }
        for item in ranked
    ]
    return base
