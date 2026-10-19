"""Zero-shot emotion matching for the analysis pipeline (CLAP).

Model: the same LAION CLAP checkpoint the subgenre stage uses,
`laion/clap-htsat-unfused` (HTSAT audio encoder + RoBERTa text
encoder), from the same local copy in models/clap-htsat/, loaded with
local_files_only=True — no analysis ever touches the network, and no
new model is downloaded for this stage. See
workers/analysis/subgenre.py for the checkpoint's verification notes
and why the sibling `larger_clap_music` port was rejected.

Honesty contract:
- Emotion scores are ZERO-SHOT EMOTION SIMILARITY SHARES, not trained
  emotion probabilities and not a reading of what a song actually
  feels: each analysis window's audio embedding is compared (cosine
  similarity) against one text prompt per candidate emotion,
  similarities are averaged across windows, scaled by the model's own
  learned logit scale, and softmaxed over the fixed candidate set, so
  shares sum to 100%. They rank how close the audio sounds to each
  emotion *description* — nothing here was trained to classify
  emotions, and the bundle/report/UI all say so.
- The primary emotion commits only when the top share is decisive
  (>= PRIMARY_MIN_SHARE of the softmax mass, or >= PRIMARY_DOMINANCE
  times the runner-up) — the same floors the subgenre stage uses.
  Otherwise primary_emotion is None and the ranked matches are shown
  without a forced label.
- If the model directory is absent or fails to load/run, the stage
  degrades gracefully: empty predictions + an explicit note. An
  analysis never crashes over emotion.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "LAION CLAP (clap-htsat-unfused)"
MODEL_VERSION = "laion/clap-htsat-unfused · Apache-2.0"
MODEL_ID = f"{MODEL_NAME} — {MODEL_VERSION}"

SAMPLE_RATE = 48000  # CLAP's native rate
WINDOW_SECONDS = 10.0
MAX_WINDOWS = 12  # evenly spaced cap; ~0.3-1.0 s CPU inference per window
TOP_N = 5

# Primary-commit floors — identical to the subgenre stage: a uniform
# share over this candidate set is ~8.3%, so 15% means one description
# holds a real plurality of the similarity mass; the 1.5x dominance
# alternative lets a lower-share but clearly separated winner commit.
PRIMARY_MIN_SHARE = 0.15
PRIMARY_DOMINANCE = 1.5

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "clap-htsat"

# Candidate vocabulary: (label, text prompt). Prompts describe the
# emotional character of the SOUND (vocal delivery, energy, mood of
# the production) in the same descriptive style as the subgenre
# vocabulary — CLAP matches audio against descriptions, so prompt
# wording is part of the method and is disclosed here in full. The
# set is fixed for every track: no per-track additions or tuning.
EMOTIONS: list[tuple[str, str]] = [
    ("heartbroken confession", "a heartbroken confessional song, raw sorrowful singing about lost love, heartbreak and regret"),
    ("defiant anthem", "a defiant anthem, bold unbowed singing pushing back against the odds with triumphant energy"),
    ("numb dissociation", "a numb dissociated song, flat detached vocals, emotionally distant and hazy, drifting and empty"),
    ("euphoric celebration", "a euphoric celebration song, joyful explosive energy, made for dancing and shouting along"),
    ("anxious spiral", "an anxious spiraling song, tense restless energy, racing thoughts over uneasy production"),
    ("quiet grief", "a quietly grieving song, soft mourning singing, sparse and heavy with loss and absence"),
    ("bitter resentment", "a bitter resentful song, cold cutting delivery, anger and resentment simmering under the surface"),
    ("hopeful longing", "a hopeful longing song, yearning tender singing reaching toward something better"),
    ("reckless joy", "a reckless joyful song, carefree wild energy, laughing and living fast without consequences"),
    ("weary resignation", "a weary resigned song, tired defeated singing, quietly accepting how things turned out"),
    ("tender devotion", "a tender devoted love song, warm intimate romantic singing, gentle and close"),
    ("brooding menace", "a dark menacing song, brooding threatening energy, low, heavy and ominous"),
]

SIMILARITY_NOTE = (
    "Emotion matches are zero-shot emotion similarity shares from "
    "CLAP: they rank how close the audio sounds to each emotion "
    "description and are NOT trained classifier probabilities — and "
    "not a claim about what the song actually feels."
)


def model_dir() -> Path:
    """Model location: $PYTHAGORA_EMOTION_MODEL, else repo models/clap-htsat."""
    override = os.environ.get("PYTHAGORA_EMOTION_MODEL")
    return Path(override) if override else DEFAULT_MODEL_DIR


def summarize_emotion_outputs(mean_similarities: np.ndarray, logit_scale: float) -> dict:
    """Map mean per-prompt cosine similarities to the bundle contract.

    Pure function (no torch, no model files) so the softmax, top-N
    truncation, and primary-commit floors are unit testable. Shares are
    softmax(logit_scale * mean_sim) over the fixed candidate set and
    therefore sum to 1 across ALL candidates; `emotion_predictions`
    reports only the top 5, `emotion_all_shares` carries every
    candidate's share for consumers that need the full distribution.
    """
    sims = np.asarray(mean_similarities, dtype=np.float64).reshape(-1)
    if sims.shape[0] != len(EMOTIONS):
        return {
            "primary_emotion": None,
            "emotion_predictions": [],
            "emotion_all_shares": [],
            "emotion_model": MODEL_ID,
            "emotion_note": (
                f"Emotion model returned {sims.shape[0]} scores for "
                f"{len(EMOTIONS)} candidates. " + SIMILARITY_NOTE
            ),
        }
    logits = float(logit_scale) * sims
    logits = logits - logits.max()  # numeric stability; softmax unchanged
    exp = np.exp(logits)
    shares = exp / exp.sum()
    order = np.argsort(-shares, kind="stable")
    all_shares = [
        {
            "label": EMOTIONS[i][0],
            "percent": round(float(shares[i]) * 100.0, 1),
        }
        for i in order
    ]
    predictions = all_shares[:TOP_N]
    top_share = float(shares[order[0]])
    runner_share = float(shares[order[1]]) if shares.shape[0] > 1 else 0.0
    if top_share >= PRIMARY_MIN_SHARE or (
        runner_share > 0 and top_share >= PRIMARY_DOMINANCE * runner_share
    ):
        primary = EMOTIONS[int(order[0])][0]
        note = SIMILARITY_NOTE
    else:
        primary = None
        note = (
            f"Top emotion share {top_share * 100:.1f}% is not decisive "
            f"(needs {PRIMARY_MIN_SHARE * 100:.0f}% or "
            f"{PRIMARY_DOMINANCE:g}x the runner-up), so no single "
            "emotion is claimed. " + SIMILARITY_NOTE
        )
    return {
        "primary_emotion": primary,
        "emotion_predictions": predictions,
        "emotion_all_shares": all_shares,
        "emotion_model": MODEL_ID,
        "emotion_note": note,
    }


def _degraded(reason: str) -> dict:
    return {
        "primary_emotion": None,
        "emotion_predictions": [],
        "emotion_all_shares": [],
        "emotion_model": MODEL_ID,
        "emotion_note": f"Emotion model unavailable: {reason}",
    }


_STATE = None  # (model, processor, text_embeddings, logit_scale)
_STATE_ERROR: str | None = None
_LOAD_LOCK = threading.Lock()


def _load_model():
    """Lazy singleton. Returns (state, error_reason).

    Loads strictly from the local model directory (local_files_only) —
    the one-time download is a setup step, never an analysis-time act.

    The load is serialized behind a lock: pipeline jobs run on a thread
    pool, and transformers' lazy class imports are not safe under
    concurrent first use (see subgenre.py for the observed failure).
    One thread loads; the rest wait and then share the state.
    """
    global _STATE, _STATE_ERROR
    if _STATE is not None or _STATE_ERROR is not None:
        return _STATE, _STATE_ERROR
    with _LOAD_LOCK:
        if _STATE is not None or _STATE_ERROR is not None:
            return _STATE, _STATE_ERROR
        path = model_dir()
        if not path.is_dir():
            _STATE_ERROR = f"model directory not found at {path}"
            return None, _STATE_ERROR
        try:
            import torch
            from transformers import ClapModel, ClapProcessor

            model = ClapModel.from_pretrained(str(path), local_files_only=True)
            processor = ClapProcessor.from_pretrained(str(path), local_files_only=True)
            model.eval()
            prompts = [prompt for _label, prompt in EMOTIONS]
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
            with torch.no_grad():
                text_emb = model.get_text_features(**text_inputs)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            logit_scale = float(model.logit_scale_a.detach().exp())
            _STATE = (model, processor, text_emb, logit_scale)
        except Exception as exc:  # noqa: BLE001 - degradation is the contract
            _STATE_ERROR = f"failed to load {path.name} ({exc.__class__.__name__}: {exc})"
        return _STATE, _STATE_ERROR


def _window_starts(total_samples: int, window: int) -> list[int]:
    if total_samples <= window:
        return [0]
    last = total_samples - window
    count = min(MAX_WINDOWS, max(1, total_samples // window))
    if count == 1:
        return [0]
    return [round(i * last / (count - 1)) for i in range(count)]


def predict_emotion(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Run zero-shot emotion matching on a mono waveform.

    Returns the bundle emotion dict, or a degraded dict (empty
    predictions + explicit note) when the model is unavailable.
    Never raises.
    """
    state, error = _load_model()
    if state is None:
        return _degraded(error or "unknown load failure")
    model, processor, text_emb, logit_scale = state
    try:
        import torch

        y = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return _degraded("no audio samples to classify")
        window = int(WINDOW_SECONDS * sample_rate)
        starts = _window_starts(y.size, window)
        clips = []
        for start in starts:
            clip = y[start : start + window]
            if clip.size < window:
                clip = np.pad(clip, (0, window - clip.size))
            clips.append(clip)
        audio_inputs = processor(
            audio=clips, sampling_rate=sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            aud_emb = model.get_audio_features(**audio_inputs)
        aud_emb = aud_emb / aud_emb.norm(p=2, dim=-1, keepdim=True)
        sims = (aud_emb @ text_emb.T).cpu().numpy()  # (windows, candidates)
        mean_sims = sims.mean(axis=0)
        return summarize_emotion_outputs(mean_sims, logit_scale)
    except Exception as exc:  # noqa: BLE001 - degradation is the contract
        return _degraded(f"inference failed ({exc.__class__.__name__}: {exc})")


def analyze_emotion(audio_path: str) -> dict:
    """Full emotion step for the pipeline: load audio, match, time it.

    Adds `emotion_seconds` (wall time of this step) to the returned
    dict; the pipeline strips it before building the bundle.
    Never raises.
    """
    started = time.perf_counter()
    try:
        import librosa

        y, _sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        result = predict_emotion(y, SAMPLE_RATE)
    except Exception as exc:  # noqa: BLE001 - degradation is the contract
        result = _degraded(f"audio load failed ({exc.__class__.__name__}: {exc})")
    result["emotion_seconds"] = round(time.perf_counter() - started, 3)
    return result
