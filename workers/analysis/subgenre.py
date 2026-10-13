"""Zero-shot subgenre matching for the analysis pipeline (CLAP).

Model: LAION CLAP, checkpoint `laion/clap-htsat-unfused` (HTSAT audio
encoder + RoBERTa text encoder), pretrained on LAION-Audio-630K +
music/sound datasets. License: Apache-2.0 (checkpoint card metadata).
The local copy in models/clap-htsat/ was downloaded once from Hugging
Face and its weights file verified byte-identical to the official
release (sha256 1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b
296c3219bb). Inference loads with local_files_only=True — after the
one-time download, no analysis ever touches the network.

Why this model, and why the sibling checkpoint was rejected:
`laion/larger_clap_music`'s Hugging Face port is broken (LAION-AI/CLAP
issue #126): every text prompt scores a cosine similarity of ~0.00-0.01
against any audio, i.e. its embedding spaces are misaligned. This
checkpoint was probed before integration and shows healthy behaviour:
logit scale exp ≈ 18.66 (the broken port reads ≈ 1.03) and per-window
cosine similarities of 0.06-0.44 with clear family separation.

Honesty contract:
- Subgenre scores are ZERO-SHOT SIMILARITY SHARES, not trained genre
  probabilities: each analysis window's audio embedding is compared
  (cosine similarity) against one text prompt per candidate subgenre,
  similarities are averaged across windows, scaled by the model's own
  learned logit scale, and softmaxed over the fixed candidate set, so
  shares sum to 100%. They rank how close the audio sounds to each
  subgenre *description* — nothing here was trained to classify
  subgenres, and the bundle/report/UI all say so.
- The primary subgenre commits only when the top share is decisive
  (>= PRIMARY_MIN_SHARE of the softmax mass, or >= PRIMARY_DOMINANCE
  times the runner-up). Otherwise primary_subgenre is None and the
  ranked matches are shown without a forced label.
- If the model directory is absent or fails to load/run, the stage
  degrades gracefully: empty predictions + an explicit note. An
  analysis never crashes over subgenre.
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

# Primary-commit floors: with ~46 candidates, a uniform share is ~2.2%.
# 15% means one description holds a real plurality of the similarity
# mass; the 1.5x dominance alternative lets a lower-share but clearly
# separated winner commit.
PRIMARY_MIN_SHARE = 0.15
PRIMARY_DOMINANCE = 1.5

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "clap-htsat"

# Candidate vocabulary: (label, family, text prompt). Prompts describe
# the SOUND (instrumentation, rhythm, vocal character) — CLAP matches
# audio against descriptions, so prompt wording is part of the method
# and is disclosed here in full. The set is fixed for every track:
# no per-track additions or tuning.
SUBGENRES: list[tuple[str, str, str]] = [
    # --- Hip hop -----------------------------------------------------
    ("trap", "Hip hop", "trap music with heavy 808 bass, sharp rolling hi-hats and dark atmospheric melodies"),
    ("melodic trap", "Hip hop", "melodic trap music with emotional autotuned vocals, piano melodies and heavy 808s"),
    ("emo rap", "Hip hop", "emo rap music with sad emotional vocals, guitar-tinged melodies and trap drums"),
    ("sad rap", "Hip hop", "sad rap music, melancholic and emotional with slow melodic beats"),
    ("drill", "Hip hop", "drill music with sliding 808 bass, aggressive hi-hats and dark piano"),
    ("boom bap", "Hip hop", "boom bap hip hop with dusty sampled drums, jazzy loops and punchy kicks"),
    ("phonk", "Hip hop", "phonk music with lo-fi memphis samples, cowbell and heavy distorted bass"),
    ("cloud rap", "Hip hop", "cloud rap music with dreamy ethereal synths, reverb and laid-back vocals"),
    ("pluggnb", "Hip hop", "pluggnb music with soft dreamy synth melodies, gentle drums and smooth vocals"),
    ("rage", "Hip hop", "rage music with aggressive distorted synths, hard drums and energetic vocals"),
    ("alternative hip-hop", "Hip hop", "alternative hip hop music, experimental and left-field beats with unconventional flows"),
    ("jazz rap", "Hip hop", "jazz rap music with live jazz instrumentation, saxophone, piano chords and a laid-back flow"),
    ("lo-fi hip-hop", "Hip hop", "lo-fi hip hop beats with vinyl crackle, warm chords and relaxed drums"),
    ("gangsta rap", "Hip hop", "gangsta rap music with hard west-coast style beats and aggressive delivery"),
    ("conscious hip-hop", "Hip hop", "conscious hip hop music with thoughtful lyrics, soulful samples and steady beats"),
    ("horrorcore", "Hip hop", "horrorcore rap music, dark and eerie with a horror-movie atmosphere"),
    ("grime", "Hip hop", "grime music with fast electronic beats, sharp synths and rapid-fire MC delivery"),
    ("crunk", "Hip hop", "crunk music with loud shouted hooks, heavy synths and party energy"),
    # --- R&B ----------------------------------------------------------
    ("contemporary R&B", "R&B", "contemporary R&B music with smooth sensual vocals and modern production"),
    ("alternative R&B", "R&B", "alternative R&B music, moody and atmospheric with experimental production"),
    ("trap soul", "R&B", "trap soul music blending R&B singing with trap drums and emotional lyrics"),
    ("neo-soul", "R&B", "neo-soul music with warm live instrumentation, jazzy chords and soulful singing"),
    # --- Pop ----------------------------------------------------------
    ("dance pop", "Pop", "dance pop music, upbeat and catchy with electronic dance beats"),
    ("electropop", "Pop", "electropop music with bright synthesizers and polished pop vocals"),
    ("indie pop", "Pop", "indie pop music with jangly guitars, soft vocals and a homemade feel"),
    ("bedroom pop", "Pop", "bedroom pop music, lo-fi and intimate with soft vocals and gentle melodies"),
    ("hyperpop", "Pop", "hyperpop music with pitched-up vocals and glitchy maximalist production"),
    # --- Rock / metal -------------------------------------------------
    ("emo", "Rock / metal", "emo rock music with emotional guitars, confessional vocals and dynamic drums"),
    ("pop punk", "Rock / metal", "pop punk music with fast power chords, catchy hooks and youthful energy"),
    ("alternative rock", "Rock / metal", "alternative rock music with electric guitars, live drums and melodic vocals"),
    ("shoegaze", "Rock / metal", "shoegaze music with dense walls of distorted guitar, reverb and hazy vocals"),
    ("indie rock", "Rock / metal", "indie rock music with guitar-driven arrangements and understated vocals"),
    ("metalcore", "Rock / metal", "metalcore music with screaming vocals, breakdowns and heavy distorted guitars"),
    ("nu metal", "Rock / metal", "nu metal music mixing heavy guitars with hip hop beats and turntables"),
    # --- Electronic ---------------------------------------------------
    ("house", "Electronic", "house music with four-on-the-floor kick drums, disco grooves and dance energy"),
    ("techno", "Electronic", "techno music with driving repetitive beats, dark synths and warehouse energy"),
    ("dubstep", "Electronic", "dubstep music with heavy wobble bass drops and half-time drums"),
    ("drum and bass", "Electronic", "drum and bass music with fast breakbeats and deep rolling basslines"),
    ("ambient electronic", "Electronic", "ambient electronic music with slow evolving pads, no drums and spacious textures"),
    ("synthwave", "Electronic", "synthwave music with retro 1980s synthesizers, gated reverb drums and a neon atmosphere"),
    # --- Broad fallbacks ----------------------------------------------
    ("singer-songwriter", "Folk / acoustic", "singer-songwriter music with acoustic guitar or piano and intimate storytelling vocals"),
    ("folk", "Folk / acoustic", "folk music with acoustic instruments, harmonies and traditional storytelling"),
    ("country", "Country", "country music with twangy guitars, fiddle, pedal steel and southern storytelling vocals"),
    ("jazz", "Jazz", "jazz music with improvisation, saxophone, piano, upright bass and brushed drums"),
    ("soul / funk", "Soul / funk", "soul and funk music with groovy basslines, horns and powerful expressive vocals"),
    ("classical", "Classical", "classical music with a symphony orchestra, strings, piano and formal composition"),
]

SIMILARITY_NOTE = (
    "Subgenre matches are zero-shot similarity shares from CLAP: they "
    "rank how close the audio sounds to each subgenre description and "
    "are NOT trained classifier probabilities."
)


def model_dir() -> Path:
    """Model location: $PYTHAGORA_SUBGENRE_MODEL, else repo models/clap-htsat."""
    override = os.environ.get("PYTHAGORA_SUBGENRE_MODEL")
    return Path(override) if override else DEFAULT_MODEL_DIR


def summarize_subgenre_outputs(mean_similarities: np.ndarray, logit_scale: float) -> dict:
    """Map mean per-prompt cosine similarities to the bundle contract.

    Pure function (no torch, no model files) so the softmax, top-N
    truncation, and primary-commit floors are unit testable. Shares are
    softmax(logit_scale * mean_sim) over the fixed candidate set and
    therefore sum to 1 across ALL candidates (only the top 5 are
    reported).
    """
    sims = np.asarray(mean_similarities, dtype=np.float64).reshape(-1)
    if sims.shape[0] != len(SUBGENRES):
        return {
            "primary_subgenre": None,
            "subgenre_predictions": [],
            "subgenre_model": MODEL_ID,
            "subgenre_note": (
                f"Subgenre model returned {sims.shape[0]} scores for "
                f"{len(SUBGENRES)} candidates. " + SIMILARITY_NOTE
            ),
        }
    logits = float(logit_scale) * sims
    logits = logits - logits.max()  # numeric stability; softmax unchanged
    exp = np.exp(logits)
    shares = exp / exp.sum()
    order = np.argsort(-shares, kind="stable")
    predictions = [
        {
            "label": SUBGENRES[i][0],
            "family": SUBGENRES[i][1],
            "percent": round(float(shares[i]) * 100.0, 1),
        }
        for i in order[:TOP_N]
    ]
    top_share = float(shares[order[0]])
    runner_share = float(shares[order[1]]) if shares.shape[0] > 1 else 0.0
    if top_share >= PRIMARY_MIN_SHARE or (
        runner_share > 0 and top_share >= PRIMARY_DOMINANCE * runner_share
    ):
        primary = SUBGENRES[int(order[0])][0]
        note = SIMILARITY_NOTE
    else:
        primary = None
        note = (
            f"Top subgenre share {top_share * 100:.1f}% is not decisive "
            f"(needs {PRIMARY_MIN_SHARE * 100:.0f}% or "
            f"{PRIMARY_DOMINANCE:g}x the runner-up), so no single "
            "subgenre is claimed. " + SIMILARITY_NOTE
        )
    return {
        "primary_subgenre": primary,
        "subgenre_predictions": predictions,
        "subgenre_model": MODEL_ID,
        "subgenre_note": note,
    }


def _degraded(reason: str) -> dict:
    return {
        "primary_subgenre": None,
        "subgenre_predictions": [],
        "subgenre_model": MODEL_ID,
        "subgenre_note": f"Subgenre model unavailable: {reason}",
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
    concurrent first use (observed live: four simultaneous job threads
    raced the first `from transformers import ClapModel` and three of
    them got a spurious ImportError). One thread loads; the rest wait
    and then share the state.
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
            prompts = [prompt for _label, _family, prompt in SUBGENRES]
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


def predict_subgenre(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Run zero-shot subgenre matching on a mono waveform.

    Returns the bundle subgenre dict, or a degraded dict (empty
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
        return summarize_subgenre_outputs(mean_sims, logit_scale)
    except Exception as exc:  # noqa: BLE001 - degradation is the contract
        return _degraded(f"inference failed ({exc.__class__.__name__}: {exc})")


def analyze_subgenre(audio_path: str) -> dict:
    """Full subgenre step for the pipeline: load audio, match, time it.

    Adds `subgenre_seconds` (wall time of this step) to the returned
    dict; the pipeline strips it before building the bundle.
    Never raises.
    """
    started = time.perf_counter()
    try:
        import librosa

        y, _sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        result = predict_subgenre(y, SAMPLE_RATE)
    except Exception as exc:  # noqa: BLE001 - degradation is the contract
        result = _degraded(f"audio load failed ({exc.__class__.__name__}: {exc})")
    result["subgenre_seconds"] = round(time.perf_counter() - started, 3)
    return result
