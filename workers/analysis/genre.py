"""Trained genre classification for the analysis pipeline.

Model: PANNs Cnn10 (Kong et al., "PANNs: Large-Scale Pretrained Audio
Neural Networks for Audio Pattern Recognition", arXiv:1912.10211),
pretrained on AudioSet (527 classes). Checkpoint `Cnn10_mAP=0.380.pth`
from the authors' official Zenodo record 3987831
(md5 bfb1f1f9968938fa8ef4012b8471f5f6), licensed CC BY 4.0 — commercial
use permitted with attribution. The Cnn10 architecture and its
log-mel front-end below follow the authors' MIT-licensed upstream code
(github.com/qiuqiangkong/audioset_tagging_cnn); the front-end itself is
the upstream `torchlibrosa` implementation, so inference matches the
features the checkpoint was trained on.

Honesty contract:
- Genre probabilities are the model's REAL outputs: per-window sigmoid
  scores from the AudioSet head, averaged across windows, restricted to
  the AudioSet genre classes and reported un-renormalized. Nothing is
  rescaled, sharpened, or replaced with rank-based placeholder scores.
- AudioSet's genre taxonomy is broad: it has "Hip hop music" but no
  trap / melodic-trap / emo-rap class, so those subgenres cannot be
  named by this model. That limit is disclosed on the bundle note.
- If the top genre probability is below LOW_CONFIDENCE_THRESHOLD, the
  primary genre reads "Mixed / uncertain" and the top candidates are
  still shown — no forced confident label.
- An alternate decisive rule (top score >= 0.10 AND >= 1.5x the
  runner-up) exists behind PYTHAGORA_GENRE_DECISIVE=1 for evaluation.
  It is OFF by default; when it produces the primary label, the note
  says so explicitly.
- If the model file is absent or fails to load/run, the stage degrades
  gracefully: primary_genre=None with an explicit note. An analysis
  never crashes over genre.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "PANNs Cnn10 (AudioSet)"
MODEL_VERSION = "Cnn10_mAP=0.380 · Zenodo 3987831 · CC BY 4.0"
MODEL_ID = f"{MODEL_NAME} — {MODEL_VERSION}"

# Below this top-probability, the model has not committed to a genre in
# any meaningful sense for a multi-label AudioSet head (scores are
# independent sigmoids, not a softmax, so even a "confident" genre often
# lands in the 0.3-0.6 range; 0.35 keeps the forced-label bar honest
# without labelling everything uncertain).
LOW_CONFIDENCE_THRESHOLD = 0.35

# Alternate "decisive" primary rule, behind PYTHAGORA_GENRE_DECISIVE=1
# (default OFF — the strict rule above stays the shipped behavior).
# Under the decisive rule the primary commits when the top score is at
# least DECISIVE_MIN_PROB AND at least DECISIVE_DOMINANCE times the
# runner-up: a clear relative winner at a plausible absolute level,
# even though it never reaches the strict 0.35 bar (common for this
# head — e.g. hip-hop at 0.156 vs electronic at 0.092 on a melodic
# trap record). The genre_note always discloses which rule produced
# the primary label.
DECISIVE_MIN_PROB = 0.10
DECISIVE_DOMINANCE = 1.5


def _decisive_enabled() -> bool:
    return os.environ.get("PYTHAGORA_GENRE_DECISIVE", "") == "1"

SAMPLE_RATE = 32000
WINDOW_SECONDS = 10.0
MAX_WINDOWS = 24  # evenly spaced cap so very long tracks stay bounded
TOP_N = 5

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "Cnn10_mAP=0.380.pth"

# AudioSet genre classes: (index into the 527-class output, display
# label). Indices verified against Google's official AudioSet
# class_labels_indices.csv. This is the full genre branch of the
# ontology (music-style classes only — instruments, moods, and
# context classes like "Song" or "Background music" are excluded).
GENRE_CLASSES: list[tuple[int, str]] = [
    (216, "Pop music"),
    (217, "Hip hop music"),
    (219, "Rock music"),
    (220, "Heavy metal"),
    (221, "Punk rock"),
    (222, "Grunge"),
    (223, "Progressive rock"),
    (224, "Rock and roll"),
    (225, "Psychedelic rock"),
    (226, "Rhythm and blues"),
    (227, "Soul music"),
    (228, "Reggae"),
    (229, "Country"),
    (230, "Swing music"),
    (231, "Bluegrass"),
    (232, "Funk"),
    (233, "Folk music"),
    (234, "Middle Eastern music"),
    (235, "Jazz"),
    (236, "Disco"),
    (237, "Classical music"),
    (238, "Opera"),
    (239, "Electronic music"),
    (240, "House music"),
    (241, "Techno"),
    (242, "Dubstep"),
    (243, "Drum and bass"),
    (244, "Electronica"),
    (245, "Electronic dance music"),
    (246, "Ambient music"),
    (247, "Trance music"),
    (249, "Salsa music"),
    (251, "Blues"),
    (253, "New-age music"),
    (257, "Afrobeat"),
    (259, "Gospel music"),
    (263, "Ska"),
    (265, "Independent music"),
]

TAXONOMY_NOTE = (
    "AudioSet's genre taxonomy is broad (e.g. 'Hip hop music'); it has no "
    "trap, melodic-trap, or emo-rap class, so subgenre granularity is "
    "beyond this model."
)


def model_path() -> Path:
    """Checkpoint location: $PYTHAGORA_GENRE_MODEL, else the repo models/ dir."""
    override = os.environ.get("PYTHAGORA_GENRE_MODEL")
    return Path(override) if override else DEFAULT_MODEL_PATH


def summarize_genre_outputs(clipwise_output: np.ndarray) -> dict:
    """Map a raw 527-class AudioSet output vector to the bundle contract.

    Pure function (no torch, no model file) so the mapping, the real
    probability passthrough, and the low-confidence rule are unit
    testable. Probabilities are the model's own sigmoid scores —
    passed through untouched.
    """
    probs = np.asarray(clipwise_output, dtype=np.float64).reshape(-1)
    scored = [
        (label, float(probs[idx]))
        for idx, label in GENRE_CLASSES
        if idx < probs.shape[0]
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    predictions = [
        {"label": label, "probability": round(prob, 4)} for label, prob in scored[:TOP_N]
    ]
    if not predictions:
        return {
            "primary_genre": None,
            "genre_predictions": [],
            "genre_model": MODEL_ID,
            "genre_note": "Genre model returned no genre-class scores.",
        }
    top_label, top_prob = scored[0]
    runner_prob = scored[1][1] if len(scored) > 1 else 0.0
    if top_prob >= LOW_CONFIDENCE_THRESHOLD:
        primary = top_label
        note = TAXONOMY_NOTE
    elif (
        _decisive_enabled()
        and top_prob >= DECISIVE_MIN_PROB
        and top_prob >= DECISIVE_DOMINANCE * runner_prob
    ):
        primary = top_label
        note = (
            f"Primary genre committed under the decisive rule "
            f"(PYTHAGORA_GENRE_DECISIVE=1): top score {top_prob:.2f} is at "
            f"least {DECISIVE_MIN_PROB:.2f} and at least "
            f"{DECISIVE_DOMINANCE:g}x the runner-up score "
            f"({runner_prob:.2f}), though below the strict "
            f"{LOW_CONFIDENCE_THRESHOLD:.2f} threshold. " + TAXONOMY_NOTE
        )
    else:
        primary = "Mixed / uncertain"
        note = (
            f"Top genre score {top_prob:.2f} is below the "
            f"{LOW_CONFIDENCE_THRESHOLD:.2f} confidence threshold, so no "
            "single genre is claimed. " + TAXONOMY_NOTE
        )
    return {
        "primary_genre": primary,
        "genre_predictions": predictions,
        "genre_model": MODEL_ID,
        "genre_note": note,
    }


def _degraded(reason: str) -> dict:
    return {
        "primary_genre": None,
        "genre_predictions": [],
        "genre_model": MODEL_ID,
        "genre_note": f"Genre model unavailable: {reason}",
    }


# ---------------------------------------------------------------------------
# PANNs Cnn10 network (upstream architecture, MIT-licensed; see docstring).
# ---------------------------------------------------------------------------

def _build_model():
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torchlibrosa.augmentation import SpecAugmentation
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x, pool_size=(2, 2)):
            x = F.relu_(self.bn1(self.conv1(x)))
            x = F.relu_(self.bn2(self.conv2(x)))
            return F.avg_pool2d(x, kernel_size=pool_size)

    class Cnn10(nn.Module):
        def __init__(self, classes_num=527):
            super().__init__()
            self.spectrogram_extractor = Spectrogram(
                n_fft=1024, hop_length=320, win_length=1024, window="hann",
                center=True, pad_mode="reflect", freeze_parameters=True,
            )
            self.logmel_extractor = LogmelFilterBank(
                sr=SAMPLE_RATE, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
                ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
            )
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2
            )
            self.bn0 = nn.BatchNorm2d(64)
            self.conv_block1 = ConvBlock(1, 64)
            self.conv_block2 = ConvBlock(64, 128)
            self.conv_block3 = ConvBlock(128, 256)
            self.conv_block4 = ConvBlock(256, 512)
            self.fc1 = nn.Linear(512, 512, bias=True)
            self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        def forward(self, x):
            x = self.spectrogram_extractor(x)  # (batch, 1, time, freq)
            x = self.logmel_extractor(x)  # (batch, 1, time, mel)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            x = self.conv_block1(x)
            x = F.dropout(x, p=0.2, training=False)
            x = self.conv_block2(x)
            x = F.dropout(x, p=0.2, training=False)
            x = self.conv_block3(x)
            x = F.dropout(x, p=0.2, training=False)
            x = self.conv_block4(x)
            x = F.dropout(x, p=0.2, training=False)
            x = torch.mean(x, dim=3)
            x1, _ = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=False)
            x = F.relu_(self.fc1(x))
            return torch.sigmoid(self.fc_audioset(x))

    return Cnn10()


_MODEL = None
_MODEL_ERROR: str | None = None


def _load_model():
    """Lazy singleton. Returns (model, error_reason)."""
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None or _MODEL_ERROR is not None:
        return _MODEL, _MODEL_ERROR
    path = model_path()
    if not path.is_file():
        _MODEL_ERROR = f"model file not found at {path}"
        return None, _MODEL_ERROR
    try:
        import torch

        model = _build_model()
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
    except Exception as exc:  # noqa: BLE001 - degradation is the contract
        _MODEL_ERROR = f"failed to load {path.name} ({exc.__class__.__name__}: {exc})"
    return _MODEL, _MODEL_ERROR


def _window_starts(total_samples: int, window: int) -> list[int]:
    if total_samples <= window:
        return [0]
    last = total_samples - window
    count = min(MAX_WINDOWS, max(1, total_samples // window))
    if count == 1:
        return [0]
    return [round(i * last / (count - 1)) for i in range(count)]


def predict_genre(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Run the trained model on a mono waveform and map its real outputs.

    Returns the bundle genre dict, or a degraded dict (primary_genre
    None + explicit note) when the model is unavailable. Never raises.
    """
    model, error = _load_model()
    if model is None:
        return _degraded(error or "unknown load failure")
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
        batch = torch.from_numpy(np.stack(clips))
        with torch.no_grad():
            clipwise = model(batch).cpu().numpy()
        mean_output = clipwise.mean(axis=0)
        return summarize_genre_outputs(mean_output)
    except Exception as exc:  # noqa: BLE001 - degradation is the contract
        return _degraded(f"inference failed ({exc.__class__.__name__}: {exc})")


def analyze_genre(audio_path: str) -> dict:
    """Full genre stage for the pipeline: load audio, predict, and time it.

    Adds `genre_seconds` (wall time of the stage) to the returned dict;
    the pipeline strips it before building the bundle. Never raises.
    """
    started = time.perf_counter()
    try:
        import librosa

        y, _sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        result = predict_genre(y, SAMPLE_RATE)
    except Exception as exc:  # noqa: BLE001 - degradation is the contract
        result = _degraded(f"audio load failed ({exc.__class__.__name__}: {exc})")
    result["genre_seconds"] = round(time.perf_counter() - started, 3)
    return result
