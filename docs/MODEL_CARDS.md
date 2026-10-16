# Model Cards

The two learned models in Project Pythagoras. Everything else in the
pipeline is deterministic signal processing. Both models degrade
gracefully: if the weights are absent or fail to load, the stage
returns an explicit "model unavailable" note and the rest of the
analysis completes — an analysis never crashes over a model.

## Version manifest

| Stage | Model ID (as reported on every bundle) | Weights location | License |
|---|---|---|---|
| Genre | `PANNs Cnn10 (AudioSet) — Cnn10_mAP=0.380 · Zenodo 3987831 · CC BY 4.0` | `models/Cnn10_mAP=0.380.pth` (override: `$PYTHAGORA_GENRE_MODEL`) | Checkpoint CC BY 4.0; architecture follows the authors' MIT-licensed upstream code |
| Subgenre | `LAION CLAP (clap-htsat-unfused) — laion/clap-htsat-unfused · Apache-2.0` | `models/clap-htsat/` (override: `$PYTHAGORA_SUBGENRE_MODEL`) | Apache-2.0 |

Weights are not committed to the repository (`models/` is gitignored);
they are a one-time setup download (see README, "Genre model weights" /
"Subgenre model weights"). Inference loads local files only.

---

## 1. Genre — PANNs Cnn10 (trained classifier)

**Code:** `workers/analysis/genre.py`

**Source.** PANNs Cnn10 (Kong et al., "PANNs: Large-Scale Pretrained
Audio Neural Networks for Audio Pattern Recognition", arXiv:1912.10211),
pretrained on AudioSet (527 classes). Checkpoint `Cnn10_mAP=0.380.pth`
from the authors' official Zenodo record 3987831
(md5 bfb1f1f9968938fa8ef4012b8471f5f6). The Cnn10 architecture and its
log-mel front-end follow the authors' MIT-licensed upstream code
(github.com/qiuqiangkong/audioset_tagging_cnn) and the upstream
`torchlibrosa` implementation, so inference matches the features the
checkpoint was trained on.

**Input / inference.** Mono audio resampled to 32 kHz; 10-second
windows (up to 24, evenly spaced); per-window sigmoid outputs averaged
across windows.

**What it outputs.** The model's real per-class probabilities for the
38 AudioSet genre classes, top 5 reported. Scores are passed through
untouched: multi-label sigmoids, **not** renormalized, and they do not
sum to 1. Nothing is rescaled, sharpened, or replaced with rank-based
placeholder scores.

**Primary-label rule (confidence semantics).**
- Strict: top probability ≥ 0.35 commits that genre as primary.
- Decisive (the DEFAULT; Marcus's decision, 2026-10-13): top score
  ≥ 0.10 **and** ≥ 1.5× the runner-up also commits — a clear relative
  winner at a plausible absolute level. The bundle note discloses when
  this rule produced the label. `PYTHAGORA_GENRE_DECISIVE=0` restores
  strict-only behavior.
- Below both bars the primary reads **"Mixed / uncertain"** and the
  top candidates are still shown — no forced confident label.

**Known taxonomy limit.** AudioSet's genre taxonomy is broad: it has
"Hip hop music" but no trap / melodic-trap / emo-rap class, so those
subgenres cannot be named by this model. This limit is disclosed on
every bundle's genre note.

**When weights are absent.** `analyze_genre` returns
`primary_genre=None`, empty predictions, and a note of the form
"Genre model unavailable: <reason>". It never raises.

---

## 2. Subgenre — LAION CLAP (zero-shot similarity)

**Code:** `workers/analysis/subgenre.py`

**Source.** LAION CLAP, `laion/clap-htsat-unfused` checkpoint
(HTS-AT audio encoder), Apache-2.0. Loaded with
`local_files_only=True` — loading from the local directory is a setup
step, never an analysis-time download.

**Input / inference.** Mono audio at CLAP's native 48 kHz; 10-second
windows (up to 12, evenly spaced). Each window is embedded by the
audio encoder; the 46 candidate subgenres are embedded from fixed
text descriptions (e.g. "melodic trap", "emo rap", "country") by the
text encoder. Embeddings are L2-normalized; per-window cosine
similarities to every candidate are averaged across windows.

**What it outputs.** Top 5 candidates with **similarity shares**:
softmax(logit_scale × mean similarity) over the fixed 46-candidate
set, so shares sum to 1 across all 46 candidates (only the top 5 are
reported). These are zero-shot similarity rankings — **not** trained
classifier probabilities — and are labeled as such everywhere they
appear (bundle note, dashboard, reports).

**Primary-label rule.** The top candidate commits as primary when its
share is ≥ 0.15 **or** ≥ 1.5× the runner-up's share. Below both, no
primary subgenre is claimed and the note explains why.

**When weights are absent.** `analyze_subgenre` returns
`primary_subgenre=None`, empty predictions, and a note of the form
"Subgenre model unavailable: <reason>". It never raises.
