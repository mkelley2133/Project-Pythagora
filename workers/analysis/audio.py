"""Audio analysis engine for Project Pythagoras (ported from the local Pythagora app).

Always uses librosa. If essentia is installed it contributes extra
rhythm/key descriptors. Instrument labels are heuristic texture estimates
computed from the mix -- not stem separation -- and are labeled as such.
"""
from __future__ import annotations

import numpy as np

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles
MAJOR_PROF = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROF = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def load_audio(path, sr=22050, max_seconds=900):
    import librosa

    # The cap matches the upload duration cap (PYTHAGORA_MAX_DURATION_SECONDS,
    # default 900s): every pipeline stage must measure the SAME audio. This
    # loader previously capped at 300s while chords/vocals/genre loaded the
    # full file, so a 301-900s upload produced a bundle whose stated
    # duration and sections covered 300s while its chord timeline ran to
    # the true end.
    y, _ = librosa.load(path, sr=sr, mono=True, duration=max_seconds)
    return y, sr


def estimate_tempo(y, sr):
    import librosa

    dur = len(y) / sr
    estimates = []
    for start in (0, 30, 60, 90):
        if start >= dur:
            break
        seg = y[int(start * sr): int(min(start + 30, dur) * sr)]
        if len(seg) < sr * 5:
            continue
        t, _ = librosa.beat.beat_track(y=seg, sr=sr)
        t = float(np.asarray(t).flat[0])
        if 40 <= t <= 220:
            estimates.append(t)
    if not estimates:
        return None
    return round(float(np.median(estimates)), 1)


def estimate_key(y, sr):
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    m = chroma.mean(axis=1)
    best, bestr = None, -2.0
    for i in range(12):
        for name, prof in (("major", MAJOR_PROF), ("minor", MINOR_PROF)):
            r = float(np.corrcoef(np.roll(m, -i), prof)[0, 1])
            if np.isfinite(r) and r > bestr:
                bestr, best = r, f"{PITCH_NAMES[i]} {name}"
    if best is None:
        # No finite correlation exists — digital silence (zero-energy
        # chroma) makes every correlation NaN. Report the key as
        # undetermined honestly instead of inventing one; callers that
        # require a key must handle None.
        return {"key": None, "confidence": 0.0}
    return {"key": best, "confidence": round(_clip01((bestr + 1) / 2), 3)}


# ---------------------------------------------------------------------------
# HPSS with an exact fast median filter.
#
# librosa.effects.hpss spends nearly all of its time in scipy's generic
# median_filter (a rank filter): ~13 s for a 3-minute track on this
# machine, and the pipeline calls HPSS once per track (core features),
# once per 15 s section (section map), and once for chord chroma. The
# filters HPSS needs are 1-D sliding medians of window 31 — along time
# for the harmonic mask, along frequency for the percussive mask — over
# the magnitude spectrogram. The numba kernel below computes exactly
# those medians (same window; same edge convention as scipy.ndimage
# mode="reflect", where index -1 maps to sample 0), maintaining a sorted
# window with insert/remove steps. It is verified bit-identical to
# scipy.ndimage.median_filter on real spectrograms (see
# tests/test_hpss_fast.py), so every downstream number is unchanged —
# only the filter engine differs. STFT, softmask (librosa.util.softmask
# with librosa's power/margin defaults), and ISTFT are librosa's own.
# If numba is unavailable, this falls back to librosa.effects.hpss.
# ---------------------------------------------------------------------------

_HPSS_KERNEL = 31


def _sliding_median_kernel():
    """Compile (once) the numba sliding-median kernel; None if no numba."""
    try:
        from numba import njit
    except Exception:  # pragma: no cover - numba ships with librosa
        return None

    @njit
    def _median_axis0(a, k, out):
        # Sliding median along axis 0 of a float64 matrix, one column at
        # a time, edges reflected scipy-style (-1 -> 0, n -> n-1).
        rows, cols = a.shape
        half = k // 2
        buf = np.empty(k, dtype=np.float64)
        for j in range(cols):
            for t in range(k):
                idx = t - half
                if idx < 0:
                    idx = -idx - 1
                buf[t] = a[idx, j]
            srt = np.sort(buf)
            out[0, j] = srt[half]
            for i in range(1, rows):
                drop_idx = i - 1 - half
                if drop_idx < 0:
                    drop_idx = -drop_idx - 1
                add_idx = i + half
                if add_idx >= rows:
                    add_idx = 2 * rows - 1 - add_idx
                drop = a[drop_idx, j]
                add = a[add_idx, j]
                pos = 0
                while srt[pos] < drop:
                    pos += 1
                while pos < k - 1 and srt[pos] != drop:
                    pos += 1
                for t in range(pos, k - 1):
                    srt[t] = srt[t + 1]
                pos = k - 2
                while pos >= 0 and srt[pos] > add:
                    srt[pos + 1] = srt[pos]
                    pos -= 1
                srt[pos + 1] = add
                out[i, j] = srt[half]

    return _median_axis0


_MEDIAN_KERNEL = None
_MEDIAN_KERNEL_TRIED = False


def _median_filter_axis(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Exact 1-D sliding median of `a` along `axis` (scipy reflect edges)."""
    global _MEDIAN_KERNEL, _MEDIAN_KERNEL_TRIED
    if not _MEDIAN_KERNEL_TRIED:
        _MEDIAN_KERNEL_TRIED = True
        _MEDIAN_KERNEL = _sliding_median_kernel()
    if _MEDIAN_KERNEL is None or a.shape[axis] < 3:
        from scipy.ndimage import median_filter

        size = [1] * a.ndim
        size[axis] = k
        return median_filter(a, size=size, mode="reflect")
    work = np.ascontiguousarray(a, dtype=np.float64)
    if axis == 1:
        work = work.T
    out = np.empty_like(work)
    _MEDIAN_KERNEL(work, k, out)
    if axis == 1:
        out = out.T
    return out.astype(a.dtype, copy=False)


def hpss_waveform(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(harmonic, percussive) waveforms — librosa.effects.hpss(y) with
    default parameters, computed with the fast exact median filters."""
    import librosa

    try:
        stft = librosa.core.stft(y)
        mag, phase = librosa.core.magphase(stft)
        harm_med = _median_filter_axis(mag, _HPSS_KERNEL, axis=1)
        perc_med = _median_filter_axis(mag, _HPSS_KERNEL, axis=0)
        mask_harm = librosa.util.softmask(harm_med, perc_med, power=2.0, split_zeros=True)
        mask_perc = librosa.util.softmask(perc_med, harm_med, power=2.0, split_zeros=True)
        y_harm = librosa.core.istft(
            mag * mask_harm * phase, dtype=y.dtype, length=y.shape[-1]
        )
        y_perc = librosa.core.istft(
            mag * mask_perc * phase, dtype=y.dtype, length=y.shape[-1]
        )
        return y_harm, y_perc
    except Exception:
        return librosa.effects.hpss(y)


def core_features(y, sr, hpss_pair=None):
    import librosa

    dur = len(y) / sr
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    total = float(S.sum()) + 1e-9
    sub_share = float(S[(freqs >= 25) & (freqs <= 80)].sum() / total * 100)

    # `hpss_pair` lets the pipeline share one HPSS across stages; the
    # values are identical to computing it here (same function, same y).
    H, P = hpss_pair if hpss_pair is not None else hpss_waveform(y)
    rms_h = float(np.sqrt(np.mean(H ** 2)))
    rms_p = float(np.sqrt(np.mean(P ** 2)))
    perc_share = rms_p / (rms_h + rms_p + 1e-9)

    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    onset_density = float(len(onsets) / max(dur, 1e-6))
    energy = float(np.sqrt(np.mean(y ** 2)))

    return {
        "duration": round(dur, 1),
        "sub_bass_share": round(sub_share, 2),
        "percussive_share": round(perc_share, 3),
        "harmonic_share": round(1 - perc_share, 3),
        "spectral_centroid": round(centroid, 1),
        "spectral_rolloff": round(rolloff, 1),
        "zero_crossing_rate": round(zcr, 4),
        "spectral_contrast": [round(float(c), 2) for c in contrast],
        "onset_density": round(onset_density, 2),
        "energy": round(energy, 4),
    }


def section_map(y, sr, window=15, perc_waveform=None):
    """Split into windows, describe each, and cluster into Peak/Build/Mellow.

    `perc_waveform` may carry the full track's HPSS percussive component
    (same y, same HPSS) — sections then slice it instead of re-running
    HPSS per window. NOTE: sliced values can differ from per-window HPSS
    by HPSS boundary effects at section edges (bounded and checked in
    tests/test_shared_hpss.py); the default per-window form remains for
    standalone callers.
    """
    import librosa

    dur = len(y) / sr
    sections = []
    for start in range(0, int(dur), window):
        seg = y[int(start * sr): int(min(start + window, dur) * sr)]
        if len(seg) < sr * 3:
            continue
        e = float(np.sqrt(np.mean(seg ** 2)))
        c = float(librosa.feature.spectral_centroid(y=seg, sr=sr).mean())
        if perc_waveform is not None:
            Pp = perc_waveform[int(start * sr): int(min(start + window, dur) * sr)]
        else:
            _, Pp = hpss_waveform(seg)
        pr = float(np.sqrt(np.mean(Pp ** 2)) / (np.sqrt(np.mean(seg ** 2)) + 1e-9))
        sections.append(
            {
                "start": start,
                "end": min(start + window, int(dur)),
                "energy": round(e, 4),
                "centroid": round(c, 1),
                "percussive_share": round(pr, 3),
            }
        )
    # Unsupervised ML: K-Means over section descriptors -> section roles
    try:
        from sklearn.cluster import KMeans

        if len(sections) >= 3:
            X = np.array(
                [[s["energy"], s["centroid"] / 8000.0, s["percussive_share"]] for s in sections]
            )
            k = min(3, len(sections))
            km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(X)
            # Order clusters by mean energy -> role names
            order = np.argsort(km.cluster_centers_[:, 0])
            roles = ["Mellow / breakdown", "Build / transition", "Peak energy"][-k:]
            role_of = {int(cluster): roles[rank] for rank, cluster in enumerate(order)}
            for s, lab in zip(sections, km.labels_):
                s["role"] = role_of[int(lab)]
    except Exception:
        pass
    for s in sections:
        s.setdefault("role", "—")
    return sections


def instrument_estimates(feats):
    """Heuristic presence scores (0-100) for instrument textures in the mix."""
    perc = feats["percussive_share"]
    harm = feats["harmonic_share"]
    onset = feats["onset_density"]
    sub = feats["sub_bass_share"]
    bright = _clip01(feats["spectral_centroid"] / 8000.0)

    onset_n = _clip01(onset / 6.0)
    sub_n = _clip01(sub / 10.0)

    cands = [
        (
            "Drums / percussion",
            0.55 * perc + 0.45 * onset_n,
            f"percussive share {perc:.0%} with {onset:.1f} onsets/sec",
        ),
        (
            "808 / sub bass",
            0.70 * sub_n + 0.30 * (1 - bright),
            f"sub-bass (25-80 Hz) is {sub:.1f}% of spectral energy",
        ),
        (
            "Keys / piano-like harmony",
            0.70 * harm + 0.30 * (1 - onset_n),
            f"harmonic share {harm:.0%} with steady pitched content",
        ),
        (
            "Plucked strings / guitar-like",
            0.55 * harm + 0.45 * onset_n,
            f"harmonic share {harm:.0%} with clear attacks ({onset:.1f}/s)",
        ),
        (
            "Pads / strings-like wash",
            0.60 * harm + 0.40 * (1 - _clip01(onset / 3.0)),
            f"sustained harmonic content, low attack density",
        ),
        (
            "Synth / lead texture",
            0.50 * bright + 0.50 * (1 - perc),
            f"brightness {feats['spectral_centroid']:.0f} Hz centroid, smooth lead band",
        ),
        (
            "Vocal-like lead",
            0.60 * harm + 0.40 * (1 - bright * 0.5),
            f"midrange harmonic energy consistent with a lead voice/instrument",
        ),
    ]
    out = [
        {"instrument": name, "score": int(round(_clip01(s) * 100)), "evidence": ev}
        for name, s, ev in cands
    ]
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def style_tags(feats, tempo, key):
    tags = []

    def add(tag, reason):
        tags.append({"tag": tag, "reason": reason})

    t = tempo or 0
    perc = feats["percussive_share"]
    sub = feats["sub_bass_share"]
    bright = feats["spectral_centroid"]
    if t >= 135 and sub >= 3 and perc >= 0.35:
        add("Trap / modern hip-hop", f"{t:.0f} BPM, heavy sub ({sub:.1f}%) and punchy drums")
    elif 118 <= t <= 132 and perc >= 0.45:
        add("House / dance-pop", f"{t:.0f} BPM with driving percussion ({perc:.0%})")
    elif t >= 150:
        add("Fast / high-energy", f"{t:.0f} BPM")
    elif t and t < 95 and perc < 0.45 and feats["harmonic_share"] > 0.5:
        add("Lo-fi / chill", f"{t:.0f} BPM, mellow harmonic bed")
    if perc >= 0.55 and 85 <= t <= 105:
        add("Boom-bap / classic hip-hop", f"{t:.0f} BPM, drum-forward mix")
    if bright > 4500 and t and t >= 115:
        add("Bright pop energy", f"high spectral centroid ({bright:.0f} Hz)")
    if feats["harmonic_share"] >= 0.65 and perc < 0.35:
        add("Ambient / pads", "dominant sustained harmonic texture")
    if not tags:
        add("Eclectic / unclassified", "doesn't strongly match the built-in style rules")
    return tags[:4]


def mood_label(feats, tempo, key):
    t = tempo or 100
    arousal = 0.4 * _clip01((t - 60) / 120) + 0.3 * _clip01(feats["energy"] * 8) + 0.3 * _clip01(
        feats["spectral_centroid"] / 8000
    )
    minor = bool(key and (key.get("key") or "").endswith("minor"))
    valence = 0.5 * feats["harmonic_share"] + (0.2 if not minor else 0.0) + 0.3 * _clip01(
        feats["spectral_centroid"] / 6000
    )
    if arousal >= 0.6 and valence < 0.45:
        label = "Intense / aggressive"
    elif arousal >= 0.6:
        label = "Upbeat / euphoric"
    elif arousal < 0.4 and valence < 0.45:
        label = "Dark / brooding"
    elif arousal < 0.4:
        label = "Mellow / reflective"
    else:
        label = "Balanced / mid-tempo"
    return {"mood": label, "arousal": round(arousal, 2), "valence": round(valence, 2)}


def essentia_extras(path):
    """Optional essentia descriptors; returns {} when essentia isn't installed."""
    try:
        import essentia.standard as es
    except ImportError:
        return {}
    try:
        loader = es.MonoLoader(filename=path)
        audio = loader()
        rhythm = es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm(audio)
        keyx = es.KeyExtractor()
        key, scale, strength = keyx(audio)
        return {
            "bpm": round(float(bpm), 1),
            "key": f"{key} {scale}",
            "key_strength": round(float(strength), 3),
        }
    except Exception as e:  # pragma: no cover - best effort
        return {"error": str(e)}


def analyze_file(path):
    y, sr = load_audio(path)
    tempo = estimate_tempo(y, sr)
    key = estimate_key(y, sr)
    feats = core_features(y, sr)
    sections = section_map(y, sr)
    result = {
        **feats,
        "tempo_bpm": tempo,
        "key": key,
        "instruments": instrument_estimates(feats),
        "style_tags": style_tags(feats, tempo, key),
        "mood": mood_label(feats, tempo, key),
        "sections": sections,
        "method_notes": (
            "librosa feature extraction + K-Means section clustering"
            + (" + essentia descriptors" if essentia_extras(path) else "")
            + "; instrument scores are mix-level heuristics, not stem separation."
        ),
    }
    ess = essentia_extras(path)
    if ess:
        result["essentia"] = ess
    return result
