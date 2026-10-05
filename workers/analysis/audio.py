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


def load_audio(path, sr=22050, max_seconds=300):
    import librosa

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
            if r > bestr:
                bestr, best = r, f"{PITCH_NAMES[i]} {name}"
    return {"key": best, "confidence": round(_clip01((bestr + 1) / 2), 3)}


def core_features(y, sr):
    import librosa

    dur = len(y) / sr
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    total = float(S.sum()) + 1e-9
    sub_share = float(S[(freqs >= 25) & (freqs <= 80)].sum() / total * 100)

    H, P = librosa.effects.hpss(y)
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


def section_map(y, sr, window=15):
    """Split into windows, describe each, and cluster into Peak/Build/Mellow."""
    import librosa

    dur = len(y) / sr
    sections = []
    for start in range(0, int(dur), window):
        seg = y[int(start * sr): int(min(start + window, dur) * sr)]
        if len(seg) < sr * 3:
            continue
        e = float(np.sqrt(np.mean(seg ** 2)))
        c = float(librosa.feature.spectral_centroid(y=seg, sr=sr).mean())
        _, Pp = librosa.effects.hpss(seg)
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
    minor = bool(key and key.get("key", "").endswith("minor"))
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
