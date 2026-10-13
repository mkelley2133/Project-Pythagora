"""Rule-fired mix suggestions (Phase G) — zero-hallucination by design.

Every suggestion is a deterministic rule over measured values, and its
`why` quotes the measured number that fired it. If an axis has no reliable
measurement in the context, no rule fires for it — there is no generic
filler here ("consider EQing your vocals" without a measured trigger is
forbidden by the project's ethos).

Thresholds live in this file so the test-suite can pin behavior to them.
"""
from __future__ import annotations

# --- thresholds (measured axis -> trip point) ---
SUB_HEAVY_PCT = 12.0        # sub band (20-60 Hz) share of spectral energy
LOW_HEAVY_PCT = 38.0        # low band (60-250 Hz) share
HIGH_HEAVY_PCT = 30.0       # high band (2-8 kHz) share
AIR_DULL_PCT = 1.5          # air band (8 kHz+) share, below = dull top
AIR_MIN_DURATION_S = 10.0   # don't judge air on tiny clips
CLIP_HEADROOM_DB = -0.3     # peak at/above this = effectively no headroom
CREST_SQUASHED_DB = 6.0     # peak-to-RMS below this = heavily limited
CREST_WIDE_DB = 20.0        # peak-to-RMS above this = very wide dynamics
PERC_DOMINANT = 0.70        # percussive share of HPSS energy
HARM_DOMINANT = 0.85        # harmonic share of HPSS energy
AUTOTUNE_NOTICE = 0.75      # autotune-likelihood heuristic (labeled as such)
VOICED_SPARSE = 0.30        # voiced fraction at/below this, when a profile exists
KEY_LOW_CONFIDENCE = 0.40   # key confidence below this = ambiguous tonal center
LOOP_MAX_UNIQUE_CHORDS = 4  # whole-track chord vocabulary at/below this...
LOOP_MIN_EVENTS = 6         # ...with at least this many chord events...
LOOP_MIN_DURATION_S = 45.0  # ...over at least this much track
MONO_CORR = 0.98            # stereo file, correlation at/above = effectively mono
PHASE_RISK_CORR = 0.20      # correlation below this = mono-compat risk

MAX_SUGGESTIONS = 8
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def build_suggestions(ctx: dict) -> list[dict]:
    """Fire rules over a measurement context.

    ctx keys (all optional; missing = axis not measured = no rule):
      mix              — workers/analysis/measure.py output dict
      feats            — audio.core_features dict (percussive/harmonic shares)
      vocal_profile    — VocalProfile dict or None
      key_confidence   — float 0..1 or None
      unique_chords    — list[str]
      chord_events     — int
      duration         — float seconds
    """
    out: list[dict] = []

    def add(area: str, suggestion: str, why: str, severity: str) -> None:
        out.append(
            {"area": area, "suggestion": suggestion, "why": why, "severity": severity}
        )

    mix = ctx.get("mix") or {}
    duration = float(ctx.get("duration") or 0.0)

    # ---------------------------------------------------------- spectral
    sub = mix.get("sub_pct")
    if sub is not None and sub > SUB_HEAVY_PCT:
        add(
            "Low end",
            "Sub-bass is crowding the mix — check 808 vs kick masking and "
            "try high-passing the non-bass elements.",
            f"Sub-bass (20–60 Hz) is {sub:.1f}% of total spectral energy, "
            f"above the {SUB_HEAVY_PCT:.0f}% threshold.",
            "high",
        )
    low = mix.get("low_pct")
    if low is not None and low > LOW_HEAVY_PCT:
        add(
            "Low end",
            "Low-band weight is heavy — listen for low-mid buildup and "
            "consider gentle cuts where elements stack up.",
            f"The 60–250 Hz band holds {low:.1f}% of total spectral energy, "
            f"above the {LOW_HEAVY_PCT:.0f}% threshold.",
            "medium",
        )
    high = mix.get("high_pct")
    if high is not None and high > HIGH_HEAVY_PCT:
        add(
            "Top end",
            "The presence region is hot — check for harshness around "
            "2–8 kHz, especially on loud playback systems.",
            f"The 2–8 kHz band holds {high:.1f}% of total spectral energy, "
            f"above the {HIGH_HEAVY_PCT:.0f}% threshold.",
            "medium",
        )
    air = mix.get("air_pct")
    if air is not None and air < AIR_DULL_PCT and duration >= AIR_MIN_DURATION_S:
        add(
            "Top end",
            "The top octave reads dull — a gentle high-shelf lift or "
            "brighter source takes could open the mix up.",
            f"Air (8 kHz and up) is only {air:.1f}% of total spectral energy, "
            f"below the {AIR_DULL_PCT}% threshold.",
            "medium",
        )

    # ------------------------------------------------------------ levels
    peak = mix.get("peak_dbfs")
    if peak is not None and peak >= CLIP_HEADROOM_DB:
        add(
            "Levels",
            "Leave mastering headroom — pull the mix bus down a few dB "
            "before any limiting or export.",
            f"True peak is {peak:.1f} dBFS, within 0.3 dB of full scale "
            f"(clipping territory).",
            "high",
        )
    crest = mix.get("crest_db")
    if crest is not None and crest < CREST_SQUASHED_DB:
        add(
            "Dynamics",
            "Dynamics are heavily squashed — ease off bus compression or "
            "limiting if the transients feel flattened.",
            f"Peak-to-RMS distance is only {crest:.1f} dB, below the "
            f"{CREST_SQUASHED_DB:.0f} dB threshold.",
            "medium",
        )
    elif crest is not None and crest > CREST_WIDE_DB:
        add(
            "Dynamics",
            "Dynamics are very wide — quiet passages may disappear on "
            "small speakers; check the quietest sections at low volume.",
            f"Peak-to-RMS distance is {crest:.1f} dB, above the "
            f"{CREST_WIDE_DB:.0f} dB threshold.",
            "low",
        )

    # ------------------------------------------------- texture (HPSS)
    feats = ctx.get("feats") or {}
    perc = feats.get("percussive_share")
    if perc is not None and perc > PERC_DOMINANT:
        add(
            "Balance",
            "Percussion dominates the texture — check that melodic and "
            "vocal elements aren't being crowded out.",
            f"Percussive content is {float(perc):.0%} of the harmonic/"
            f"percussive energy split, above the {PERC_DOMINANT:.0%} threshold.",
            "low",
        )
    harm = feats.get("harmonic_share")
    if harm is not None and harm > HARM_DOMINANT:
        add(
            "Balance",
            "The mix is almost entirely harmonic wash — a little rhythmic "
            "definition could add contrast and movement.",
            f"Harmonic content is {float(harm):.0%} of the harmonic/"
            f"percussive energy split, above the {HARM_DOMINANT:.0%} threshold.",
            "low",
        )

    # ------------------------------------------------------------- vocal
    vocal = ctx.get("vocal_profile")
    if vocal:
        autotune = vocal.get("autotune_likelihood")
        if autotune is not None and float(autotune) > AUTOTUNE_NOTICE:
            add(
                "Vocals",
                "Pitch correction reads strong — if a natural vocal was "
                "intended, ease the correction speed or amount.",
                f"The autotune-likelihood heuristic reads {float(autotune):.2f} "
                f"(0–1), above the {AUTOTUNE_NOTICE} threshold. Heuristic, not a verdict.",
                "low",
            )
        voiced = vocal.get("voiced_fraction")
        if voiced is not None and 0.0 < float(voiced) <= VOICED_SPARSE:
            add(
                "Vocals",
                "Voiced content is sparse — if the vocal is the focus, "
                "consider tightening long instrumental stretches.",
                f"Voiced content occupies {float(voiced):.0%} of the track, "
                f"at or below the {VOICED_SPARSE:.0%} threshold.",
                "low",
            )

    # ------------------------------------------------------ key context
    key_conf = ctx.get("key_confidence")
    if key_conf is not None and float(key_conf) < KEY_LOW_CONFIDENCE:
        add(
            "Harmony",
            "The tonal center reads ambiguous — verify the key by ear "
            "before building parts or tuning to it.",
            f"Key-detection confidence is {float(key_conf):.2f} (0–1), "
            f"below the {KEY_LOW_CONFIDENCE} threshold.",
            "low",
        )

    # --------------------------------------------- chord repetitiveness
    unique = ctx.get("unique_chords") or []
    events = int(ctx.get("chord_events") or 0)
    if (
        0 < len(unique) <= LOOP_MAX_UNIQUE_CHORDS
        and events >= LOOP_MIN_EVENTS
        and duration >= LOOP_MIN_DURATION_S
    ):
        add(
            "Harmony",
            "The progression never leaves its loop — consider a bridge or "
            "pre-chorus variation to reset the ear.",
            f"Only {len(unique)} unique chords ({', '.join(unique)}) across "
            f"{events} chord events and {duration:.0f}s of track.",
            "low",
        )

    # ------------------------------------------------------------ stereo
    if mix.get("is_stereo"):
        corr = mix.get("stereo_correlation")
        if corr is not None and corr >= MONO_CORR:
            add(
                "Stereo",
                "The stereo file is effectively mono — check for a "
                "collapsed bounce or add width if width was intended.",
                f"Inter-channel correlation is {corr:.2f}, at or above the "
                f"{MONO_CORR} near-mono threshold.",
                "medium",
            )
        elif corr is not None and corr < PHASE_RISK_CORR:
            add(
                "Stereo",
                "Check mono compatibility — fold the mix to mono and "
                "listen for content that thins out or disappears.",
                f"Inter-channel correlation is only {corr:.2f}, below the "
                f"{PHASE_RISK_CORR} mono-compatibility threshold.",
                "medium",
            )

    out.sort(key=lambda s: (_SEVERITY_RANK.get(s["severity"], 3), s["area"]))
    return out[:MAX_SUGGESTIONS]
