#!/usr/bin/env python
"""Scored validation harness for the on-file catalog.

Runs the analysis pipeline's own stage functions (the same calls
workers/pipeline.py makes, minus job bookkeeping) against catalog
tracks whose expected outputs are established ground truth, and scores
each output pass/fail. Ground truth sources:

- Dirty Dyana 117.5 BPM, Way Too Long 161.5 BPM (half-time ~80.7) and
  E major: verified runs recorded 2026-10-14 (test-session catalog
  verification) — Way Too Long key confidence 0.805 on that run.
- This Side primary genre "Hip hop music": the decisive-rule default
  (workers/analysis/genre.py; Marcus's 2026-10-13 decision).
- Drowning primary genre "Mixed / uncertain" (trained PANNs model) and
  top subgenre "country" at ~26% similarity share (CLAP zero-shot):
  the artist CONFIRMED the country lean by ear on 2026-10-16, so this
  is a pass case, not an error case.

Chord timelines have no per-track ground truth, so they are reported
observationally (event count, mean/median event duration) — the table
the chord-smoothing work is measured against. When a previous run's
JSON sidecar exists, a delta section compares chord event counts.

Outputs: docs/validation-2026-10-16.md (this run's report) and
docs/validation-2026-10-16.json (machine-readable sidecar), plus the
table on stdout. Exit code 0 when every scored case passes, else 1.

Usage: ~/workspace/.venvs/pg/bin/python tools/validate_catalog.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPORT_MD = REPO_ROOT / "docs" / "validation-2026-10-16.md"
REPORT_JSON = REPO_ROOT / "docs" / "validation-2026-10-16.json"

# (title, track-id-prefix note) — first library record with this title.
SCORED_TITLES = ["Dirty Dyana", "Way Too Long", "This Side", "Drowning"]


def _catalog_paths() -> dict[str, str]:
    library = json.loads((REPO_ROOT / "storage" / "library.json").read_text())
    paths: dict[str, str] = {}
    for record in library.get("tracks", []):
        title = record.get("title")
        if title in SCORED_TITLES and title not in paths:
            paths[title] = str(REPO_ROOT / record["original_file_path"])
    return paths


def main() -> int:
    from workers.analysis import audio as audio_mod
    from workers.analysis import genre as genre_mod
    from workers.analysis import subgenre as subgenre_mod
    from workers.analysis.chords import recognize_chords_from_audio
    from workers.pipeline import _ensure_wav

    paths = _catalog_paths()
    rows: list[dict] = []
    chord_obs: dict[str, dict] = {}

    def row(track, metric, expected, got, ok, note=""):
        rows.append(
            {
                "track": track,
                "metric": metric,
                "expected": expected,
                "got": got,
                "status": "PASS" if ok else "FAIL",
                "note": note,
            }
        )

    for title in SCORED_TITLES:
        path = paths.get(title)
        if path is None or not Path(path).exists():
            rows.append(
                {
                    "track": title,
                    "metric": "audio file",
                    "expected": "present in storage/audio",
                    "got": "missing",
                    "status": "SKIP",
                    "note": "no catalog audio for this title; not scored",
                }
            )
            continue
        wav = _ensure_wav(path)
        y, sr = audio_mod.load_audio(wav)
        tempo = audio_mod.estimate_tempo(y, sr)
        key = audio_mod.estimate_key(y, sr)
        key_name, mode = (key["key"] or "C major").rsplit(" ", 1)

        # --- tempo / key ground truth ---
        if title == "Dirty Dyana":
            ok = tempo is not None and abs(tempo - 117.5) <= 2.0
            row(title, "tempo (BPM)", "117.5 ±2.0", tempo, ok)
        if title == "Way Too Long":
            full = tempo is not None and abs(tempo - 161.5) <= 3.0
            half = tempo is not None and abs(tempo - 80.7) <= 1.5
            note = "full-time reading" if full else ("half-time reading (accepted alternate)" if half else "")
            row(title, "tempo (BPM)", "161.5 ±3.0 (or half-time 80.7 ±1.5)", tempo, full or half, note)
            row(title, "key", "E major", key["key"], key["key"] == "E major")
            conf = key.get("confidence")
            okc = conf is not None and abs(conf - 0.805) <= 0.05
            row(title, "key confidence", "0.805 ±0.05", conf, okc, "verified-run value, 2026-10-14")

        # --- chords (observational; no per-track ground truth) ---
        try:
            y_harm, _ = audio_mod.hpss_waveform(y)
        except Exception:
            y_harm = None
        events = recognize_chords_from_audio(
            y, sr, key=key_name, mode=mode, y_harm=y_harm
        )
        durations = [e["end"] - e["start"] for e in events]
        chord_obs[title] = {
            "events": len(events),
            "mean_event_seconds": round(float(np.mean(durations)), 2) if durations else 0.0,
            "median_event_seconds": round(float(np.median(durations)), 2) if durations else 0.0,
            "unique_chords": len({e["chord"] for e in events}),
        }

        # --- genre / subgenre ground truth ---
        if title == "This Side":
            genre = genre_mod.analyze_genre(wav)
            row(
                title,
                "primary genre (decisive rule)",
                "Hip hop music",
                genre.get("primary_genre"),
                genre.get("primary_genre") == "Hip hop music",
            )
        if title == "Drowning":
            genre = genre_mod.analyze_genre(wav)
            row(
                title,
                "primary genre (trained model)",
                "Mixed / uncertain",
                genre.get("primary_genre"),
                genre.get("primary_genre") == "Mixed / uncertain",
            )
            sub = subgenre_mod.analyze_subgenre(wav)
            preds = sub.get("subgenre_predictions") or []
            top = preds[0] if preds else {}
            top_label = top.get("label")
            top_pct = top.get("percent")
            row(
                title,
                "top subgenre (CLAP zero-shot)",
                "country",
                top_label,
                isinstance(top_label, str) and "country" in top_label.lower(),
                "artist-confirmed by ear 2026-10-16",
            )
            okp = top_pct is not None and abs(top_pct - 26.0) <= 6.0
            row(title, "top subgenre share (%)", "26 ±6", top_pct, okp, "similarity share, not a probability")

    scored = [r for r in rows if r["status"] in ("PASS", "FAIL")]
    passed = sum(1 for r in scored if r["status"] == "PASS")

    # --- delta vs previous run (chord observations) ---
    previous = None
    if REPORT_JSON.exists():
        try:
            previous = json.loads(REPORT_JSON.read_text())
        except Exception:
            previous = None

    # --- report ---
    lines = [
        "# Catalog validation — 2026-10-16",
        "",
        "Generated by `tools/validate_catalog.py`. Stage functions are the",
        "pipeline's own (workers/pipeline.py minus job bookkeeping); genre",
        "uses the decisive-rule default (PYTHAGORA_GENRE_DECISIVE unset).",
        "",
        "## Scored ground truth",
        "",
        "| Track | Metric | Expected | Got | Result | Note |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['track']} | {r['metric']} | {r['expected']} | {r['got']} | {r['status']} | {r['note']} |"
        )
    lines += [
        "",
        f"**Aggregate: {passed}/{len(scored)} scored cases passed.**",
        "",
        "## Chord timelines (observational — no per-track ground truth)",
        "",
        "| Track | Events | Mean event (s) | Median event (s) | Unique chords |",
        "|---|---|---|---|---|",
    ]
    for title, obs in chord_obs.items():
        lines.append(
            f"| {title} | {obs['events']} | {obs['mean_event_seconds']} | "
            f"{obs['median_event_seconds']} | {obs['unique_chords']} |"
        )
    if previous and previous.get("chords"):
        lines += [
            "",
            "## Chord event delta vs previous run",
            "",
            "| Track | Events before | Events after | Mean event (s) before | Mean event (s) after |",
            "|---|---|---|---|---|",
        ]
        for title, obs in chord_obs.items():
            prev = previous["chords"].get(title)
            if prev:
                lines.append(
                    f"| {title} | {prev['events']} | {obs['events']} | "
                    f"{prev['mean_event_seconds']} | {obs['mean_event_seconds']} |"
                )
    lines.append("")
    report = "\n".join(lines)

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report)
    REPORT_JSON.write_text(
        json.dumps(
            {"rows": rows, "chords": chord_obs, "passed": passed, "scored": len(scored)},
            indent=2,
        )
    )
    print(report)
    print(f"\nWrote {REPORT_MD} and {REPORT_JSON}")
    return 0 if passed == len(scored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
