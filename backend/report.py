"""Customer-facing analysis report renderer.

Renders a completed TelemetryBundle as a clean, branded, printable HTML
document (print → PDF). Everything shown comes from measured analysis data;
the prose summary is the pipeline's template-built essay, and heuristic
scores are labeled as heuristics. No dashboard chrome, no invented claims.
"""
from __future__ import annotations

import html
import math
from datetime import datetime, timezone

from backend.models.schemas import TelemetryBundle, TrackRecord


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes}:{rest:04.1f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{round(float(value) * 100)}%"


def _waveform_svg(peaks: list[float], width: int = 960, height: int = 120) -> str:
    if not peaks:
        return '<p class="muted">No waveform data recorded for this analysis.</p>'
    step = max(1, len(peaks) // 240)
    sampled = peaks[::step][:240]
    mid = height / 2
    points_top = []
    points_bottom = []
    n = len(sampled)
    for i, peak in enumerate(sampled):
        x = (i / max(1, n - 1)) * width
        amp = max(0.0, min(1.0, float(peak))) * (mid - 4)
        points_top.append(f"{x:.1f},{mid - amp:.1f}")
        points_bottom.append(f"{x:.1f},{mid + amp:.1f}")
    polygon = " ".join(points_top + points_bottom[::-1])
    return (
        f'<svg class="wave" viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'role="img" aria-label="Waveform envelope">'
        f'<polygon points="{polygon}" fill="#e50914" opacity="0.85"/></svg>'
    )


def _level_summary(peaks: list[float]) -> tuple[str, str]:
    """Relative envelope stats. Peaks are normalized to the track maximum,
    so these are relative figures — absolute loudness (LUFS) is not claimed."""
    if not peaks:
        return "—", "—"
    peak = max(float(p) for p in peaks) or 1.0
    mean = sum(float(p) for p in peaks) / len(peaks)
    if mean <= 0:
        return "—", "—"
    crest_db = 20 * math.log10(peak / mean)
    return f"{mean:.2f} (relative to peak 1.00)", f"{crest_db:.1f} dB envelope crest"


def render_report_html(track: TrackRecord, telemetry: TelemetryBundle) -> str:
    t = telemetry
    generated = datetime.now(timezone.utc).strftime("%B %d, %Y")

    progression: list[tuple[str, str]] = []
    seen: set[str] = set()
    for event in t.chord_timeline:
        if event.chord not in seen:
            seen.add(event.chord)
            progression.append((event.chord, event.roman))
    if not progression:
        for chord in t.chords:
            progression.append((chord, ""))

    chips = "".join(
        f'<span class="chip"><strong>{_esc(chord)}</strong>'
        + (f"<em>{_esc(roman)}</em>" if roman else "")
        + "</span>"
        for chord, roman in progression
    ) or '<span class="muted">No chord progression detected.</span>'

    timeline_rows = "".join(
        f"<tr><td>{_mmss(e.start)} – {_mmss(e.end)}</td><td>{_esc(e.chord)}</td>"
        f"<td>{_esc(e.roman)}</td><td>{_pct(e.confidence)}</td></tr>"
        for e in t.chord_timeline
    )

    segment_rows = "".join(
        f"<tr><td>{_esc(s.name)}</td><td>{_mmss(s.start)}</td><td>{_mmss(s.end)}</td></tr>"
        for s in t.segments
    )

    if t.vocal_profile is not None:
        v = t.vocal_profile
        vocal_block = f"""
        <table>
          <tr><th>Mean pitch (F0)</th><td>{v.f0_mean_hz:.1f} Hz</td></tr>
          <tr><th>Median pitch (F0)</th><td>{v.f0_median_hz:.1f} Hz</td></tr>
          <tr><th>Pitch range</th><td>{v.f0_range_semitones:.1f} semitones</td></tr>
          <tr><th>Voiced fraction</th><td>{_pct(v.voiced_fraction)}</td></tr>
          <tr><th>Vibrato rate</th><td>{v.vibrato_rate_hz:.1f} Hz</td></tr>
          <tr><th>Vibrato depth</th><td>{v.vibrato_depth_cents:.0f} cents</td></tr>
          <tr><th>Autotune likelihood (heuristic)</th><td>{_pct(v.autotune_likelihood)}</td></tr>
        </table>
        <p class="note">Vocal measurements are signal-processing estimates. The autotune
        likelihood is a <strong>heuristic score, not a verdict</strong> — it reflects how
        closely the pitch track snaps to the equal-tempered grid and how abruptly it
        moves between notes.</p>"""
    else:
        reason = t.vocal_note or "No vocal profile could be measured for this track."
        vocal_block = f"""
        <p class="refusal"><strong>Vocal forensics not reported.</strong> {_esc(reason)}</p>
        <p class="note">Pythagoras refuses to guess: when a melodic line cannot be
        separated from bass-heavy content, it reports insufficient separation instead of
        a confident wrong answer.</p>"""

    if t.genre_predictions or t.primary_genre or t.genre_note:
        pred_rows = "".join(
            f"<tr><th>{_esc(p.label)}</th><td>{p.probability * 100:.1f}%</td></tr>"
            for p in t.genre_predictions
        )
        primary_line = (
            f"<p><strong>Primary genre:</strong> {_esc(t.primary_genre)}</p>"
            if t.primary_genre
            else "<p><strong>Primary genre:</strong> not classified</p>"
        )
        model_line = (
            f'<p class="note">Model: {_esc(t.genre_model)}</p>' if t.genre_model else ""
        )
        note_line = (
            f'<p class="note">{_esc(t.genre_note)}</p>' if t.genre_note else ""
        )
        genre_block = f"""
  <section>
    <h3>Genre — trained model</h3>
    {primary_line}
    {"<table>" + pred_rows + "</table>" if pred_rows else ""}
    {model_line}
    {note_line}
    <p class="note">Probabilities are the model's own output scores
    (multi-label sigmoids averaged across the track), shown as computed —
    they are not renormalized and do not sum to 100%.</p>
  </section>"""
    else:
        genre_block = ""

    if t.subgenre_predictions or t.primary_subgenre or t.subgenre_note:
        sub_rows = "".join(
            f"<tr><th>{_esc(p.label)}</th><td>{_esc(p.family)}</td>"
            f"<td>{p.percent:.1f}%</td></tr>"
            for p in t.subgenre_predictions
        )
        sub_primary_line = (
            f"<p><strong>Primary subgenre:</strong> {_esc(t.primary_subgenre)}</p>"
            if t.primary_subgenre
            else "<p><strong>Primary subgenre:</strong> no decisive match</p>"
        )
        sub_model_line = (
            f'<p class="note">Model: {_esc(t.subgenre_model)}</p>'
            if t.subgenre_model
            else ""
        )
        sub_note_line = (
            f'<p class="note">{_esc(t.subgenre_note)}</p>' if t.subgenre_note else ""
        )
        subgenre_block = f"""
  <section>
    <h3>Subgenre matches — zero-shot similarity</h3>
    {sub_primary_line}
    {"<table><tr><th>Subgenre</th><th>Family</th><th>Similarity share</th></tr>" + sub_rows + "</table>" if sub_rows else ""}
    {sub_model_line}
    {sub_note_line}
    <p class="note">These percents are <strong>similarity shares</strong>: the
    audio is compared against a text description of each subgenre and the
    shares are normalized across the fixed candidate set. They rank
    closeness to each description — they are <strong>not</strong> trained
    classifier probabilities like the genre scores above.</p>
  </section>"""
    else:
        subgenre_block = ""

    if t.texture_cues:
        cue_rows = "".join(
            f"<tr><th>{_esc(c.name)}</th><td>{_esc(c.note)}</td></tr>" for c in t.texture_cues
        )
        texture_block = f"""
  <section>
    <h3>Texture cues</h3>
    <table>{cue_rows}</table>
    <p class="note">Texture cues come from deterministic rules over measured
    texture (tempo, percussion/harmonic balance, brightness). They describe
    how the track feels sonically — they are <strong>not</strong> genre
    classifications.</p>
  </section>"""
    else:
        texture_block = ""

    mean_level, crest = _level_summary(t.waveform_peaks)

    if t.timestamped_lyrics:
        lyric_rows = "".join(
            f"<tr><td>{_mmss(line.start)} – {_mmss(line.end)}</td><td>{_esc(line.text)}</td></tr>"
            for line in t.timestamped_lyrics
        )
        lyrics_block = f"""
  <section>
    <h3>Lyrics</h3>
    <p class="note">Lyric sheet supplied with this track — shown as provided.
    Pythagoras v1 does not transcribe audio automatically.</p>
    <table><tr><th>Time</th><th>Line</th></tr>{lyric_rows}</table>
  </section>"""
    else:
        lyrics_block = ""

    if t.suggestions:
        sev_label = {"high": "High", "medium": "Medium", "low": "Low"}
        sugg_rows = "".join(
            f'<tr><td><span class="sev sev-{_esc(s.severity)}">{_esc(sev_label.get(s.severity, s.severity))}</span></td>'
            f"<td><strong>{_esc(s.area)}.</strong> {_esc(s.suggestion)}"
            f'<br><span class="note">Why: {_esc(s.why)}</span></td></tr>'
            for s in t.suggestions
        )
        suggestions_block = f"""
  <section>
    <h3>Mix Suggestions</h3>
    <p class="note">Each suggestion is fired by a deterministic rule over the
    measurements in this report, and its "why" quotes the measured value —
    no measurement, no suggestion.</p>
    <table>{sugg_rows}</table>
  </section>"""
    else:
        suggestions_block = ""

    mix_rows = ""
    if t.mix_measurements is not None:
        m = t.mix_measurements
        stereo = (
            f"correlation {m.stereo_correlation:.2f} · width {m.stereo_width:.2f}"
            if m.is_stereo and m.stereo_correlation is not None
            else "mono source"
        )
        mix_rows = f"""
      <tr><th>True peak</th><td>{m.peak_dbfs:.1f} dBFS</td></tr>
      <tr><th>RMS level</th><td>{m.rms_dbfs:.1f} dBFS</td></tr>
      <tr><th>Band energy (sub/low/mid/high/air)</th><td>{m.sub_pct:.1f}% / {m.low_pct:.1f}% / {m.mid_pct:.1f}% / {m.high_pct:.1f}% / {m.air_pct:.1f}%</td></tr>
      <tr><th>Stereo</th><td>{stereo}</td></tr>"""
    key_label = f"{_esc(t.musical_key)} {_esc(t.mode)}"
    key_conf = (
        f"<div class='stat'><span>Key confidence</span><strong>{_pct(t.key_confidence)}</strong></div>"
        if t.key_confidence is not None
        else ""
    )
    rhyme = (
        f'<p><strong>Rhyme scheme:</strong> {_esc(t.rhyme_scheme_summary)}</p>'
        if t.rhyme_scheme_summary
        else ""
    )
    essay = _esc(t.musicological_essay).replace("\n\n", "</p><p>") or "No summary recorded."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pythagoras Analysis Report — {_esc(track.title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #0b0b0d; color: #f2f2f3;
         font-family: "Inter", "Segoe UI", system-ui, sans-serif; line-height: 1.55; }}
  .page {{ max-width: 880px; margin: 0 auto; padding: 48px 40px 64px; }}
  header.brand {{ display: flex; align-items: center; gap: 16px;
                 border-bottom: 2px solid #e50914; padding-bottom: 20px; }}
  .mark {{ width: 52px; height: 52px; border-radius: 12px; background: #e50914;
           color: #fff; font-size: 30px; font-weight: 800; display: grid;
           place-items: center; font-family: Georgia, serif; }}
  .brand h1 {{ margin: 0; font-size: 15px; letter-spacing: .28em; }}
  .brand p {{ margin: 2px 0 0; color: #a7a7ad; font-size: 12px; letter-spacing: .12em; }}
  h2.title {{ font-size: 34px; margin: 30px 0 2px; line-height: 1.15; }}
  .artist {{ color: #c9c9ce; font-size: 17px; margin: 0 0 4px; }}
  .meta {{ color: #83838a; font-size: 13px; }}
  section {{ margin-top: 34px; }}
  h3 {{ font-size: 13px; letter-spacing: .22em; text-transform: uppercase;
        color: #e50914; margin: 0 0 14px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px; }}
  .stat {{ background: #151517; border: 1px solid #26262b; border-radius: 12px;
           padding: 14px 16px; }}
  .stat span {{ display: block; color: #a7a7ad; font-size: 11px;
                letter-spacing: .14em; text-transform: uppercase; }}
  .stat strong {{ font-size: 22px; }}
  .chip {{ display: inline-flex; align-items: baseline; gap: 7px; background: #151517;
           border: 1px solid #2c2c33; border-radius: 999px; padding: 7px 14px;
           margin: 0 8px 8px 0; }}
  .chip em {{ color: #e50914; font-style: normal; font-weight: 700; font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #232329; }}
  th {{ color: #a7a7ad; font-weight: 600; width: 42%; }}
  .wave {{ width: 100%; height: 120px; background: #151517; border-radius: 12px;
           border: 1px solid #26262b; }}
  .note {{ color: #a7a7ad; font-size: 13px; }}
  .sev {{ display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .08em;
          text-transform: uppercase; padding: 3px 9px; border-radius: 999px; border: 1px solid #2c2c33; }}
  .sev-high {{ color: #ff8a80; border-color: #57202a; background: #1d1214; }}
  .sev-medium {{ color: #f5b544; border-color: #57451d; background: #1d1812; }}
  .sev-low {{ color: #a7a7ad; }}
  tr.diff td {{ background: rgba(229, 9, 20, 0.07); }}
  .muted {{ color: #83838a; }}
  .refusal {{ background: #1d1214; border: 1px solid #57202a; border-radius: 12px;
              padding: 14px 16px; }}
  footer {{ margin-top: 44px; border-top: 1px solid #232329; padding-top: 16px;
            color: #83838a; font-size: 12px; }}
  @media print {{
    body {{ background: #0b0b0d; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .page {{ padding: 24px; }}
    section {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="page">
  <header class="brand">
    <div class="mark">π</div>
    <div>
      <h1>PROJECT PYTHAGORAS</h1>
      <p>SONG ANALYSIS REPORT</p>
    </div>
  </header>

  <h2 class="title">{_esc(track.title)}</h2>
  <p class="artist">{_esc(track.artist)}</p>
  <p class="meta">Report generated {generated} · Track ID {_esc(track.id)}</p>

  <section>
    <h3>Overview</h3>
    <div class="stats">
      <div class="stat"><span>Duration</span><strong>{_mmss(t.duration)}</strong></div>
      <div class="stat"><span>Tempo</span><strong>{t.bpm:.1f} BPM</strong></div>
      <div class="stat"><span>Key</span><strong>{key_label}</strong></div>
      {key_conf}
      <div class="stat"><span>Sections</span><strong>{len(t.segments)}</strong></div>
      <div class="stat"><span>Unique chords</span><strong>{len(progression)}</strong></div>
    </div>
  </section>
{genre_block}
{subgenre_block}
{texture_block}
  <section>
    <h3>Chord Progression</h3>
    <div>{chips}</div>
    {"<table><tr><th>Time</th><th>Chord</th><th>Roman</th><th>Confidence</th></tr>" + timeline_rows + "</table>" if timeline_rows else ""}
  </section>

  {"<section><h3>Structure</h3><table><tr><th>Section</th><th>Start</th><th>End</th></tr>" + segment_rows + "</table></section>" if segment_rows else ""}

  <section>
    <h3>Vocal Forensics</h3>
    {vocal_block}
  </section>
{lyrics_block}
  <section>
    <h3>Waveform &amp; Levels</h3>
    {_waveform_svg(t.waveform_peaks)}
    <table>
      <tr><th>Average envelope level</th><td>{mean_level}</td></tr>
      <tr><th>Dynamic contrast</th><td>{crest}</td></tr>{mix_rows}
    </table>
    <p class="note">Waveform peaks are normalized to the track's own maximum, so level
    figures are relative to this recording — absolute loudness (LUFS) is not measured
    by this report.</p>
  </section>

{suggestions_block}
  <section>
    <h3>Summary</h3>
    <p>{essay}</p>
    {rhyme}
  </section>

  <footer>
    Generated by Project Pythagoras. Figures are deterministic measurements from
    the audio file, except genre, which is a trained model's output shown with
    its own probabilities, and subgenre matches, which are zero-shot similarity
    shares; scores labeled "heuristic" are estimates, not verdicts.
  </footer>
</div>
</body>
</html>"""


def _fmt_val(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def render_comparison_html(comparison: dict) -> str:
    """Branded printable comparison report (Phase E) — same visual system
    as the single-track report. Differing rows are highlighted."""
    generated = datetime.now(timezone.utc).strftime("%B %d, %Y")
    a, b = comparison["a"], comparison["b"]
    rows = "".join(
        f'<tr class="{"diff" if r["different"] else ""}"><td>{_esc(r["label"])}</td>'
        f"<td>{_esc(_fmt_val(r['a']))}</td><td>{_esc(_fmt_val(r['b']))}</td></tr>"
        for r in comparison["rows"]
    )
    chords = comparison["chords"]
    shared = ", ".join(chords["shared"]) or "none"
    only_a = ", ".join(chords["only_a"]) or "—"
    only_b = ", ".join(chords["only_b"]) or "—"
    summary = comparison["summary"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pythagoras Comparison — {_esc(a['title'])} vs {_esc(b['title'])}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #0b0b0d; color: #f2f2f3;
         font-family: "Inter", "Segoe UI", system-ui, sans-serif; line-height: 1.55; }}
  .page {{ max-width: 880px; margin: 0 auto; padding: 48px 40px 64px; }}
  header.brand {{ display: flex; align-items: center; gap: 16px;
                 border-bottom: 2px solid #e50914; padding-bottom: 20px; }}
  .mark {{ width: 52px; height: 52px; border-radius: 12px; background: #e50914;
           color: #fff; font-size: 30px; font-weight: 800; display: grid;
           place-items: center; font-family: Georgia, serif; }}
  .brand h1 {{ margin: 0; font-size: 15px; letter-spacing: .28em; }}
  .brand p {{ margin: 2px 0 0; color: #a7a7ad; font-size: 12px; letter-spacing: .12em; }}
  h2.title {{ font-size: 30px; margin: 30px 0 2px; line-height: 1.2; }}
  .meta {{ color: #83838a; font-size: 13px; }}
  section {{ margin-top: 34px; }}
  h3 {{ font-size: 13px; letter-spacing: .22em; text-transform: uppercase;
        color: #e50914; margin: 0 0 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #232329; }}
  th {{ color: #a7a7ad; font-weight: 600; }}
  tr.diff td {{ background: rgba(229, 9, 20, 0.07); }}
  .note {{ color: #a7a7ad; font-size: 13px; }}
  footer {{ margin-top: 44px; border-top: 1px solid #232329; padding-top: 16px;
            color: #83838a; font-size: 12px; }}
  @media print {{
    body {{ background: #0b0b0d; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .page {{ padding: 24px; }}
    section {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="page">
  <header class="brand">
    <div class="mark">π</div>
    <div>
      <h1>PROJECT PYTHAGORAS</h1>
      <p>TRACK COMPARISON REPORT</p>
    </div>
  </header>

  <h2 class="title">{_esc(a['title'])} <span style="color:#e50914">vs</span> {_esc(b['title'])}</h2>
  <p class="meta">{_esc(a['artist'])} · {_esc(b['artist'])} · Report generated {generated}</p>
  <p class="meta">{summary['fields_different']} of {summary['fields_compared']} compared
  measurements differ (highlighted rows).</p>

  <section>
    <h3>Measurements</h3>
    <table><tr><th>Measurement</th><th>{_esc(a['title'])}</th><th>{_esc(b['title'])}</th></tr>{rows}</table>
  </section>

  <section>
    <h3>Chord vocabulary</h3>
    <table>
      <tr><th>Shared chords</th><td>{_esc(shared)}</td></tr>
      <tr><th>Only in {_esc(a['title'])}</th><td>{_esc(only_a)}</td></tr>
      <tr><th>Only in {_esc(b['title'])}</th><td>{_esc(only_b)}</td></tr>
    </table>
    <p class="note">Every figure is a deterministic measurement from the audio
    files; scores labeled "heuristic" are estimates, not verdicts.</p>
  </section>

  <footer>Generated by Project Pythagoras.</footer>
</div>
</body>
</html>"""
