"""Deterministic audio measurement workers.

chords:   CQT-chroma chord recognition + Roman numerals (deterministic).
vocals:   F0 tracking, vibrato analysis, pitch-correction heuristics.
waveform: downsampled peak envelopes for the dashboard timeline.
"""
from workers.analysis.chords import recognize_chords, roman_numeral
from workers.analysis.vocals import analyze_vocals
from workers.analysis.waveform import waveform_peaks

__all__ = ["recognize_chords", "roman_numeral", "analyze_vocals", "waveform_peaks"]
