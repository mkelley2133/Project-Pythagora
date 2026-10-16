"""Tests for the optional LLM narrative (backend/narrative.py).

Both paths are covered with the HTTP call mocked — no network, no key:
- No key in the environment -> the deterministic essay, byte-identical,
  and httpx.post is never called.
- Key present + mocked success -> the LLM text is used, and the request
  provably carries telemetry values plus the no-invention instruction.
- Key present + mocked failure -> deterministic fallback.
"""
from __future__ import annotations

import httpx
import pytest

from backend import narrative
from backend.models.schemas import (
    ChordEvent,
    GenrePrediction,
    SubgenrePrediction,
    TelemetryBundle,
    TrackRecord,
)

ESSAY = "Measured analysis of 'Way Too Long': 161.5 BPM in E major."


def _track() -> TrackRecord:
    return TrackRecord(
        id="track-test",
        title="Way Too Long",
        artist="Bl3mish",
        original_file_path="storage/audio/track-test.m4a",
    )


def _bundle() -> TelemetryBundle:
    return TelemetryBundle(
        bpm=161.5,
        duration=157.0,
        musical_key="E",
        mode="major",
        key_confidence=0.805,
        primary_genre="Hip hop music",
        genre_predictions=[GenrePrediction(label="Hip hop music", probability=0.156)],
        primary_subgenre="melodic trap",
        subgenre_predictions=[
            SubgenrePrediction(label="melodic trap", family="Trap", percent=22.4)
        ],
        chords=["E", "B", "C#m", "A"],
        chord_timeline=[
            ChordEvent(chord="E", roman="I", start=0.0, end=4.0, confidence=0.9),
            ChordEvent(chord="B", roman="V", start=4.0, end=8.0, confidence=0.88),
            ChordEvent(chord="C#m", roman="vi", start=8.0, end=12.0, confidence=0.87),
            ChordEvent(chord="A", roman="IV", start=12.0, end=16.0, confidence=0.86),
        ],
        musicological_essay=ESSAY,
    )


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_no_key_returns_deterministic_essay_and_never_calls_http(monkeypatch):
    monkeypatch.delenv(narrative.API_KEY_ENV, raising=False)

    def _forbidden(*args, **kwargs):
        raise AssertionError("httpx.post must not be called without a key")

    monkeypatch.setattr(narrative.httpx, "post", _forbidden)
    text, source = narrative.build_narrative(_track(), _bundle())
    assert text == ESSAY
    assert source == "deterministic"


def test_with_key_uses_llm_text_built_from_telemetry(monkeypatch):
    monkeypatch.setenv(narrative.API_KEY_ENV, "test-key-not-real")
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse(
            {"choices": [{"message": {"content": "  A brisk, bright record.  "}}]}
        )

    monkeypatch.setattr(narrative.httpx, "post", _fake_post)
    text, source = narrative.build_narrative(_track(), _bundle())
    assert text == "A brisk, bright record."
    assert source == "llm"

    # The request went to the chat-completions shape, carried the key,
    # and its user message is the locked telemetry — nothing else.
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer test-key-not-real"
    system, user = captured["json"]["messages"]
    assert "Never invent" in system["content"]
    assert "161.5" in user["content"]  # the bundle's measured BPM
    assert "E major" in user["content"]
    assert "Hip hop music" in user["content"]
    assert "similarity share" in user["content"]  # shares labeled, not probabilities


def test_with_key_http_failure_falls_back_to_deterministic(monkeypatch):
    monkeypatch.setenv(narrative.API_KEY_ENV, "test-key-not-real")

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("no network in tests", request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(narrative.httpx, "post", _boom)
    text, source = narrative.build_narrative(_track(), _bundle())
    assert text == ESSAY
    assert source == "deterministic-fallback"


def test_facts_sheet_contains_only_bundle_values():
    facts = narrative.telemetry_facts(_track(), _bundle())
    assert "161.5 BPM" in facts
    assert "C#m" in facts
    # A value that exists nowhere in the bundle must not appear.
    assert "140" not in facts
    assert "C major" not in facts
