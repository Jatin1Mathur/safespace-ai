"""Unit tests for the rule-based safety and helper logic in backend/tools.py."""

import pytest
import requests

import tools
from tools import (
    assess_crisis_risk, detect_mood, find_clinics_by_city, get_breathing_exercise,
    load_memory, log_mood, query_overpass,
)


@pytest.mark.parametrize(
    "message",
    [
        "I want to kill myself",
        "I have been feeling suicidal lately",
        "I keep thinking about self-harm",
        "I think about self harm a lot",
        "I want to end my life",
        "I might hurt myself tonight",
    ],
)
def test_high_risk_messages_are_flagged(message):
    assert assess_crisis_risk(message)["level"] == "HIGH"


@pytest.mark.parametrize(
    "message",
    ["I feel hopeless", "I'm just a burden to everyone", "There is no way out"],
)
def test_medium_risk_messages_are_flagged(message):
    assert assess_crisis_risk(message)["level"] == "MEDIUM"


def test_low_risk_message():
    assert assess_crisis_risk("I'm a bit stressed about exams") == {"level": "LOW", "score": 1}


def test_neutral_message_has_no_risk():
    assert assess_crisis_risk("What are the opening hours of a pharmacy?") == {"level": "NONE", "score": 0}


def test_crisis_check_is_case_insensitive():
    assert assess_crisis_risk("I WANT TO DIE")["level"] == "HIGH"


@pytest.mark.parametrize(
    "message, mood",
    [
        ("I'm so anxious before my exam", "anxious"),
        ("I feel really lonely", "lonely"),
        ("Work has me completely overwhelmed", "stressed"),
        ("The weather is cloudy", "neutral"),
    ],
)
def test_detect_mood(message, mood):
    assert detect_mood(message) == mood


def test_log_mood_persists_entry():
    log_mood("anxious")
    history = load_memory()["mood_history"]
    assert len(history) == 1
    assert history[0]["mood"] == "anxious"


@pytest.mark.parametrize("technique", ["box", "478"])
def test_breathing_exercise_returns_text(technique):
    text = get_breathing_exercise(technique)
    assert isinstance(text, str) and text.strip()


# ── Map search with backup servers (no real network calls) ──────────────────

class FakeResponse:
    def __init__(self, elements):
        self.status_code = 200
        self.text = "ok"
        self._elements = elements

    def json(self):
        return {"elements": self._elements}


def test_overpass_falls_back_to_backup_server(monkeypatch):
    called = []

    def fake_get(url, **kwargs):
        called.append(url)
        if url == tools.OVERPASS_SERVERS[0]:
            raise requests.Timeout("main server too slow")
        return FakeResponse([{"tags": {"name": "Klinikum"}}])

    monkeypatch.setattr(tools.requests, "get", fake_get)
    assert query_overpass("query") == [{"tags": {"name": "Klinikum"}}]
    assert called == tools.OVERPASS_SERVERS[:2]


def test_overpass_returns_none_when_all_servers_fail(monkeypatch):
    def always_timeout(url, **kwargs):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(tools.requests, "get", always_timeout)
    assert query_overpass("query") is None


def test_clinic_search_shows_friendly_message_when_map_service_is_down(monkeypatch):
    class GeoResponse:
        def json(self):
            return [{"lat": "48.1", "lon": "11.6", "display_name": "Munich, Bavaria"}]

    monkeypatch.setattr(tools.requests, "get", lambda url, **kwargs: GeoResponse())
    monkeypatch.setattr(tools, "query_overpass", lambda query: None)

    text = find_clinics_by_city("Munich")
    assert "busy" in text
    assert "maps.google.com" in text
    assert "HTTPSConnectionPool" not in text


def test_clinic_search_uses_nominatim_when_overpass_is_down(monkeypatch):
    class Geo:
        status_code = 200
        def __init__(self, data): self._data = data
        def json(self): return self._data

    def fake_get(url, params=None, **kwargs):
        if params and params["q"] == "Stuttgart":          # city → coordinates
            return Geo([{"lat": "48.78", "lon": "9.18", "display_name": "Stuttgart, Germany"}])
        return Geo([                                         # "hospital in Stuttgart"
            {"class": "amenity", "type": "hospital", "name": "Katharinenhospital",
             "display_name": "Katharinenhospital, 60, Kriegsbergstraße, Universität, Stuttgart"},
            {"class": "place", "type": "city", "name": "Stuttgart", "display_name": "Stuttgart"},
        ])

    monkeypatch.setattr(tools.requests, "get", fake_get)
    monkeypatch.setattr(tools, "query_overpass", lambda query: None)   # all Overpass servers down

    text = find_clinics_by_city("Stuttgart")
    assert "Katharinenhospital" in text
    assert "busy" not in text
    assert "• Stuttgart" not in text          # only real hospitals, not the city itself
