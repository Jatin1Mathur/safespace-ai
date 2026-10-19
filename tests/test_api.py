"""Tests for the FastAPI routing in backend/main.py.

External services (Twilio, Ollama/MedGemma, location and hotline lookups)
are replaced with fakes, so the tests run offline and never place a call.
"""

import pytest
from fastapi.testclient import TestClient

import main
from ai_agent import find_nearby_therapists_by_location
from main import extract_city, is_health_related


@pytest.fixture
def client(monkeypatch):
    calls = {"emergency": 0}

    def fake_emergency():
        calls["emergency"] += 1
        return "FAKE_EMERGENCY_CALL"

    monkeypatch.setattr(main, "call_emergency", fake_emergency)
    monkeypatch.setattr(main, "get_crisis_hotlines", lambda: "FAKE_HOTLINES")
    monkeypatch.setattr(main, "query_medgemma_safe", lambda prompt: "FAKE_MEDGEMMA")
    monkeypatch.setattr(main, "session_history", [])

    test_client = TestClient(main.app)
    test_client.calls = calls
    return test_client


def ask(client, message):
    response = client.post("/ask", json={"message": message})
    assert response.status_code == 200
    return response.json()


def test_off_topic_request_is_filtered(client):
    data = ask(client, "Can you write me a python script for a stock trading bot?")
    assert data["tool_called"] == "topic_filter"
    assert data["risk"] == "NONE"


def test_high_risk_triggers_emergency_flow(client):
    data = ask(client, "I want to end my life, I can't do this anymore")
    assert data["risk"] == "HIGH"
    assert data["tool_called"] == "emergency_call_tool"
    assert "FAKE_HOTLINES" in data["response"]
    assert client.calls["emergency"] == 1


def test_medium_risk_adds_hotlines_without_calling(client):
    data = ask(client, "I feel hopeless about everything lately")
    assert data["risk"] == "MEDIUM"
    assert data["tool_called"] == "crisis_support"
    assert "FAKE_MEDGEMMA" in data["response"]
    assert "FAKE_HOTLINES" in data["response"]
    assert client.calls["emergency"] == 0


def test_breathing_request_routes_to_exercise(client):
    data = ask(client, "Can you guide me through a breathing exercise please?")
    assert data["tool_called"] == "breathing_exercise"


def test_missing_message_is_rejected(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_short_messages_pass_health_filter():
    assert is_health_related("hi")


@pytest.mark.parametrize(
    "message",
    [
        # "yes" used to match inside "yesterday"
        "Tell me who won the football match yesterday evening",
        # "hi" used to match inside "history"
        "What is the capital of France and its history",
        "Write me an essay about the French revolution",
    ],
)
def test_unrelated_long_message_fails_health_filter(message):
    assert not is_health_related(message)


@pytest.mark.parametrize(
    "message",
    [
        "I have been having trouble with my feelings lately",
        "Hi, I have been struggling with anxiety this week",
        "Can you find hospitals in Munich for me",
    ],
)
def test_health_messages_pass_health_filter(message):
    assert is_health_related(message)


def test_extract_city_prefers_capitalised_word():
    assert extract_city("find hospitals in Munich") == "Munich"


def test_therapist_tool_returns_no_fabricated_contacts():
    text = find_nearby_therapists_by_location.invoke({"location": "Hof"})
    assert "555" not in text
    assert "116117" in text


# ── Greetings and agent fallback ─────────────────────────────────────────────

@pytest.mark.parametrize("message", ["hey", "Hi!", "hello", "Good morning"])
def test_greeting_gets_friendly_reply_without_ai(client, message):
    data = ask(client, message)
    assert data["tool_called"] == "greeting"
    assert "SafeSpace" in data["response"]


def test_raw_tool_call_text_from_agent_falls_back_to_medgemma(client, monkeypatch):
    raw = 'assistant**\n{"name": "breathing_exercise_tool", "parameters": {"exercise": "4-7-8"}}'
    monkeypatch.setattr(main.graph, "stream", lambda *args, **kwargs: iter([]))
    monkeypatch.setattr(main, "parse_response", lambda stream: ("None", raw))
    data = ask(client, "I've been feeling stressed about my exams")
    assert data["response"] == "FAKE_MEDGEMMA"
    assert data["tool_called"] == "medgemma_fallback"


def test_normal_agent_answer_is_kept(client, monkeypatch):
    monkeypatch.setattr(main.graph, "stream", lambda *args, **kwargs: iter([]))
    monkeypatch.setattr(main, "parse_response",
                        lambda stream: ("ask_mental_health_specialist", "Exam stress is very common."))
    data = ask(client, "I've been feeling stressed about my exams")
    assert data["response"] == "Exam stress is very common."
    assert data["tool_called"] == "ask_mental_health_specialist"
