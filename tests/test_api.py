"""Tests for the FastAPI service and the LangGraph routing behind it.

External services (Twilio, Ollama/MedGemma, location and hotline lookups)
are replaced with fakes, so the tests run offline and never place a call.
"""

import agent_graph
import main
import pytest
from ai_agent import find_nearby_therapists_by_location
from fastapi.testclient import TestClient
from routing import extract_city, is_health_related


@pytest.fixture
def client(fake_services):
    test_client = TestClient(main.app)
    test_client.calls = fake_services
    return test_client


def ask(client, message, thread_id=None):
    payload = {"message": message}
    if thread_id:
        payload["thread_id"] = thread_id
    response = client.post("/ask", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_off_topic_request_is_filtered(client):
    data = ask(client, "Can you write me a python script for a stock trading bot?")
    assert data["tool_called"] == "topic_filter"
    assert data["route"] == "off_topic"
    assert data["risk"] == "NONE"


def test_high_risk_triggers_emergency_flow(client):
    data = ask(client, "I want to end my life, I can't do this anymore")
    assert data["risk"] == "HIGH"
    assert data["tool_called"] == "emergency_call_tool"
    assert "FAKE_HOTLINES" in data["response"]
    assert client.calls["emergency"] == 1
    assert client.calls["agent"] == 0  # no model-generated text on the crisis path


def test_crisis_check_runs_before_topic_filter(client):
    # Long message with no health keywords except the crisis phrase
    data = ask(client, "Honestly after the exam results yesterday I just want to die")
    assert data["route"] == "crisis_high"


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


# ── Structured errors ────────────────────────────────────────────────────────

def test_missing_message_gives_structured_422(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"][0]["field"] == "message"


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "hi", "thread_id": "bad id!"}])
def test_invalid_fields_are_rejected(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_unexpected_error_gives_structured_500(fake_services, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("graph exploded")
    monkeypatch.setattr(agent_graph, "run_turn", boom)
    response = TestClient(main.app, raise_server_exceptions=False).post("/ask", json={"message": "hello"})
    assert response.status_code == 500
    body = response.json()["error"]
    assert body["code"] == "internal_error"
    assert "112" in body["message"]
    assert "exploded" not in body["message"]  # internals never leak


def test_health_endpoint(client):
    assert client.get("/health").json() == {"status": "ok"}


# ── Greetings and agent fallback ─────────────────────────────────────────────

@pytest.mark.parametrize("message", ["hey", "Hi!", "hello", "Good morning"])
def test_greeting_gets_friendly_reply_without_ai(client, message):
    data = ask(client, message)
    assert data["tool_called"] == "greeting"
    assert "SafeSpace" in data["response"]
    assert client.calls["agent"] == 0


def test_raw_tool_call_text_from_agent_falls_back_to_medgemma(client, set_agent_reply):
    set_agent_reply("None", 'assistant**\n{"name": "breathing_exercise_tool", "parameters": {"exercise": "4-7-8"}}')
    data = ask(client, "I've been feeling stressed about my exams")
    assert data["response"] == "FAKE_MEDGEMMA"
    assert data["tool_called"] == "medgemma_fallback"


def test_normal_agent_answer_is_kept(client, set_agent_reply):
    set_agent_reply("ask_mental_health_specialist", "Exam stress is very common.")
    data = ask(client, "I've been feeling stressed about my exams")
    assert data["response"] == "Exam stress is very common."
    assert data["tool_called"] == "ask_mental_health_specialist"
    assert data["route"] == "agent"


def test_agent_crash_falls_back_instead_of_500(client, monkeypatch):
    def broken_agent(context):
        raise ConnectionError("Ollama is not running")
    monkeypatch.setattr(agent_graph, "run_react_agent", broken_agent)
    data = ask(client, "I've been feeling stressed about my exams")  # ask() asserts 200
    assert data["response"] == "FAKE_MEDGEMMA"
    assert data["tool_called"] == "medgemma_fallback"


# ── Routing uses whole words, and every quick-button on the page works ─────────

@pytest.mark.parametrize(
    "message, expected_tool",
    [
        ("Where is the nearest pharmacy?", "pharmacy_finder"),      # "rest" in "nearest"
        ("I am interested in finding a doctor in Munich", "clinic_finder"),  # "rest" in "interested"
        # the eight quick-buttons in gradio_frontend.py
        ("Find hospitals in Hof", "clinic_finder"),
        ("Find pharmacy near me", "pharmacy_finder"),
        ("I am having trouble sleeping", "sleep_advice"),
        ("Give me a breathing exercise", "breathing_exercise"),
        ("Show my mood history", "mood_tracker"),
        ("Give me crisis hotline numbers", "hotline_directory"),
    ],
)
def test_routing_matches_whole_words(client, message, expected_tool):
    assert ask(client, message)["tool_called"] == expected_tool


@pytest.mark.parametrize("message", ["I am feeling anxious today", "I feel overwhelmed"])
def test_emotional_quick_buttons_reach_the_agent(client, set_agent_reply, message):
    set_agent_reply("ask_mental_health_specialist", "A caring answer.")
    data = ask(client, message)
    assert data["tool_called"] == "ask_mental_health_specialist"
    assert data["risk"] == "LOW"


# ── Per-thread conversation memory ───────────────────────────────────────────

def test_new_conversation_gets_a_thread_id(client):
    data = ask(client, "hello")
    assert data["thread_id"]


def test_agent_sees_earlier_messages_of_the_same_thread(client):
    first = ask(client, "My name is Jatin and I feel stressed about exams")
    ask(client, "What else can I do about feeling stressed?", thread_id=first["thread_id"])
    context = client.calls["agent_contexts"][-1]
    texts = [m.content for m in context]
    assert "My name is Jatin and I feel stressed about exams" in texts
    assert texts[-1] == "What else can I do about feeling stressed?"


def test_threads_do_not_share_history(client):
    ask(client, "My name is Jatin and I feel stressed about exams", thread_id="user-a")
    ask(client, "I feel stressed about work deadlines", thread_id="user-b")
    context_b = [m.content for m in client.calls["agent_contexts"][-1]]
    assert not any("Jatin" in text for text in context_b)


def test_session_summary_only_uses_this_thread(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(agent_graph, "generate_session_summary",
                        lambda history: seen.setdefault("history", history) and "SUMMARY")
    ask(client, "I feel stressed about exams", thread_id="t1")
    ask(client, "I feel lonely at work", thread_id="t2")
    ask(client, "Can you give me a session summary?", thread_id="t1")
    contents = [m["content"] for m in seen["history"]]
    assert "I feel stressed about exams" in contents
    assert "I feel lonely at work" not in contents
