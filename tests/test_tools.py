"""Unit tests for the rule-based safety and helper logic in backend/tools.py."""

import pytest

from tools import assess_crisis_risk, detect_mood, get_breathing_exercise, load_memory, log_mood


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
