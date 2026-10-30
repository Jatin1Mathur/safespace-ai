"""Deterministic request classification for the SafeSpace graph.

`classify_request` is a pure function: same text in, same route out. It runs
in the graph's router node *before* any LLM is called, and the evaluation
suite in `evals/` scores it against a labelled golden dataset.
"""

import re
from typing import Literal

from pydantic import BaseModel
from tools import assess_crisis_risk, detect_mood

RiskLevel = Literal["NONE", "LOW", "MEDIUM", "HIGH"]

Route = Literal[
    "crisis_high",   # emergency flow, never model-generated text
    "crisis_medium",  # supportive answer + hotlines
    "off_topic",
    "greeting",
    "breathing",
    "sleep",
    "mood",
    "hotline",
    "summary",
    "soap",
    "pharmacy",
    "clinic",
    "agent",          # open conversation -> ReAct agent
]

# Which graph node handles each route (used for the conditional edges)
CRISIS_ROUTES = {"crisis_high", "crisis_medium"}
AGENT_ROUTES = {"agent"}


class RouteDecision(BaseModel):
    route: Route
    risk: RiskLevel
    mood: str


# ── Keyword lists ────────────────────────────────────────────────────────────

HEALTH_KEYWORDS = [
    # Mental health
    "anxiety", "anxious", "stress", "stressed", "depression", "depressed",
    "sad", "lonely", "overwhelmed", "panic", "trauma", "grief", "anger",
    "mood", "emotion", "feeling", "feel", "mental", "therapy", "therapist",
    "counselor", "psychologist", "psychiatrist", "burnout", "exhausted",
    "hopeless", "worthless", "fear", "phobia", "ocd", "ptsd", "bipolar",
    "schizophrenia", "eating disorder", "addiction", "sleep", "insomnia",
    # Physical / medical
    "hospital", "clinic", "doctor", "medical", "pharmacy", "medicine",
    "medication", "prescription", "symptoms", "pain", "health", "sick",
    "ill", "disease", "diagnosis", "treatment", "nurse", "emergency",
    "urgent", "injury", "headache", "breathing", "chest", "heart",
    # Crisis
    "suicide", "self harm", "self-harm", "kill myself", "hurt myself",
    "want to die", "end my life", "crisis", "help me", "no reason",
    # Tools
    "near me", "nearby", "find", "locate", "hotline", "helpline",
    "breathe", "breathing", "meditate", "meditation", "relax", "calm",
    "summary", "mood history", "how am i", "progress",
    # Greetings / general support (allow these)
    "hello", "hi", "hey", "thanks", "thank you", "yes", "no", "okay",
    "please", "help", "support", "talk", "listen", "need someone",
    "not okay", "struggling", "difficult", "hard time", "bad day",
]

CLINIC_KEYWORDS = [
    "hospital", "clinic", "doctor", "medical", "facility",
    "health center", "urgent care", "emergency room", "arzt",
    "krankenhaus", "klinik", "find doctor",
]

PHARMACY_KEYWORDS = [
    "pharmacy", "pharmacie", "apotheke", "medicine", "medication",
    "prescription", "drug store", "pills",
]

BREATHING_KEYWORDS = [
    "breathing", "breathe", "panic attack", "can't breathe",
    "anxious right now", "calm down", "relax", "meditation",
]

SLEEP_KEYWORDS = [
    "sleep", "insomnia", "can't sleep", "trouble sleeping",
    "wake up", "waking up", "nightmare", "tired", "fatigue", "rest",
]

MOOD_KEYWORDS = [
    "mood history", "how have i been", "my mood", "mood tracker",
    "emotional pattern", "progress", "how am i doing",
]

HOTLINE_KEYWORDS = [
    "hotline", "helpline", "crisis line", "phone number",
    "who can i call", "emergency contact", "crisis help",
]

SUMMARY_KEYWORDS = [
    "summarize", "summary", "what did we talk", "session summary",
    "recap", "what have i shared",
]

SOAP_KEYWORDS = [
    "soap note", "clinical report", "session report", "generate report",
    "doctor report", "medical report", "clinical note", "generate soap",
    "soap", "icd", "clinical document",
]

GREETINGS = {"hi", "hey", "hello", "hallo", "hi there", "hey there",
             "good morning", "good evening", "guten tag", "moin"}


def _keyword_pattern(keyword: str) -> re.Pattern:
    """Match a keyword at a word start, so "hi" does not match "history".

    Keywords of up to 3 letters (hi, yes, ill, sad...) must match a whole word;
    longer ones may continue ("feel" still matches "feeling").
    """
    escaped = re.escape(keyword)
    if len(keyword) <= 3:
        return re.compile(rf"\b{escaped}\b")
    return re.compile(rf"\b{escaped}")


def _patterns(keywords: list[str]) -> list[re.Pattern]:
    return [_keyword_pattern(kw) for kw in keywords]


HEALTH_PATTERNS = _patterns(HEALTH_KEYWORDS)

# Order matters: the first matching rule wins.
ROUTE_RULES: list[tuple[Route, list[re.Pattern]]] = [
    ("breathing", _patterns(BREATHING_KEYWORDS)),
    ("sleep", _patterns(SLEEP_KEYWORDS)),
    ("mood", _patterns(MOOD_KEYWORDS)),
    ("hotline", _patterns(HOTLINE_KEYWORDS)),
    ("summary", _patterns(SUMMARY_KEYWORDS)),
    ("soap", _patterns(SOAP_KEYWORDS)),
    ("pharmacy", _patterns(PHARMACY_KEYWORDS)),
    ("clinic", _patterns(CLINIC_KEYWORDS)),
]


def mentions(msg: str, patterns: list[re.Pattern]) -> bool:
    """Word-boundary keyword match, so "rest" does not match "nearest"."""
    return any(pattern.search(msg) for pattern in patterns)


def is_greeting(message: str) -> bool:
    """True for a plain greeting like "hey" or "Hello!"."""
    return message.lower().strip(" !.,?") in GREETINGS


def is_health_related(message: str) -> bool:
    """Check if message is related to health/mental health."""
    # Allow short messages (greetings, yes/no responses)
    if len(message.strip()) < 15:
        return True
    return mentions(message.lower(), HEALTH_PATTERNS)


def extract_city(message: str) -> str:
    """Extract a city name from a message such as "find hospitals in Munich"."""
    skip = {"find", "hospital", "hospitals", "clinic", "clinics", "doctor",
            "doctors", "medical", "near", "nearby", "in", "at", "around",
            "me", "my", "area", "please", "show", "get", "a", "the",
            "pharmacy", "pharmacies", "health", "center", "i", "am",
            "interested", "finding", "for"}
    words = [w.strip(".,!?") for w in message.split()]
    for word in words:
        if word.lower() not in skip and len(word) > 2 and word[0].isupper():
            return word
    for word in words:
        if word.lower() not in skip and len(word) > 2:
            return word
    return ""


def classify_request(message: str) -> RouteDecision:
    """Decide how a message is handled, before any LLM sees it.

    Order: crisis check -> greeting -> intent rules -> topic filter -> agent.
    Safety comes first, so a message with crisis language can never be
    answered with the off-topic reply.
    """
    msg = message.lower()
    risk = assess_crisis_risk(message)["level"]
    mood = detect_mood(message)

    if risk == "HIGH":
        return RouteDecision(route="crisis_high", risk=risk, mood=mood)
    if risk == "MEDIUM":
        return RouteDecision(route="crisis_medium", risk=risk, mood=mood)
    if is_greeting(message):
        return RouteDecision(route="greeting", risk=risk, mood=mood)
    # A clear support intent (hotline, pharmacy, ...) is on-topic by definition,
    # so intent rules run before the generic topic filter.
    for route, patterns in ROUTE_RULES:
        if mentions(msg, patterns):
            return RouteDecision(route=route, risk=risk, mood=mood)
    if not is_health_related(message):
        return RouteDecision(route="off_topic", risk="NONE", mood="neutral")
    return RouteDecision(route="agent", risk=risk, mood=mood)
