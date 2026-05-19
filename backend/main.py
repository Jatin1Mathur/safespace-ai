from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
from urllib.parse import quote

from ai_agent import graph, SYSTEM_PROMPT, parse_response
from tools import (
    call_emergency, find_clinics_by_city, find_pharmacies,
    get_crisis_hotlines, get_breathing_exercise, get_sleep_advice,
    get_mood_summary, detect_mood, log_mood, assess_crisis_risk,
    generate_session_summary, generate_soap_note, load_memory
)

app = FastAPI()

# Store session history in memory
session_history = []

class Query(BaseModel):
    message: str


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
    "krankenhaus", "klinik", "find doctor"
]

PHARMACY_KEYWORDS = [
    "pharmacy", "pharmacie", "apotheke", "medicine", "medication",
    "prescription", "drug store", "pills"
]

EMERGENCY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "want to die", "no reason to live", "hurt myself", "overdose"
]

BREATHING_KEYWORDS = [
    "breathing", "breathe", "panic attack", "can't breathe",
    "anxious right now", "calm down", "relax", "meditation"
]

SLEEP_KEYWORDS = [
    "sleep", "insomnia", "can't sleep", "trouble sleeping",
    "wake up", "nightmares", "tired", "fatigue", "rest"
]

MOOD_KEYWORDS = [
    "mood history", "how have i been", "my mood", "mood tracker",
    "emotional pattern", "progress", "how am i doing"
]

HOTLINE_KEYWORDS = [
    "hotline", "helpline", "crisis line", "phone number",
    "who can i call", "emergency contact", "crisis help"
]

SUMMARY_KEYWORDS = [
    "summarize", "summary", "what did we talk", "session summary",
    "recap", "what have i shared"
]

SOAP_KEYWORDS = [
    "soap note", "clinical report", "session report", "generate report",
    "doctor report", "medical report", "clinical note", "generate soap",
    "soap", "icd", "clinical document"
]

OFF_TOPIC_RESPONSE = """🌿 I'm SafeSpace — a mental health and medical support assistant.

I'm here to help with:
• 💬 Mental health support & emotional guidance
• 🏥 Finding hospitals, clinics & pharmacies nearby
• 🫁 Breathing exercises & relaxation techniques
• 😴 Sleep improvement advice
• 📞 Crisis hotlines & emergency support
• 📊 Mood tracking & session summaries

I'm not able to help with topics outside of healthcare and wellbeing. Is there something health-related I can support you with today? 💚"""


def is_health_related(message: str) -> bool:
    """Check if message is related to health/mental health."""
    msg_lower = message.lower()
    # Allow short messages (greetings, yes/no responses)
    if len(message.strip()) < 15:
        return True
    return any(kw in msg_lower for kw in HEALTH_KEYWORDS)


def extract_city(message: str) -> str:
    """Extract city name from message."""
    skip = {"find", "hospital", "hospitals", "clinic", "clinics", "doctor",
            "doctors", "medical", "near", "nearby", "in", "at", "around",
            "me", "my", "area", "please", "show", "get", "a", "the",
            "pharmacy", "pharmacies", "health", "center"}
    for word in message.split():
        if word.lower() not in skip and len(word) > 2 and word[0].isupper():
            return word
    # fallback: any word not in skip
    for word in message.split():
        if word.lower() not in skip and len(word) > 2:
            return word
    return ""


@app.post("/ask")
async def ask(query: Query):
    global session_history
    msg = query.message.lower()
    original = query.message

    # ── Health topic filter ──────────────────────────────────────────────────
    if not is_health_related(original):
        return {
            "response": OFF_TOPIC_RESPONSE,
            "tool_called": "topic_filter",
            "mood": "neutral",
            "risk": "NONE"
        }

    # ── Crisis risk assessment ───────────────────────────────────────────────
    risk = assess_crisis_risk(original)

    # ── Mood detection & logging ─────────────────────────────────────────────
    mood = detect_mood(original)
    if mood != "neutral":
        log_mood(mood)

    # ── HIGH risk → emergency call immediately ───────────────────────────────
    if risk["level"] == "HIGH":
        call_result = call_emergency()
        hotlines = get_crisis_hotlines()
        response = f"""🚨 I'm very concerned about what you've shared.

You are not alone, and your life has value. I've initiated an emergency contact on your behalf.

{hotlines}

Please reach out to one of these lines right now — they are free, confidential, and available 24/7.

{call_result}"""
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "emergency_call_tool",
                "mood": mood, "risk": risk["level"]}

    # ── MEDIUM risk → suggest hotlines + therapy ─────────────────────────────
    if risk["level"] == "MEDIUM":
        hotlines = get_crisis_hotlines()
        ai_response = query_medgemma_safe(original)
        response = f"{ai_response}\n\n---\n💙 Support is available:\n{hotlines}"
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "crisis_support",
                "mood": mood, "risk": risk["level"]}

    # ── Breathing exercise ───────────────────────────────────────────────────
    if any(kw in msg for kw in BREATHING_KEYWORDS):
        technique = "478" if "sleep" in msg else "box"
        response = get_breathing_exercise(technique)
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "breathing_exercise",
                "mood": mood, "risk": risk["level"]}

    # ── Sleep advice ─────────────────────────────────────────────────────────
    if any(kw in msg for kw in SLEEP_KEYWORDS):
        response = get_sleep_advice(original)
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "sleep_advice",
                "mood": mood, "risk": risk["level"]}

    # ── Mood history ─────────────────────────────────────────────────────────
    if any(kw in msg for kw in MOOD_KEYWORDS):
        response = get_mood_summary()
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "mood_tracker",
                "mood": mood, "risk": risk["level"]}

    # ── Crisis hotlines ───────────────────────────────────────────────────────
    if any(kw in msg for kw in HOTLINE_KEYWORDS):
        response = get_crisis_hotlines()
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "hotline_directory",
                "mood": mood, "risk": risk["level"]}

    # ── Session summary ───────────────────────────────────────────────────────
    if any(kw in msg for kw in SUMMARY_KEYWORDS):
        response = generate_session_summary(session_history)
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "session_summary",
                "mood": mood, "risk": risk["level"]}


    # ── SOAP Note Generator ───────────────────────────────────────────────────
    if any(kw in msg for kw in SOAP_KEYWORDS):
        memory = load_memory()
        mood_history = memory.get("mood_history", [])
        response = generate_soap_note(session_history, mood_history)
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "soap_note_generator",
                "mood": mood, "risk": risk["level"]}

    # ── Pharmacy finder ───────────────────────────────────────────────────────
    if any(kw in msg for kw in PHARMACY_KEYWORDS):
        response = find_pharmacies()
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "pharmacy_finder",
                "mood": mood, "risk": risk["level"]}

    # ── Clinic/hospital finder ────────────────────────────────────────────────
    if any(kw in msg for kw in CLINIC_KEYWORDS):
        city = extract_city(original)
        if city:
            response = find_clinics_by_city(city)
        else:
            response = "📍 Which city should I search in?\n\nFor example: *find hospitals in Hof* or *clinics in Munich*"
        session_history.append({"role": "user", "content": original})
        session_history.append({"role": "assistant", "content": response})
        return {"response": response, "tool_called": "clinic_finder",
                "mood": mood, "risk": risk["level"]}

    # ── General AI agent (mental health conversation) ─────────────────────────
    inputs = {
        "messages": [
            ("system", SYSTEM_PROMPT),
            ("user", original)
        ]
    }
    stream = graph.stream(inputs, stream_mode="updates")
    tool_called_name, final_response = parse_response(stream)

    session_history.append({"role": "user", "content": original})
    session_history.append({"role": "assistant", "content": final_response or ""})

    # Keep history manageable
    if len(session_history) > 40:
        session_history = session_history[-40:]

    return {
        "response": final_response,
        "tool_called": tool_called_name,
        "mood": mood,
        "risk": risk["level"]
    }


def query_medgemma_safe(prompt: str) -> str:
    """MedGemma with safe messaging for medium-risk cases."""
    from tools import query_medgemma
    return query_medgemma(prompt)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)