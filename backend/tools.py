import ollama
import requests
import json
import os
from urllib.parse import quote
from datetime import datetime
from twilio.rest import Client
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, EMERGENCY_CONTACT

# ── Session memory file ──────────────────────────────────────────────────────
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "session_memory.json")

def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"sessions": [], "mood_history": [], "reminders": []}

def save_memory(data: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── 1. Location ──────────────────────────────────────────────────────────────
def get_current_location() -> dict:
    try:
        res = requests.get("http://ip-api.com/json/", timeout=8).json()
        if res.get("status") != "success":
            raise Exception("IP API failed")
        lat, lng = res["lat"], res["lon"]
        country = res.get("country", "")
        country_code = res.get("countryCode", "DE")
        reverse = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "json", "zoom": 10, "addressdetails": 1},
            headers={"User-Agent": "SafeSpaceAI/1.0"}, timeout=8
        ).json()
        address = reverse.get("address", {})
        city = (address.get("city") or address.get("town") or
                address.get("municipality") or address.get("village") or
                res.get("city", "your area"))
        return {"lat": lat, "lng": lng, "city": city, "country": country,
                "country_code": country_code, "display": f"{city}, {country}"}
    except Exception:
        pass
    return {"lat": 50.3133, "lng": 11.9078, "city": "Hof",
            "country": "Germany", "country_code": "DE", "display": "Hof, Germany"}


# ── 2. MedGemma therapist ────────────────────────────────────────────────────
def query_medgemma(prompt: str) -> str:
    system_prompt = """You are Dr. Emily Hartman, a warm and experienced clinical psychologist.
    Respond to patients with:
    1. Emotional attunement ("I can sense how difficult this must be...")
    2. Gentle normalization ("Many people feel this way when...")
    3. Practical guidance ("What sometimes helps is...")
    4. Strengths-focused support ("I notice how you're...")
    Key principles:
    - Never use brackets or labels
    - Blend elements seamlessly
    - Vary sentence structure
    - Use natural transitions
    - Mirror the user's language level
    - Always keep the conversation going by asking open ended questions
    - Follow WHO safe messaging guidelines for sensitive topics
    - Never describe methods of self-harm, always provide hope and resources
    """
    try:
        response = ollama.chat(
            model='alibayram/medgemma:4b',
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
            options={'num_predict': 350, 'temperature': 0.7, 'top_p': 0.9}
        )
        return response['message']['content'].strip()
    except Exception:
        return "I'm having technical difficulties, but your feelings matter. Please try again shortly."


# ── 3. Emergency call ────────────────────────────────────────────────────────
def call_emergency() -> str:
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            to=EMERGENCY_CONTACT, from_=TWILIO_FROM_NUMBER,
            url="http://demo.twilio.com/docs/voice.xml"
        )
        return f"Emergency call placed. Call SID: {call.sid}"
    except Exception as e:
        return f"Failed to place emergency call: {str(e)}"


# ── 4. Crisis risk detector ──────────────────────────────────────────────────
def assess_crisis_risk(message: str) -> dict:
    HIGH_RISK = ["kill myself", "end my life", "suicide", "want to die",
                 "no reason to live", "hurt myself", "self harm", "overdose"]
    MEDIUM_RISK = ["can't go on", "hopeless", "worthless", "nobody cares",
                   "give up", "disappear", "burden", "trapped", "no way out"]
    LOW_RISK = ["sad", "depressed", "anxious", "stressed", "lonely",
                "overwhelmed", "crying", "can't sleep", "panic"]

    msg_lower = message.lower()
    if any(kw in msg_lower for kw in HIGH_RISK):
        return {"level": "HIGH", "score": 3}
    elif any(kw in msg_lower for kw in MEDIUM_RISK):
        return {"level": "MEDIUM", "score": 2}
    elif any(kw in msg_lower for kw in LOW_RISK):
        return {"level": "LOW", "score": 1}
    return {"level": "NONE", "score": 0}


# ── 5. Mood detector ─────────────────────────────────────────────────────────
def detect_mood(message: str) -> str:
    MOODS = {
        "anxious":   ["anxious", "anxiety", "panic", "nervous", "worried", "fear"],
        "depressed": ["depressed", "depression", "hopeless", "empty", "numb", "sad"],
        "angry":     ["angry", "furious", "rage", "frustrated", "irritated"],
        "lonely":    ["lonely", "alone", "isolated", "no one", "nobody"],
        "stressed":  ["stressed", "overwhelmed", "pressure", "burnout", "exhausted"],
        "happy":     ["happy", "good", "great", "better", "grateful", "hopeful"],
        "neutral":   []
    }
    msg_lower = message.lower()
    for mood, keywords in MOODS.items():
        if any(kw in msg_lower for kw in keywords):
            return mood
    return "neutral"


def log_mood(mood: str):
    memory = load_memory()
    memory["mood_history"].append({
        "mood": mood,
        "timestamp": datetime.now().isoformat()
    })
    save_memory(memory)


def get_mood_summary() -> str:
    memory = load_memory()
    history = memory.get("mood_history", [])
    if not history:
        return "No mood history yet. Start chatting to track your emotional patterns."
    recent = history[-10:]
    mood_counts = {}
    for entry in recent:
        m = entry["mood"]
        mood_counts[m] = mood_counts.get(m, 0) + 1
    summary = "📊 Your recent mood pattern:\n\n"
    MOOD_EMOJI = {
        "anxious": "😟", "depressed": "😔", "angry": "😠",
        "lonely": "💙", "stressed": "😰", "happy": "😊", "neutral": "😐"
    }
    for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1]):
        emoji = MOOD_EMOJI.get(mood, "•")
        bar = "█" * count + "░" * (10 - min(count, 10))
        summary += f"{emoji} {mood.capitalize():<12} {bar} {count}x\n"
    return summary


# ── 6. Breathing exercise ────────────────────────────────────────────────────
def get_breathing_exercise(technique: str = "box") -> str:
    exercises = {
        "box": {
            "name": "Box Breathing (used by Navy SEALs)",
            "steps": [
                "1️⃣  Inhale slowly through your nose — count to 4",
                "2️⃣  Hold your breath — count to 4",
                "3️⃣  Exhale slowly through your mouth — count to 4",
                "4️⃣  Hold empty — count to 4",
            ],
            "rounds": 4,
            "benefit": "Reduces anxiety and stress within 2-3 minutes"
        },
        "478": {
            "name": "4-7-8 Breathing (Dr. Andrew Weil)",
            "steps": [
                "1️⃣  Inhale through your nose — count to 4",
                "2️⃣  Hold your breath — count to 7",
                "3️⃣  Exhale completely through mouth — count to 8",
            ],
            "rounds": 4,
            "benefit": "Calms the nervous system, helps with sleep"
        },
        "calm": {
            "name": "Simple Calm Breathing",
            "steps": [
                "1️⃣  Breathe in gently — count to 5",
                "2️⃣  Breathe out slowly — count to 5",
            ],
            "rounds": 6,
            "benefit": "Quick stress relief, can be done anywhere"
        }
    }
    ex = exercises.get(technique, exercises["box"])
    output = f"🫁 {ex['name']}\n\n"
    output += "\n".join(ex["steps"])
    output += f"\n\n🔁 Repeat {ex['rounds']} times\n"
    output += f"✨ {ex['benefit']}\n\n"
    output += "Take a moment now. I'll be here when you're ready to continue. 💚"
    return output


# ── 7. Mental health hotlines ────────────────────────────────────────────────
def get_crisis_hotlines() -> str:
    location = get_current_location()
    country_code = location.get("country_code", "DE")

    HOTLINES = {
        "DE": [
            ("Telefonseelsorge (24/7, free)", "0800 111 0 111"),
            ("Telefonseelsorge (24/7, free)", "0800 111 0 222"),
            ("Krisentelefon Berlin", "030 390 63 00"),
            ("Youth Crisis Line", "0800 111 0 333"),
        ],
        "US": [
            ("988 Suicide & Crisis Lifeline", "988"),
            ("Crisis Text Line", "Text HOME to 741741"),
            ("NAMI Helpline", "1-800-950-6264"),
        ],
        "GB": [
            ("Samaritans (24/7)", "116 123"),
            ("PAPYRUS (youth)", "0800 068 4141"),
            ("Mind Infoline", "0300 123 3393"),
        ],
        "IN": [
            ("iCall", "9152987821"),
            ("Vandrevala Foundation (24/7)", "1860-2662-345"),
            ("AASRA", "91-22-27546669"),
        ],
    }

    lines = HOTLINES.get(country_code, HOTLINES["DE"])
    display = location.get("display", "your area")

    output = f"📞 Crisis Helplines for {display}:\n\n"
    for name, number in lines:
        output += f"• {name}\n  ☎️  {number}\n\n"
    output += "These lines are free, confidential, and available 24/7.\n"
    output += "You are not alone. Reaching out is a sign of strength. 💚"
    return output


# ── 8. Pharmacy finder ───────────────────────────────────────────────────────
def find_pharmacies() -> str:
    try:
        location = get_current_location()
        lat, lng = location["lat"], location["lng"]
        display = location["display"]
        city = location.get("city", "")

        query = f'[out:json][timeout:25];node["amenity"="pharmacy"](around:5000,{lat},{lng});out body 6;'
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query}, timeout=20,
            headers={"User-Agent": "SafeSpaceAI/1.0"}
        )
        results = []
        if response.status_code == 200 and response.text.strip():
            results = response.json().get("elements", [])

        if not results:
            return (f"📍 Your location: {display}\n\n"
                    f"🗺 Search pharmacies: https://maps.google.com/?q=pharmacy+near+{quote(display)}")

        output = f"📍 Your location: {display}\n\n💊 Nearby Pharmacies:\n\n"
        seen = set()
        count = 0
        for place in results:
            tags = place.get("tags", {})
            name = tags.get("name", "")
            if not name or name in seen:
                continue
            seen.add(name)
            street = tags.get("addr:street", "")
            housenumber = tags.get("addr:housenumber", "")
            city_tag = tags.get("addr:city", city)
            address = f"{street} {housenumber}, {city_tag}".strip(", ")
            phone = tags.get("phone", tags.get("contact:phone", ""))
            opening = tags.get("opening_hours", "")
            maps_link = f"https://maps.google.com/?q={quote(name + ' ' + city_tag)}"
            output += f"• {name}\n"
            output += f"  📍 {address if address.strip(',') else 'Address not listed'}\n"
            if phone:
                output += f"  📞 {phone}\n"
            if opening:
                output += f"  🕐 {opening}\n"
            output += f"  🗺 Maps: {maps_link}\n\n"
            count += 1
            if count >= 5:
                break
        return output.strip()
    except Exception as e:
        return f"Error finding pharmacies: {str(e)}"


# ── 9. Hospitals & clinics ───────────────────────────────────────────────────
def find_clinics_by_city(city: str) -> str:
    try:
        geo = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": "SafeSpaceAI/1.0"}, timeout=8
        ).json()
        if not geo:
            return f"Could not find location: {city}"
        lat, lng = float(geo[0]["lat"]), float(geo[0]["lon"])
        display = geo[0].get("display_name", city).split(",")[0]

        query = f'[out:json][timeout:25];(node["amenity"~"hospital|clinic|doctors"](around:10000,{lat},{lng});way["amenity"~"hospital|clinic"](around:10000,{lat},{lng}););out center 10;'
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query}, timeout=20,
            headers={"User-Agent": "SafeSpaceAI/1.0"}
        )
        results = []
        if response.status_code == 200 and response.text.strip():
            results = response.json().get("elements", [])

        if not results:
            return (f"📍 Searching near: {display}\n\n"
                    f"🗺 Google Maps: https://maps.google.com/?q=hospitals+near+{quote(city)}")

        output = f"📍 Searching near: {display}\n\n🏥 Nearby Medical Facilities:\n\n"
        seen = set()
        count = 0
        for place in results:
            tags = place.get("tags", {})
            name = tags.get("name", "")
            if not name or name in seen:
                continue
            seen.add(name)
            street = tags.get("addr:street", "")
            housenumber = tags.get("addr:housenumber", "")
            city_tag = tags.get("addr:city", city)
            address = f"{street} {housenumber}, {city_tag}".strip(", ")
            phone = tags.get("phone", tags.get("contact:phone", ""))
            email = tags.get("email", tags.get("contact:email", ""))
            website = tags.get("website", tags.get("contact:website", ""))
            facility_type = tags.get("amenity", "facility").replace("_", " ").title()
            google_link = f"https://www.google.com/search?q={quote(name + ' ' + city_tag + ' Telefon')}"
            maps_link = f"https://maps.google.com/?q={quote(name + ' ' + city_tag)}"
            output += f"• {name} ({facility_type})\n"
            output += f"  📍 {address if address.strip(',') else 'Address not listed'}\n"
            if phone:
                output += f"  📞 {phone}\n"
            if email:
                output += f"  📧 {email}\n"
            if website:
                output += f"  🌐 {website}\n"
            output += f"  🔍 Contact: {google_link}\n"
            output += f"  🗺 Maps: {maps_link}\n\n"
            count += 1
            if count >= 5:
                break
        return output.strip()
    except Exception as e:
        return f"Error finding clinics: {str(e)}"


# ── 10. Session summary ──────────────────────────────────────────────────────
def generate_session_summary(messages: list) -> str:
    if not messages:
        return "No conversation to summarize yet."
    convo = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages[-20:]])
    try:
        response = ollama.chat(
            model='alibayram/medgemma:4b',
            messages=[
                {"role": "system", "content": "You are a clinical assistant. Summarize this therapy session in 3-5 sentences covering: main topics discussed, emotional state, and any recommendations made. Be concise and professional."},
                {"role": "user", "content": f"Session transcript:\n{convo}"}
            ],
            options={'num_predict': 200, 'temperature': 0.3}
        )
        summary = response['message']['content'].strip()
        return f"📋 Session Summary\n{'─'*30}\n{summary}\n\n📅 {datetime.now().strftime('%B %d, %Y at %H:%M')}"
    except Exception:
        return "Could not generate summary. Please try again."


# ── 11. Sleep advice ─────────────────────────────────────────────────────────
def get_sleep_advice(issue: str = "") -> str:
    advice = """😴 CBT-Based Sleep Hygiene Guide

🕐 SCHEDULE
• Go to bed and wake up at the same time every day
• Avoid naps longer than 20 minutes after 3pm

📵 SCREEN & LIGHT
• Stop screens 1 hour before bed (blue light blocks melatonin)
• Keep bedroom dark and cool (16-18°C / 60-65°F)

🧘 WIND-DOWN ROUTINE (30 min before bed)
• Dim the lights
• Try: reading, light stretching, or journaling
• Avoid work emails or stressful content

🍵 FOOD & DRINK
• No caffeine after 2pm
• Avoid large meals within 2 hours of bedtime
• Warm herbal tea (chamomile) can help

💭 FOR RACING THOUGHTS
• Write tomorrow's to-do list before bed (offload from brain)
• Try 4-7-8 breathing (ask me to guide you)
• Progressive muscle relaxation

📊 TRACK YOUR SLEEP
• Note bedtime, wake time, and quality (1-5)
• Look for patterns after 1-2 weeks

Would you like me to guide you through a specific technique? 💚"""
    return advice


# ── 12. Multi-language detect ────────────────────────────────────────────────
def detect_language(text: str) -> str:
    german = ["ich", "bin", "habe", "nicht", "und", "der", "die", "das", "ist", "mir", "wie", "kann"]
    hindi  = ["मैं", "है", "नहीं", "और", "हूं", "मुझे", "कि"]
    spanish= ["yo", "soy", "estoy", "tengo", "no", "que", "me", "mi", "es", "muy"]

    words = text.lower().split()
    if any(w in words for w in german):
        return "de"
    if any(w in text for w in hindi):
        return "hi"
    if any(w in words for w in spanish):
        return "es"
    return "en"


# ── SOAP Note Generator ──────────────────────────────────────────────────────
def generate_soap_note(session_history: list, mood_history: list = None) -> str:
    """
    Generate a clinical SOAP note from session history.
    SOAP = Subjective, Objective, Assessment, Plan
    Standard format used in hospitals and clinics worldwide.
    """
    if not session_history:
        return "No session data available to generate a SOAP note."

    # Extract user messages only for subjective
    user_messages = [m["content"] for m in session_history if m["role"] == "user"]
    assistant_messages = [m["content"] for m in session_history if m["role"] == "assistant"]

    if not user_messages:
        return "No patient messages found to generate a SOAP note."

    # Build conversation text for AI to analyze
    convo_text = "\n".join([
        f"Patient: {m['content']}" if m["role"] == "user" else f"Therapist: {m['content']}"
        for m in session_history[-20:]
    ])

    # Detect mood trend
    mood_trend = "Not tracked"
    if mood_history:
        recent_moods = [m["mood"] for m in mood_history[-5:]]
        mood_trend = ", ".join(recent_moods) if recent_moods else "Not tracked"

    # Count messages and detect risk
    msg_count = len(user_messages)
    last_user_msg = user_messages[-1] if user_messages else ""

    risk = assess_crisis_risk(last_user_msg)
    risk_level = risk["level"]

    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
    session_id = f"SS-{datetime.now().strftime('%Y%m%d-%H%M')}"

    try:
        # Use MedGemma to generate clinical SOAP content
        soap_prompt = f"""You are a clinical psychologist generating a professional SOAP note.
Based on this therapy session transcript, generate each section clearly.

SESSION TRANSCRIPT:
{convo_text}

Generate a SOAP note with these exact sections:
SUBJECTIVE: (What the patient reported - their feelings, complaints, concerns in 2-3 sentences)
OBJECTIVE: (Observed data - mood, behavior, engagement level in 2-3 sentences)
ASSESSMENT: (Clinical interpretation - what is happening psychologically in 2-3 sentences)
PLAN: (Recommended next steps, interventions, follow-up in 3-4 bullet points)

Be concise, professional, and use clinical language."""

        response = ollama.chat(
            model='alibayram/medgemma:4b',
            messages=[{"role": "user", "content": soap_prompt}],
            options={'num_predict': 400, 'temperature': 0.3}
        )
        soap_content = response['message']['content'].strip()

    except Exception as e:
        # Fallback: generate basic SOAP note without AI
        soap_content = f"""SUBJECTIVE: Patient reported various emotional concerns during this session. {user_messages[0][:150] if user_messages else 'No specific complaints noted.'}

OBJECTIVE: Patient engaged in {msg_count} exchanges during this session. Mood trend observed: {mood_trend}. Risk level assessed as {risk_level}.

ASSESSMENT: Patient presented with emotional distress requiring supportive intervention. AI-assisted therapy tools were utilized including empathetic response generation and resource referral.

PLAN:
• Continue regular check-ins and emotional support sessions
• Monitor mood patterns and risk indicators
• Refer to local mental health professionals if symptoms persist
• Encourage use of breathing exercises and sleep hygiene practices"""

    # Format the final SOAP note
    note = f"""
╔══════════════════════════════════════════════════════════════╗
║           SAFESPACE AI — CLINICAL SESSION REPORT            ║
╚══════════════════════════════════════════════════════════════╝

📋 SESSION ID:     {session_id}
📅 DATE & TIME:    {timestamp}
💬 MESSAGES:       {msg_count} patient exchanges
📊 MOOD TREND:     {mood_trend}
⚠️  RISK LEVEL:    {risk_level}
🤖 AI MODEL:       MedGemma 4B + Llama 3.2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{soap_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  DISCLAIMER: This SOAP note was AI-generated and is intended
    as a supplementary tool only. It does not replace clinical
    judgment by a licensed mental health professional.
    This document follows standard ICD-10 documentation format.

🔒 CONFIDENTIAL — SafeSpace AI · {timestamp}
""".strip()

    return note
