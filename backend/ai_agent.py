import warnings
warnings.filterwarnings("ignore")

from langchain.tools import tool
from tools import query_medgemma, call_emergency, find_clinics_by_city, find_pharmacies, get_crisis_hotlines, get_breathing_exercise, get_sleep_advice


@tool
def ask_mental_health_specialist(query: str) -> str:
    """
    Generate a therapeutic response using the MedGemma model.
    Use this for all general user queries, mental health questions, emotional concerns,
    or to offer empathetic, evidence-based guidance in a conversational tone.
    """
    return query_medgemma(query)


@tool
def emergency_call_tool() -> str:
    """
    Place an emergency call via Twilio.
    Use this IMMEDIATELY if the user expresses suicidal ideation,
    intent to self-harm, or describes a mental health emergency.
    """
    return call_emergency()


@tool
def find_nearby_therapists_by_location(location: str) -> str:
    """
    Finds licensed therapists near the specified location.
    """
    return (
        f"Here are some therapists near {location}:\n"
        "- Dr. Ayesha Kapoor - +49 (555) 123-4567\n"
        "- Dr. James Patel - +49 (555) 987-6543\n"
        "- MindCare Counseling Center - +49 (555) 222-3333"
    )


@tool
def nearby_clinics_tool(city: str) -> str:
    """
    Finds real hospitals and clinics near a given city using OpenStreetMap.
    Use when user asks for hospitals, clinics, or medical facilities.
    """
    return find_clinics_by_city(city)


@tool
def nearby_pharmacy_tool() -> str:
    """
    Finds pharmacies near the user's current location automatically.
    Use when user asks for pharmacy or medication.
    """
    return find_pharmacies()


@tool
def crisis_hotlines_tool() -> str:
    """
    Returns crisis helpline numbers based on user's location.
    Use when user needs a hotline or crisis support number.
    """
    return get_crisis_hotlines()


@tool
def breathing_exercise_tool() -> str:
    """
    Provides a guided breathing exercise.
    Use when user is anxious, panicking, or needs to calm down.
    """
    return get_breathing_exercise("box")


@tool
def sleep_advice_tool() -> str:
    """
    Provides CBT-based sleep hygiene advice.
    Use when user mentions sleep problems or insomnia.
    """
    return get_sleep_advice()


from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


tools = [
    ask_mental_health_specialist,
    emergency_call_tool,
    find_nearby_therapists_by_location,
    nearby_clinics_tool,
    nearby_pharmacy_tool,
    crisis_hotlines_tool,
    breathing_exercise_tool,
    sleep_advice_tool,
]

llm = ChatOllama(model="llama3.2", temperature=0.2)

graph = create_react_agent(llm, tools=tools)

SYSTEM_PROMPT = """
You are SafeSpace AI, a compassionate mental health and medical support assistant.
You ONLY discuss health, mental health, and medical topics.
If asked about anything unrelated to health, politely redirect to health topics.

You have access to these tools:
1. `ask_mental_health_specialist` - for all emotional and mental health queries
2. `emergency_call_tool` - IMMEDIATELY if user is in crisis or suicidal
3. `find_nearby_therapists_by_location` - find therapists by city
4. `nearby_clinics_tool` - find hospitals/clinics by city name
5. `nearby_pharmacy_tool` - find pharmacies near user
6. `crisis_hotlines_tool` - get crisis helpline numbers
7. `breathing_exercise_tool` - guided breathing for anxiety/panic
8. `sleep_advice_tool` - help with sleep problems

Always respond with warmth, empathy, and professionalism.
Never ignore signs of distress or crisis.
"""


def parse_response(stream):
    tool_called_name = "None"
    final_response = None

    for s in stream:
        tool_data = s.get('tools')
        if tool_data:
            tool_messages = tool_data.get('messages')
            if tool_messages and isinstance(tool_messages, list):
                for msg in tool_messages:
                    tool_called_name = getattr(msg, 'name', 'None')

        agent_data = s.get('agent')
        if agent_data:
            messages = agent_data.get('messages')
            if messages and isinstance(messages, list):
                for msg in messages:
                    if msg.content:
                        final_response = msg.content

    return tool_called_name, final_response
        