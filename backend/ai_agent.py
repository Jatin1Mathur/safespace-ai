"""ReAct agent and its tools.

Every tool has an explicit Pydantic argument schema. LangChain validates the
model's tool-call arguments against it before the function runs, so a
malformed call (empty city, unknown breathing technique, ...) is rejected
instead of executed.

The emergency call is deliberately NOT exposed to the model: placing a phone
call is an irreversible action, so it only happens on the deterministic
crisis path in the graph (see agent_graph.py).
"""

import os
import warnings

# Silence library deprecation warnings before the imports below trigger them
warnings.filterwarnings("ignore")

from typing import Literal  # noqa: E402

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from tools import (  # noqa: E402
    OLLAMA_TIMEOUT,
    find_clinics_by_city,
    find_pharmacies,
    get_breathing_exercise,
    get_crisis_hotlines,
    get_sleep_advice,
    query_medgemma,
)

AGENT_MODEL = os.getenv("AGENT_MODEL", "llama3.2")


# ── Tool argument schemas ────────────────────────────────────────────────────

class SpecialistArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       description="The user's message or question, in their own words.")


class LocationArgs(BaseModel):
    location: str = Field(..., min_length=2, max_length=80, pattern=r"^[^\d{}<>\[\]]+$",
                          description="City or region name, e.g. 'Stuttgart'.")


class CityArgs(BaseModel):
    city: str = Field(..., min_length=2, max_length=80, pattern=r"^[^\d{}<>\[\]]+$",
                      description="City name to search in, e.g. 'Munich'.")


class BreathingArgs(BaseModel):
    technique: Literal["box", "478", "calm"] = Field(
        "box", description="'box' for anxiety, '478' for sleep, 'calm' for a gentle start.")


class NoArgs(BaseModel):
    """This tool takes no arguments."""


# ── Tools ────────────────────────────────────────────────────────────────────

@tool(args_schema=SpecialistArgs)
def ask_mental_health_specialist(query: str) -> str:
    """Generate a supportive, empathetic answer with the MedGemma model.
    Use this for general emotional and mental-health conversation."""
    return query_medgemma(query)


@tool(args_schema=LocationArgs)
def find_nearby_therapists_by_location(location: str) -> str:
    """Explain how to find a licensed therapist near the given location.
    Use when the user asks for a therapist, psychologist or counsellor."""
    # No live therapist directory is integrated, so point to official
    # services instead of returning names or numbers that could be wrong.
    return (
        f"I can't look up individual therapists in {location}, "
        "but these official services can help you find one:\n"
        "- Germany: call 116117 (Patientenservice) for a psychotherapy "
        "appointment, or search at https://arztsuche.116117.de\n"
        "- Elsewhere: ask your family doctor or your health insurer for a "
        "referral to a licensed therapist nearby.\n"
        "If you are in crisis, please call 112 (EU) or your local "
        "emergency number now."
    )


@tool(args_schema=CityArgs)
def nearby_clinics_tool(city: str) -> str:
    """Find real hospitals and clinics in a city using OpenStreetMap."""
    return find_clinics_by_city(city)


@tool(args_schema=NoArgs)
def nearby_pharmacy_tool() -> str:
    """Find pharmacies near the user's current location."""
    return find_pharmacies()


@tool(args_schema=NoArgs)
def crisis_hotlines_tool() -> str:
    """Return crisis helpline numbers for the user's country."""
    return get_crisis_hotlines()


@tool(args_schema=BreathingArgs)
def breathing_exercise_tool(technique: str = "box") -> str:
    """Give a guided breathing exercise for anxiety, panic or trouble sleeping."""
    return get_breathing_exercise(technique)


@tool(args_schema=NoArgs)
def sleep_advice_tool() -> str:
    """Give CBT-based sleep hygiene advice for sleep problems or insomnia."""
    return get_sleep_advice()


tools = [
    ask_mental_health_specialist,
    find_nearby_therapists_by_location,
    nearby_clinics_tool,
    nearby_pharmacy_tool,
    crisis_hotlines_tool,
    breathing_exercise_tool,
    sleep_advice_tool,
]

SYSTEM_PROMPT = """
You are SafeSpace AI, a supportive mental-health and medical information assistant.
You are not a doctor or therapist and never claim to be one.
You ONLY discuss health, mental health, and medical topics.
If asked about anything unrelated to health, politely redirect to health topics.

Use the tools when they help:
1. `ask_mental_health_specialist` - emotional and mental-health conversation
2. `find_nearby_therapists_by_location` - how to find a licensed therapist
3. `nearby_clinics_tool` - hospitals/clinics by city name
4. `nearby_pharmacy_tool` - pharmacies near the user
5. `crisis_hotlines_tool` - crisis helpline numbers
6. `breathing_exercise_tool` - guided breathing for anxiety/panic
7. `sleep_advice_tool` - help with sleep problems

You can see earlier messages of this conversation; use them for context.
Always respond with warmth and professionalism, and never ignore signs of distress.
"""


def build_react_agent(model: str = AGENT_MODEL, llm=None):
    """Build the ReAct agent (reason -> call tool -> observe -> answer)."""
    llm = llm or ChatOllama(model=model, temperature=0.2,
                            client_kwargs={"timeout": OLLAMA_TIMEOUT})
    return create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


react_agent = build_react_agent()


def summarize_agent_run(new_messages: list[AnyMessage]) -> tuple[str, str | None, dict]:
    """Return (last tool called, final answer, token usage) from an agent run."""
    tool_called = "None"
    final_response = None
    usage = {"input_tokens": 0, "output_tokens": 0}
    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            tool_called = msg.name or "None"
        elif isinstance(msg, AIMessage):
            meta = getattr(msg, "usage_metadata", None) or {}
            usage["input_tokens"] += meta.get("input_tokens", 0)
            usage["output_tokens"] += meta.get("output_tokens", 0)
            if msg.content and not msg.tool_calls:
                final_response = msg.content if isinstance(msg.content, str) else str(msg.content)
    return tool_called, final_response, usage
