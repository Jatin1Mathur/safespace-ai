"""SafeSpace agent harness as an explicit LangGraph StateGraph.

    START -> router --(conditional edge)--> crisis          -> END
                                        +-> fixed_response  -> END
                                        +-> agent (ReAct)   -> END

* router          deterministic: crisis check, topic filter, intent rules.
                  No LLM runs here, so it is fast, testable and auditable.
* crisis          HIGH risk: emergency flow + hotlines, never model text.
                  MEDIUM risk: supportive answer + hotlines.
* fixed_response  greetings, tools with a known answer (breathing, sleep,
                  hotlines, clinics, pharmacies, mood, summary, SOAP note).
* agent           open conversation: ReAct agent with tool calling, given
                  the recent conversation from this thread as context.

Conversation memory is a LangGraph checkpointer keyed by `thread_id`, so
every user/session has its own history and nothing leaks between them.
"""

import re
from typing import Annotated, Callable, Literal, TypedDict

import ai_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from routing import AGENT_ROUTES, CRISIS_ROUTES, RiskLevel, Route, classify_request, extract_city
from tools import (
    call_emergency,
    find_clinics_by_city,
    find_pharmacies,
    generate_session_summary,
    generate_soap_note,
    get_breathing_exercise,
    get_crisis_hotlines,
    get_mood_history,
    get_mood_summary,
    get_sleep_advice,
    log_mood,
    query_medgemma,
)

# How many earlier messages the ReAct agent sees as context
AGENT_CONTEXT_MESSAGES = 12


class SafeSpaceState(TypedDict, total=False):
    """Shared state that flows through every node of the graph."""
    messages: Annotated[list[AnyMessage], add_messages]  # persisted per thread
    route: Route
    risk: RiskLevel
    mood: str
    tool_called: str
    token_usage: dict


GREETING_RESPONSE = """🌿 Hi, I'm SafeSpace, an AI support assistant (not a therapist).

You can talk to me about how you're feeling, ask for a breathing exercise or sleep tips,
or find hospitals, pharmacies and crisis hotlines. How are you doing today? 💚"""

OFF_TOPIC_RESPONSE = """🌿 I'm SafeSpace — a mental health and medical support assistant.

I'm here to help with:
• 💬 Mental health support & emotional guidance
• 🏥 Finding hospitals, clinics & pharmacies nearby
• 🫁 Breathing exercises & relaxation techniques
• 😴 Sleep improvement advice
• 📞 Crisis hotlines & emergency support
• 📊 Mood tracking & session summaries

I'm not able to help with topics outside of healthcare and wellbeing. Is there something health-related I can support you with today? 💚

If you're going through something difficult, you can tell me. In an emergency, call 112 (EU) or your local emergency number."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _thread_id(config: RunnableConfig | None) -> str | None:
    return ((config or {}).get("configurable") or {}).get("thread_id")


def _last_user_text(state: SafeSpaceState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _as_dicts(messages: list[AnyMessage]) -> list[dict]:
    """Convert LangChain messages to the {"role", "content"} dicts the tools use."""
    out = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        out.append({"role": role, "content": str(msg.content)})
    return out


def looks_like_raw_tool_call(text: str) -> bool:
    """Small local models sometimes write a tool call as plain JSON text
    (e.g. {"name": "breathing_exercise_tool", "parameters": {...}})
    instead of really calling the tool. That must never reach the user."""
    return bool(re.search(r'"name"\s*:\s*"\w+"', text)) and (
        '"parameters"' in text or '"arguments"' in text)


def _reply(text: str, tool_called: str) -> dict:
    return {"messages": [AIMessage(content=text)], "tool_called": tool_called}


# ── Nodes ────────────────────────────────────────────────────────────────────

def router_node(state: SafeSpaceState, config: RunnableConfig) -> dict:
    """Classify the latest message. Deterministic; no LLM call."""
    decision = classify_request(_last_user_text(state))
    if decision.mood != "neutral":
        log_mood(decision.mood, _thread_id(config))
    return {"route": decision.route, "risk": decision.risk, "mood": decision.mood,
            "token_usage": {"input_tokens": 0, "output_tokens": 0}}


def select_branch(state: SafeSpaceState) -> Literal["crisis", "agent", "fixed_response"]:
    """Conditional edge out of the router."""
    route = state["route"]
    if route in CRISIS_ROUTES:
        return "crisis"
    if route in AGENT_ROUTES:
        return "agent"
    return "fixed_response"


def crisis_node(state: SafeSpaceState) -> dict:
    hotlines = get_crisis_hotlines()
    if state["route"] == "crisis_high":
        call_result = call_emergency()
        text = f"""🚨 I'm very concerned about what you've shared.

You are not alone, and your life has value.

{hotlines}

Please reach out to one of these lines right now — they are free, confidential, and available 24/7.

{call_result}"""
        return _reply(text, "emergency_call_tool")

    supportive = query_medgemma(_last_user_text(state))
    return _reply(f"{supportive}\n\n---\n💙 Support is available:\n{hotlines}", "crisis_support")


def _clinic_answer(text: str) -> str:
    city = extract_city(text)
    if not city:
        return "📍 Which city should I search in?\n\nFor example: *find hospitals in Hof* or *clinics in Munich*"
    return find_clinics_by_city(city)


# route -> (handler(text, earlier_messages, thread_id), tool name reported to the client)
FIXED_HANDLERS: dict[str, tuple[Callable[[str, list, str | None], str], str]] = {
    "off_topic": (lambda text, hist, tid: OFF_TOPIC_RESPONSE, "topic_filter"),
    "greeting": (lambda text, hist, tid: GREETING_RESPONSE, "greeting"),
    "breathing": (lambda text, hist, tid: get_breathing_exercise("478" if "sleep" in text.lower() else "box"),
                  "breathing_exercise"),
    "sleep": (lambda text, hist, tid: get_sleep_advice(text), "sleep_advice"),
    "mood": (lambda text, hist, tid: get_mood_summary(tid), "mood_tracker"),
    "hotline": (lambda text, hist, tid: get_crisis_hotlines(), "hotline_directory"),
    "summary": (lambda text, hist, tid: generate_session_summary(_as_dicts(hist)), "session_summary"),
    "soap": (lambda text, hist, tid: generate_soap_note(_as_dicts(hist), get_mood_history(tid)),
             "soap_note_generator"),
    "pharmacy": (lambda text, hist, tid: find_pharmacies(), "pharmacy_finder"),
    "clinic": (lambda text, hist, tid: _clinic_answer(text), "clinic_finder"),
}


def fixed_response_node(state: SafeSpaceState, config: RunnableConfig) -> dict:
    handler, tool_name = FIXED_HANDLERS[state["route"]]
    earlier = state.get("messages", [])[:-1]  # conversation before this message
    return _reply(handler(_last_user_text(state), earlier, _thread_id(config)), tool_name)


def run_react_agent(context: list[AnyMessage]) -> tuple[str, str | None, dict]:
    """Run the ReAct agent on the conversation and summarize what it did."""
    result = ai_agent.react_agent.invoke({"messages": context})
    new_messages = result["messages"][len(context):]
    return ai_agent.summarize_agent_run(new_messages)


def agent_node(state: SafeSpaceState) -> dict:
    """Open conversation: ReAct agent with tools, falling back to MedGemma."""
    context = state.get("messages", [])[-AGENT_CONTEXT_MESSAGES:]
    try:
        tool_called, answer, usage = run_react_agent(context)
    except Exception:
        # e.g. Ollama is not running: never crash, use the fallback below
        tool_called, answer, usage = "None", None, {"input_tokens": 0, "output_tokens": 0}

    # No answer, or a tool call written as text instead of executed -> fallback
    if not answer or looks_like_raw_tool_call(answer):
        answer = query_medgemma(_last_user_text(state))
        tool_called = "medgemma_fallback"

    return {**_reply(answer, tool_called), "token_usage": usage}


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    builder = StateGraph(SafeSpaceState)
    builder.add_node("router", router_node)
    builder.add_node("crisis", crisis_node)
    builder.add_node("fixed_response", fixed_response_node)
    builder.add_node("agent", agent_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", select_branch,
                                  ["crisis", "fixed_response", "agent"])
    builder.add_edge("crisis", END)
    builder.add_edge("fixed_response", END)
    builder.add_edge("agent", END)

    # In-memory per-thread history; swap for SqliteSaver/PostgresSaver to persist
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


graph = build_graph()


def run_turn(message: str, thread_id: str, app_graph=None) -> SafeSpaceState:
    """Send one user message through the graph for a given conversation thread."""
    app_graph = app_graph or graph
    return app_graph.invoke({"messages": [HumanMessage(content=message)]},
                            config={"configurable": {"thread_id": thread_id}})
