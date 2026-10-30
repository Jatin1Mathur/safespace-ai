"""Tests for the graph structure, the typed tool schemas and the real ReAct loop.

The ReAct tests use a scripted chat model instead of Ollama, so the full
reason -> tool call -> observe -> answer loop runs offline and deterministically.
"""

import agent_graph
import ai_agent
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError

# ── Graph structure ──────────────────────────────────────────────────────────

def test_graph_has_router_and_three_branches():
    drawable = agent_graph.build_graph().get_graph()
    nodes = set(drawable.nodes)
    assert {"router", "crisis", "fixed_response", "agent"} <= nodes
    conditional = {(e.source, e.target) for e in drawable.edges if e.conditional}
    assert conditional == {("router", "crisis"), ("router", "fixed_response"), ("router", "agent")}


@pytest.mark.parametrize(
    "route, branch",
    [("crisis_high", "crisis"), ("crisis_medium", "crisis"), ("agent", "agent"),
     ("greeting", "fixed_response"), ("off_topic", "fixed_response"), ("clinic", "fixed_response")],
)
def test_select_branch(route, branch):
    assert agent_graph.select_branch({"route": route}) == branch


def test_every_fixed_route_has_a_handler():
    from typing import get_args

    from routing import AGENT_ROUTES, CRISIS_ROUTES, Route
    fixed_routes = set(get_args(Route)) - CRISIS_ROUTES - AGENT_ROUTES
    assert fixed_routes == set(agent_graph.FIXED_HANDLERS)


def test_state_is_checkpointed_per_thread(fake_services):
    agent_graph.run_turn("I feel stressed about exams", "thread-1")
    agent_graph.run_turn("I feel stressed again today", "thread-1")
    snapshot = agent_graph.graph.get_state({"configurable": {"thread_id": "thread-1"}})
    humans = [m.content for m in snapshot.values["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["I feel stressed about exams", "I feel stressed again today"]
    other = agent_graph.graph.get_state({"configurable": {"thread_id": "thread-2"}})
    assert not other.values


# ── Tool schemas ─────────────────────────────────────────────────────────────

def test_emergency_call_is_not_exposed_to_the_model():
    names = {t.name for t in ai_agent.tools}
    assert "emergency_call_tool" not in names


def test_every_tool_has_an_explicit_pydantic_schema():
    for t in ai_agent.tools:
        assert t.args_schema is not None
        assert t.args_schema.__module__ == "ai_agent", t.name


@pytest.mark.parametrize(
    "tool, bad_args",
    [
        (ai_agent.nearby_clinics_tool, {"city": ""}),
        (ai_agent.nearby_clinics_tool, {"city": "12345"}),
        (ai_agent.nearby_clinics_tool, {}),
        (ai_agent.breathing_exercise_tool, {"technique": "hold-for-60s"}),
        (ai_agent.find_nearby_therapists_by_location, {"location": "x" * 200}),
    ],
)
def test_invalid_tool_arguments_are_rejected_before_execution(tool, bad_args, monkeypatch):
    executed = []
    monkeypatch.setattr(ai_agent, "find_clinics_by_city", lambda city: executed.append(city))
    with pytest.raises(ValidationError):
        tool.invoke(bad_args)
    assert executed == []


# ── Real ReAct loop with a scripted model ────────────────────────────────────

class ScriptedChatModel(BaseChatModel):
    """Returns pre-written AIMessages one by one; supports bind_tools."""
    script: list
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_react_agent_calls_tool_then_answers(monkeypatch):
    monkeypatch.setattr(ai_agent, "find_clinics_by_city", lambda city: f"Clinics in {city}: Klinikum")
    model = ScriptedChatModel(script=[
        tool_call("nearby_clinics_tool", {"city": "Stuttgart"}),
        AIMessage(content="Klinikum Stuttgart is close to you.",
                  usage_metadata={"input_tokens": 50, "output_tokens": 9, "total_tokens": 59}),
    ])
    monkeypatch.setattr(ai_agent, "react_agent", ai_agent.build_react_agent(llm=model))

    tool_called, answer, usage = agent_graph.run_react_agent([HumanMessage("Clinics in Stuttgart?")])

    assert tool_called == "nearby_clinics_tool"
    assert answer == "Klinikum Stuttgart is close to you."
    assert usage["output_tokens"] == 9
    observation = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert observation[0].content == "Clinics in Stuttgart: Klinikum"


def test_react_agent_gets_schema_error_instead_of_running_tool(monkeypatch):
    ran = []
    monkeypatch.setattr(ai_agent, "find_clinics_by_city", lambda city: ran.append(city) or "x")
    model = ScriptedChatModel(script=[
        tool_call("nearby_clinics_tool", {"city": ""}),
        AIMessage(content="Which city should I search in?"),
    ])
    monkeypatch.setattr(ai_agent, "react_agent", ai_agent.build_react_agent(llm=model))

    tool_called, answer, _ = agent_graph.run_react_agent([HumanMessage("Find a clinic")])

    assert ran == []                       # tool body never executed
    error = [m for m in model.seen[-1] if isinstance(m, ToolMessage)][0]
    assert error.status == "error"         # the model observed a validation error
    assert answer == "Which city should I search in?"
