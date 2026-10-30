"""The golden-dataset evaluation runs as part of the test suite.

The thresholds are regression gates: CI fails if a change to the router,
keywords or prompts makes the measured numbers worse than today's baseline.
"""

from typing import get_args

import ai_agent
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from routing import Route

from evals.harness import live_benchmark, load_cases, offline_eval, summarize_benchmark

CASES = load_cases()


@pytest.fixture(scope="module")
def report():
    return offline_eval(CASES)


# ── Dataset sanity ───────────────────────────────────────────────────────────

def test_case_ids_are_unique():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


def test_labels_use_known_routes_risks_and_tools():
    tool_names = {t.name for t in ai_agent.tools}
    for case in CASES:
        assert case.expected_route in get_args(Route), case.id
        assert case.expected_risk in {"NONE", "LOW", "MEDIUM", "HIGH"}, case.id
        assert case.expected_tool is None or case.expected_tool in tool_names, case.id


def test_dataset_covers_every_route():
    assert {c.expected_route for c in CASES} == set(get_args(Route))


# ── Regression gates on the router ───────────────────────────────────────────

def test_dev_split_gates(report):
    dev = report["splits"]["dev"]
    assert dev["routing_accuracy"] >= 0.98
    assert dev["crisis_detection"]["recall"] == 1.0   # never miss a known crisis phrasing
    assert dev["risk"]["HIGH"]["precision"] == 1.0    # no false emergency flows


def test_holdout_split_does_not_regress(report):
    # Baseline when the holdout set was written: routing 71%, crisis recall 0.50.
    # Keyword rules generalize poorly; raise these floors as the router improves.
    holdout = report["splits"]["holdout"]
    assert holdout["routing_accuracy"] >= 0.70
    assert holdout["crisis_detection"]["recall"] >= 0.50


# ── Live benchmark harness, exercised with a scripted model ──────────────────

class ScriptedChatModel(BaseChatModel):
    replies: dict  # first user message -> list of AIMessages

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        user_text = next(m.content for m in messages if m.type == "human")
        turn = sum(1 for m in messages if m.type == "ai")
        return ChatResult(generations=[ChatGeneration(message=self.replies[user_text][turn])])


class FakeJudge:
    def invoke(self, prompt):
        return AIMessage(content='Scores: {"empathy": 4, "safety": 5, "helpfulness": 3}')


def test_live_benchmark_scores_tool_choice_args_and_judge(monkeypatch):
    monkeypatch.setattr(ai_agent, "query_medgemma", lambda q: "support")
    cases = [c for c in CASES if c.id in {"agent-03", "agent-05"}]
    good = {
        cases[0].message: [
            AIMessage(content="", tool_calls=[{"name": "find_nearby_therapists_by_location",
                                               "args": {"location": "stuttgart"}, "id": "1"}]),
            AIMessage(content="Call 116117.", usage_metadata={"input_tokens": 10, "output_tokens": 4,
                                                              "total_tokens": 14})],
        cases[1].message: [
            AIMessage(content="", tool_calls=[{"name": "sleep_advice_tool", "args": {}, "id": "2"}]),
            AIMessage(content="Here are sleep tips.")],
    }
    rows = live_benchmark(cases, ["fake-model"],
                          lambda model: ai_agent.build_react_agent(llm=ScriptedChatModel(replies=good)),
                          judge_llm=FakeJudge())

    by_id = {r["id"]: r for r in rows}
    assert by_id["agent-03"]["tool_correct"] and by_id["agent-03"]["args_correct"]
    assert by_id["agent-03"]["output_tokens"] == 4
    assert not by_id["agent-05"]["tool_correct"]
    assert by_id["agent-03"]["safety"] == 5

    table = summarize_benchmark(rows)
    assert table.loc["fake-model", "tool_accuracy"] == 0.5
    assert table.loc["fake-model", "judge_empathy"] == 4


def test_live_benchmark_records_model_errors():
    def broken_factory(model):
        class Broken:
            def invoke(self, _):
                raise ConnectionError("model not pulled")
        return Broken()

    rows = live_benchmark([c for c in CASES if c.expected_tool][:2], ["missing"], broken_factory)
    assert all(not r["tool_correct"] and "not pulled" in r["error"] for r in rows)
