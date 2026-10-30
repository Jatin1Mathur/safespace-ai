import pytest


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Keep tests from writing to the real session_memory.json."""
    import tools

    monkeypatch.setattr(tools, "MEMORY_FILE", str(tmp_path / "session_memory.json"))
    # Each test starts with the map servers "not known to be down"
    monkeypatch.setattr(tools, "_overpass_down_until", 0.0)


@pytest.fixture
def fake_services(monkeypatch):
    """Replace every external service the graph can reach with a fake,
    and give each test a fresh graph (so conversation memory is empty)."""
    import agent_graph

    calls = {"emergency": 0, "agent": 0, "agent_contexts": [], "agent_reply": ("None", "FAKE_AGENT")}

    def fake_emergency():
        calls["emergency"] += 1
        return "FAKE_EMERGENCY_CALL"

    def fake_agent(context):
        calls["agent"] += 1
        calls["agent_contexts"].append(list(context))
        tool, answer = calls["agent_reply"]
        return tool, answer, {"input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(agent_graph, "call_emergency", fake_emergency)
    monkeypatch.setattr(agent_graph, "get_crisis_hotlines", lambda: "FAKE_HOTLINES")
    monkeypatch.setattr(agent_graph, "query_medgemma", lambda prompt: "FAKE_MEDGEMMA")
    monkeypatch.setattr(agent_graph, "find_pharmacies", lambda: "FAKE_PHARMACIES")
    monkeypatch.setattr(agent_graph, "find_clinics_by_city", lambda city: f"FAKE_CLINICS {city}")
    monkeypatch.setattr(agent_graph, "generate_session_summary", lambda history: "FAKE_SUMMARY")
    monkeypatch.setattr(agent_graph, "generate_soap_note", lambda history, moods: "FAKE_SOAP")
    monkeypatch.setattr(agent_graph, "run_react_agent", fake_agent)
    monkeypatch.setattr(agent_graph, "graph", agent_graph.build_graph())
    return calls


@pytest.fixture
def set_agent_reply(fake_services):
    """Choose what the fake ReAct agent answers: set_agent_reply(tool, text)."""
    def _set(tool_called, answer):
        fake_services["agent_reply"] = (tool_called, answer)
    return _set
