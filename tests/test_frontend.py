"""Tests for how the Gradio frontend shows backend replies."""

import gradio_frontend


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_crisis_reply_is_never_hidden(monkeypatch):
    reply = {"response": "Call Telefonseelsorge 0800 111 0 111 (error 500 in log)",
             "tool_called": "emergency_call_tool", "mood": "depressed", "risk": "HIGH"}
    monkeypatch.setattr(gradio_frontend.requests, "post", lambda *a, **k: FakeResponse(reply))
    response, tool, mood, risk = gradio_frontend.call_backend("i want to die")
    assert "0800 111 0 111" in response
    assert risk == "HIGH"


def test_technical_error_is_hidden_for_normal_messages(monkeypatch):
    reply = {"response": "Traceback: ConnectionError", "tool_called": "x",
             "mood": "neutral", "risk": "LOW"}
    monkeypatch.setattr(gradio_frontend.requests, "post", lambda *a, **k: FakeResponse(reply))
    response, *_ = gradio_frontend.call_backend("hello")
    assert response == gradio_frontend.GENERAL_SERVICE_MESSAGE
