import json

import pandas as pd

import gemini_handler


class _FakeModels:
    def __init__(self, text):
        self.text, self.calls = text, []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return type("Resp", (), {"text": self.text})()


def _client(text):
    return type("Client", (), {"models": _FakeModels(text)})()


RESULTS = pd.DataFrame([{"title": "T1", "abstract": "A1"}, {"title": "T2", "abstract": "A2"}])


def test_analyze_results_single_call(monkeypatch):
    client = _client(json.dumps({"summary": "Themes [1].", "questions": ["Q1?", " ", "Q2?"]}))
    monkeypatch.setattr(gemini_handler, "get_client", lambda key: client)

    summary, questions = gemini_handler.analyze_results(RESULTS, "key")

    assert summary == "Themes [1]."
    assert questions == ["Q1?", "Q2?"]
    assert len(client.models.calls) == 1
    config = client.models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert "[1] T1\nA1" in client.models.calls[0]["contents"].text


def test_analyze_results_non_json_falls_back_to_text(monkeypatch):
    monkeypatch.setattr(gemini_handler, "get_client", lambda key: _client("Plain summary."))
    assert gemini_handler.analyze_results(RESULTS, "key") == ("Plain summary.", [])


def test_analyze_results_without_key():
    summary, questions = gemini_handler.analyze_results(RESULTS, "")
    assert summary.startswith("Error") and questions == []
