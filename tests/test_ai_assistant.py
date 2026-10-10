"""Provider-neutral AI summary/chat: request shapes, retries and parsing (no network)."""
import json

import pandas as pd
import pytest

import ai_assistant as ai

RESULTS = pd.DataFrame([{"title": "T1", "abstract": "A1"}, {"title": "T2", "abstract": "A2"}])


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


def fake_post(responses, calls):
    def post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "body": dict(json)})
        return responses.pop(0)
    return post


def test_openai_compatible_summary_and_numbering(monkeypatch):
    calls = []
    answer = {"summary": "Themes [1].", "questions": ["Q1?", " ", "Q2?"]}
    monkeypatch.setattr(ai.requests, "post", fake_post(
        [Resp(200, {"choices": [{"message": {"content": "```json\n" + json.dumps(answer) + "\n```"}}]})], calls))
    summary, questions = ai.analyze_results(RESULTS, "openrouter", "k", "some/model")
    assert (summary, questions) == ("Themes [1].", ["Q1?", "Q2?"])
    call = calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k" and call["body"]["model"] == "some/model"
    assert call["body"]["response_format"] == {"type": "json_object"}
    assert "[1] T1\nA1" in call["body"]["messages"][-1]["content"]


def test_retries_without_temperature_and_json_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(ai.requests, "post", fake_post([
        Resp(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.7"}}),
        Resp(200, {"choices": [{"message": {"content": "Plain summary."}}]}),
    ], calls))
    assert ai.analyze_results(RESULTS, "openai", "k") == ("Plain summary.", [])
    assert "temperature" not in calls[1]["body"] and "response_format" not in calls[1]["body"]
    assert calls[0]["body"]["model"] == ai.PROVIDERS["openai"].default_model


def test_anthropic_chat_request(monkeypatch):
    calls = []
    monkeypatch.setattr(ai.requests, "post", fake_post(
        [Resp(200, {"content": [{"type": "text", "text": "Paper [2] says so."}]})], calls))
    history = [{"role": "user", "content": "What?"}, {"role": "assistant", "content": "This."},
               {"role": "user", "content": "And?"}]
    assert ai.chat_with_context(history, RESULTS, "anthropic", "k", "claude-sonnet-5-5") == "Paper [2] says so."
    call = calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages" and call["headers"]["x-api-key"] == "k"
    assert [m["role"] for m in call["body"]["messages"]] == ["user", "assistant", "user"]
    assert "[2] T2\nA2" in call["body"]["system"]


def test_errors_become_readable_messages(monkeypatch):
    monkeypatch.setattr(ai.requests, "post", fake_post([Resp(401, {"error": {"message": "bad key"}})], []))
    summary, questions = ai.analyze_results(RESULTS, "groq", "k")
    assert "rejected" in summary and questions == []
    assert ai.analyze_results(RESULTS, "groq", "")[0] == "Add your API key first."


def test_gemini_goes_through_its_sdk_path(monkeypatch):
    seen = {}

    def fake(api_key, model, system, messages, want_json, temperature):
        seen.update(model=model, want_json=want_json)
        return json.dumps({"summary": "S", "questions": ["Q?"]})
    monkeypatch.setattr(ai, "_gemini", fake)
    assert ai.analyze_results(RESULTS, "google", "k") == ("S", ["Q?"])
    assert seen == {"model": ai.PROVIDERS["google"].default_model, "want_json": True}


def test_model_list_filters_and_puts_default_first(monkeypatch):
    class R(Resp):
        def raise_for_status(self):
            pass
    monkeypatch.setattr(ai.requests, "get", lambda *a, **k: R(200, {"data": [
        {"id": "text-embedding-3-large"}, {"id": "gpt-5"}, {"id": "gpt-5-mini"}, {"id": "whisper-1"}]}))
    assert ai.list_models("openai", "k") == ["gpt-5-mini", "gpt-5"]


@pytest.mark.parametrize("provider_id", list(ai.PROVIDERS))
def test_every_provider_is_complete(provider_id):
    p = ai.PROVIDERS[provider_id]
    assert p.label and p.default_model and p.key_url.startswith("https://")
    assert p.kind in ("gemini", "anthropic", "openai") and (p.kind == "gemini" or p.base_url.startswith("https://"))
