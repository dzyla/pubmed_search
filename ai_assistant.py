"""
AI summary and chat over search results, with the user's own key for one of
several providers. Three API styles cover them all:

- Google Gemini (google-genai SDK)
- Anthropic Claude (Messages API)
- OpenAI-compatible chat completions: OpenAI, OpenRouter, Groq, Mistral, DeepSeek

Keys are passed per call and never stored or cached here. Model lists come
from each provider's own /models endpoint, so new models appear without code
changes; the defaults below are only a starting point.
"""
import json
import logging
import os
import re
from dataclasses import dataclass

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
TIMEOUT_S = 120


@dataclass(frozen=True)
class Provider:
    label: str
    kind: str                 # "gemini" | "anthropic" | "openai"
    default_model: str
    key_url: str
    note: str = ""
    base_url: str = ""


PROVIDERS = {
    "google": Provider("Google Gemini", "gemini", os.environ.get("MSS_GEMINI_MODEL", "gemini-3-flash-preview"),
                       "https://aistudio.google.com/apikey", "Free tier available."),
    "anthropic": Provider("Anthropic Claude", "anthropic", "claude-sonnet-5-5",
                          "https://console.anthropic.com/settings/keys", base_url="https://api.anthropic.com/v1"),
    "openai": Provider("OpenAI", "openai", "gpt-5-mini", "https://platform.openai.com/api-keys",
                       base_url="https://api.openai.com/v1"),
    "openrouter": Provider("OpenRouter", "openai", "openrouter/auto", "https://openrouter.ai/keys",
                           "One key for hundreds of models; those ending in ':free' cost nothing.",
                           "https://openrouter.ai/api/v1"),
    "groq": Provider("Groq", "openai", "llama-3.3-70b-versatile", "https://console.groq.com/keys",
                     "Free tier available.", "https://api.groq.com/openai/v1"),
    "mistral": Provider("Mistral", "openai", "mistral-small-latest", "https://console.mistral.ai/api-keys",
                        "Free tier available.", "https://api.mistral.ai/v1"),
    "deepseek": Provider("DeepSeek", "openai", "deepseek-chat", "https://platform.deepseek.com/api_keys",
                         base_url="https://api.deepseek.com/v1"),
}

# Not chat models: hidden from the model lists
_NOT_CHAT = re.compile(r"embed|whisper|tts|audio|realtime|transcri|moderation|image|dall-e|davinci|babbage|"
                       r"search|guard|rerank|ocr|computer-use|veo|imagen|lyria|aqa", re.IGNORECASE)


class AIError(RuntimeError):
    """A message that can be shown to the user as is."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def list_models(provider_id: str, api_key: str) -> list:
    """Chat models this key can use, default first; [default] if the list cannot be fetched."""
    p = PROVIDERS[provider_id]
    try:
        if p.kind == "gemini":
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                             params={"key": api_key, "pageSize": 1000}, timeout=20)
            r.raise_for_status()
            ids = [m["name"].removeprefix("models/") for m in r.json().get("models", [])
                   if "generateContent" in m.get("supportedGenerationMethods", [])]
        elif p.kind == "anthropic":
            r = requests.get(f"{p.base_url}/models", headers=_anthropic_headers(api_key),
                             params={"limit": 1000}, timeout=20)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", [])]
        else:
            r = requests.get(f"{p.base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError, KeyError) as exc:
        LOGGER.info(f"Model list for {provider_id} unavailable: {type(exc).__name__}")
        return [p.default_model]
    ids = sorted({i for i in ids if i and not _NOT_CHAT.search(i)})
    if p.default_model in ids:
        ids.remove(p.default_model)
    return [p.default_model] + ids


# ---------------------------------------------------------------------------
# One completion, any provider
# ---------------------------------------------------------------------------

def _anthropic_headers(api_key: str) -> dict:
    return {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}


def _error_text(r) -> str:
    try:
        body = r.json()
        err = body.get("error", body)
        return str(err.get("message") if isinstance(err, dict) else err)[:300]
    except ValueError:
        return r.text[:300]


def _post(url: str, headers: dict, body: dict) -> dict:
    try:
        r = requests.post(url, headers=headers, json=body, timeout=TIMEOUT_S)
    except requests.RequestException as exc:
        raise AIError(f"Could not reach the AI provider ({type(exc).__name__}).") from None
    if r.status_code in (401, 403):
        raise AIError("The API key was rejected. Check that it is correct and active.")
    if r.status_code == 429:
        raise AIError("The provider's rate limit or quota was reached. Wait a moment or check your plan.")
    if r.status_code >= 400:
        raise _BadRequest(r.status_code, _error_text(r))
    return r.json()


class _BadRequest(AIError):
    def __init__(self, status, detail):
        super().__init__(f"The AI provider returned an error ({status}): {detail}")
        self.detail = detail.lower()


def _openai_compatible(p: Provider, api_key: str, model: str, system: str, messages: list,
                       want_json: bool, temperature: float) -> str:
    body = {"model": model, "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature}
    if want_json:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}"}
    if "openrouter.ai" in p.base_url:
        headers.update({"HTTP-Referer": "https://manuscript-search.org", "X-Title": "Manuscript Search"})
    try:
        data = _post(f"{p.base_url}/chat/completions", headers, body)
    except _BadRequest as exc:
        # Some models accept only the default temperature, or no JSON mode: retry plainly once.
        if not any(w in exc.detail for w in ("temperature", "response_format", "json", "unsupported")):
            raise
        body.pop("temperature", None)
        body.pop("response_format", None)
        data = _post(f"{p.base_url}/chat/completions", headers, body)
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise AIError("The AI provider returned an empty answer.") from None


def _anthropic(p: Provider, api_key: str, model: str, system: str, messages: list, temperature: float) -> str:
    body = {"model": model, "max_tokens": 2048, "system": system, "messages": messages,
            "temperature": temperature}
    try:
        data = _post(f"{p.base_url}/messages", _anthropic_headers(api_key), body)
    except _BadRequest as exc:
        if "temperature" not in exc.detail:
            raise
        body.pop("temperature")
        data = _post(f"{p.base_url}/messages", _anthropic_headers(api_key), body)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def _gemini(api_key: str, model: str, system: str, messages: list, want_json: bool, temperature: float) -> str:
    import google.genai as genai
    from google.genai import types
    try:
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        contents = [types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part.from_text(text=m["content"])]) for m in messages]
        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                             response_mime_type="application/json" if want_json else None)
        return client.models.generate_content(model=model, contents=contents, config=config).text or ""
    except Exception as exc:                      # the SDK raises many types; show its message
        text = str(exc)
        if "API key" in text or "PERMISSION_DENIED" in text or "UNAUTHENTICATED" in text:
            raise AIError("The API key was rejected. Check that it is correct and active.") from None
        if "RESOURCE_EXHAUSTED" in text or "429" in text:
            raise AIError("The provider's rate limit or quota was reached. Wait a moment or check your plan.") from None
        raise AIError(f"The AI provider returned an error: {text[:300]}") from None


def complete(provider_id: str, api_key: str, model: str, system: str, messages: list,
             want_json: bool = False, temperature: float = 0.5) -> str:
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...] ending with a user turn."""
    if not api_key:
        raise AIError("Add your API key first.")
    p = PROVIDERS[provider_id]
    model = (model or p.default_model).strip()
    if p.kind == "gemini":
        return _gemini(api_key, model, system, messages, want_json, temperature)
    if p.kind == "anthropic":
        return _anthropic(p, api_key, model, system, messages, temperature)
    return _openai_compatible(p, api_key, model, system, messages, want_json, temperature)


# ---------------------------------------------------------------------------
# Summary and chat
# ---------------------------------------------------------------------------

def build_context(df_results: pd.DataFrame, top_n: int) -> str:
    """Numbered abstracts, numbered like the result list."""
    return "\n\n".join(f"[{i + 1}] {row.get('title', 'No Title')}\n{row.get('abstract', '')}"
                       for i, (_, row) in enumerate(df_results.head(top_n).iterrows()))


def _parse_json(text: str):
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)        # tolerate code fences and preambles
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def analyze_results(df_results, provider_id: str, api_key: str, model: str = "", top_n: int = 8):
    """One call returning (summary, suggested questions); on failure (message, [])."""
    system = "You are a skilled research assistant who writes precise, well-cited summaries."
    prompt = (
        "For the academic papers below, reply with a JSON object with two keys:\n"
        '- "summary": the key themes, findings and trends in 2-3 concise paragraphs (Markdown); '
        "highlight commonalities, contradictions or gaps, and cite papers by their number [x].\n"
        '- "questions": a list of 3 insightful questions a researcher might ask about this '
        "specific collection of papers.\n\n" + build_context(df_results, top_n)
    )
    try:
        text = complete(provider_id, api_key, model, system, [{"role": "user", "content": prompt}],
                        want_json=True, temperature=0.7)
    except AIError as exc:
        return str(exc), []
    data = _parse_json(text)
    if not isinstance(data, dict):                       # the model ignored the format: show its text
        return text.strip() or "No summary was returned.", []
    summary = str(data.get("summary", "")).strip()
    questions = [str(q).strip() for q in data.get("questions", []) if str(q).strip()]
    return summary or "No summary was returned.", questions[:4]


def chat_with_context(history: list, df_results, provider_id: str, api_key: str, model: str = "",
                      top_n: int = 15) -> str:
    """history: the conversation so far, ending with the user's new question."""
    system = ("You are a helpful research assistant. Answer the user's question based strictly on the "
              "provided academic abstracts. Cite papers by their number [x] when referencing specific "
              "findings. If the answer is not in the context, state that clearly.\n\n"
              "Context — Search Results:\n" + build_context(df_results, top_n))
    messages = [{"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
                for m in history]
    try:
        return complete(provider_id, api_key, model, system, messages, temperature=0.5)
    except AIError as exc:
        return f"I could not answer: {exc}"
