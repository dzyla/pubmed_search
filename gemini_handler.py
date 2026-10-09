import json
import logging
import os
import pandas as pd
import google.genai as genai
from google.genai import types

LOGGER = logging.getLogger(__name__)

# Override via MSS_GEMINI_MODEL env var if needed.
DEFAULT_MODEL = os.environ.get("MSS_GEMINI_MODEL", "gemini-3-flash-preview")


def get_client(api_key: str):
    """
    Creates a GenAI client for this call only. Users' keys are not cached at
    process level, so a key never outlives the session that supplied it.
    """
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    except Exception as e:
        LOGGER.error(f"Failed to initialize GenAI client: {e}")
        return None


def _build_context(df_results: pd.DataFrame, top_n: int) -> str:
    """Builds a numbered abstract block for use in LLM prompts."""
    parts = []
    for idx, row in df_results.head(top_n).iterrows():
        parts.append(
            f"[{idx + 1}] {row.get('title', 'No Title')}\n{row.get('abstract', '')}"
        )
    return "\n\n".join(parts)


_ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "questions": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["summary", "questions"],
}


def analyze_results(df_results, api_key, top_n=8):
    """
    One Gemini call that returns (summary, suggested_questions) for the top
    results — half the wait of separate summary and question requests.
    On failure returns (error_message, []).
    """
    client = get_client(api_key)
    if not client:
        return "Error: Invalid API Key or Client initialization failed.", []

    prompt = (
        "You are a skilled research assistant. For the academic papers below:\n"
        "1. summary: summarize the key themes, findings and trends in 2-3 concise "
        "paragraphs; highlight commonalities, contradictions or gaps, and cite papers "
        "by their number [x].\n"
        "2. questions: 3 insightful questions a researcher might ask to understand this "
        "specific collection of papers better.\n\n"
        + _build_context(df_results, top_n)
    )

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=types.Part.from_text(text=prompt),
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json",
                response_schema=_ANALYSIS_SCHEMA,
            ),
        )
    except Exception as e:
        LOGGER.error(f"Analysis failed: {e}")
        return f"An error occurred during summarization: {e}", []

    try:
        data = json.loads(response.text)
        summary = str(data.get("summary", "")).strip()
        questions = [str(q).strip() for q in data.get("questions", []) if str(q).strip()]
    except (ValueError, TypeError, AttributeError):
        # Model ignored the schema: show its text as the summary.
        summary, questions = (response.text or "").strip(), []
    return summary or "No summary was returned.", questions[:4]


def chat_with_context(history, user_message, df_results, api_key, top_n=15):
    """
    Chat with the papers using the SDK's structured multi-turn conversation format.

    history already contains the current user_message as its last entry (appended
    by the caller before invoking this function).  We convert the full history to
    typed Content objects so the model receives proper role alternation instead of
    a hand-concatenated string — which prevents prompt-injection via role markers.
    """
    client = get_client(api_key)
    if not client:
        return "Please provide a valid API Key."

    context_text = "Context — Search Results:\n" + _build_context(df_results, top_n)
    system_instruction = (
        "You are a helpful research assistant. Answer the user's question based strictly on "
        "the provided academic abstracts. Cite papers by their number [x] when referencing "
        "specific findings. If the answer is not in the context, state that clearly.\n\n"
        + context_text
    )

    # Convert history list-of-dicts to typed Content objects.
    # The last entry is the current user query.
    contents = [
        types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[types.Part.from_text(text=msg["content"])],
        )
        for msg in history
    ]

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.5,
            ),
        )
        return response.text
    except Exception as e:
        LOGGER.error(f"Chat failed: {e}")
        return f"I encountered an error: {e}"
