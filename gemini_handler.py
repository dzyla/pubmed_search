import logging
import google.genai as genai
from google.genai import types

LOGGER = logging.getLogger(__name__)

# Configuration: Updated to the requested Flash Preview model
DEFAULT_MODEL = "gemini-3-flash-preview"

def get_client(api_key):
    """Initializes the GenAI client."""
    if not api_key:
        return None
    try:
        # Using v1alpha as per previous configuration patterns
        return genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    except Exception as e:
        LOGGER.error(f"Failed to initialize GenAI client: {e}")
        return None

def summarize_search_results(df_results, api_key, top_n=8):
    """
    Generates a summary of the top search results.
    """
    client = get_client(api_key)
    if not client:
        return "Error: Invalid API Key or Client initialization failed."

    # Prepare context from top N results
    top_docs = df_results.head(top_n)
    context_text = ""
    for idx, row in top_docs.iterrows():
        context_text += f"Paper {idx+1}: {row.get('title', 'No Title')}\n"
        context_text += f"Abstract: {row.get('abstract', 'No Abstract')}\n\n"

    prompt = (
        "You are a skilled research assistant. Summarize the key themes, findings, and trends "
        "from the following academic papers. Keep it concise (2-3 paragraphs). "
        "Highlight commonalities, contradictions, or gaps if any.\n\n"
        f"{context_text}"
    )

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=types.Part.from_text(text=prompt),
            config=types.GenerateContentConfig(temperature=0.7)
        )
        return response.text
    except Exception as e:
        LOGGER.error(f"Summarization failed: {e}")
        return f"An error occurred during summarization: {e}"

def generate_example_questions(df_results, api_key, top_n=8):
    """
    Generates 3-4 example questions based on the abstracts.
    """
    client = get_client(api_key)
    if not client:
        return []

    top_docs = df_results.head(top_n)
    context_text = ""
    for idx, row in top_docs.iterrows():
        context_text += f"{row.get('title', '')}\n{row.get('abstract', '')}\n"

    prompt = (
        "Based on these scientific abstracts, generate 3 insightful questions "
        "a researcher might ask to understand this specific collection of papers better. "
        "Return only the questions, one per line. Do not number them."
        f"\n\n{context_text}"
    )

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=types.Part.from_text(text=prompt),
            config=types.GenerateContentConfig(temperature=0.7)
        )
        questions = [q.strip() for q in response.text.split('\n') if q.strip()]
        return questions[:4]
    except Exception as e:
        LOGGER.error(f"Question generation failed: {e}")
        return []

def chat_with_context(history, user_message, df_results, api_key, top_n=15):
    """
    Chat with the papers using the context.
    """
    client = get_client(api_key)
    if not client:
        return "Please provide a valid API Key."

    # Prepare context (we include more docs for chat to have broader context)
    top_docs = df_results.head(top_n)
    context_text = "Context - Search Results:\n"
    for idx, row in top_docs.iterrows():
        context_text += f"[{idx+1}] {row.get('title', 'No Title')}\n{row.get('abstract', '')}\n\n"

    system_instruction = (
        "You are a helpful research assistant. Answer the user's question based strictly on the provided academic abstracts. "
        "Cite the papers by their number [x] when referencing specific findings. "
        "If the answer is not in the context, state that clearly."
    )
    
    # Format history for the prompt
    conversation = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n"
    
    final_prompt = f"{system_instruction}\n\n{context_text}\n\nConversation History:\n{conversation}\nUser: {user_message}\nAssistant:"

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=types.Part.from_text(text=final_prompt),
            config=types.GenerateContentConfig(temperature=0.5)
        )
        return response.text
    except Exception as e:
        LOGGER.error(f"Chat failed: {e}")
        return f"I encountered an error: {e}"
    
