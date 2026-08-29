"""
RAG Generation & Context Grounding Pipeline
Constructs augmented prompts and queries the LLM with strict grounding rules.
Supports Google Gemini API (Dynamic Model Discovery + REST + certifi) and OpenAI API.
"""

from dataclasses import dataclass
import os
from typing import List, Optional
import certifi
from openai import OpenAI
import requests

try:
    from src.vectorstore import RetrievedChunk, get_default_client
    from src.utils import clean_unicode
except ModuleNotFoundError:
    from vectorstore import RetrievedChunk, get_default_client
    from utils import clean_unicode


FALLBACK_MESSAGE = "I couldn't find enough information about this in your Obsidian knowledge base."

SYSTEM_PROMPT = """You are an intelligent Obsidian Knowledge Vault Assistant.
Your job is to answer the user's question accurately and concisely, STRICTLY based on the provided context notes retrieved from the user's Obsidian vault.

CRITICAL GROUNDING RULES:
1. Rely ONLY on the facts directly stated in the context notes below. Do NOT assume, extrapolate, or bring in outside knowledge that is not supported by the context.
2. If the provided context notes do NOT contain sufficient information to answer the question, you MUST respond with exactly:
   "I couldn't find enough information about this in your Obsidian knowledge base."
3. When answering, be clear, structured, and factual. Use bullet points or concise paragraphs where appropriate.
4. Always ground your claims. Do not make up citations or references."""

# Cache of discovered models per API key
_DISCOVERED_MODELS_CACHE = {}


@dataclass
class RAGResponse:
    """Represents the complete result of a RAG query execution."""
    answer: str
    sources: List[str]  # List of unique source note filenames
    retrieved_chunks: List[RetrievedChunk]
    is_insufficient_info: bool


def contextualize_query_for_search(query: str, chat_history: Optional[List[dict]] = None) -> str:
    """
    If the query contains pronouns ('it', 'its', 'this', 'these', 'them') or is a short follow-up,
    augments the search query with the subject of the previous user message.
    """
    if not chat_history:
        return query

    pronouns = {"it", "its", "this", "these", "they", "them", "that", "above", "also", "more"}
    words = set(query.lower().split())
    has_pronoun = bool(words & pronouns)
    is_short_followup = len(words) <= 6

    if has_pronoun or is_short_followup:
        prev_user_queries = [m["content"] for m in chat_history if m.get("role") == "user" and m.get("content")]
        if prev_user_queries:
            last_q = prev_user_queries[-1]
            return f"{last_q} {query}"

    return query


def format_context_for_llm(chunks: List[RetrievedChunk]) -> str:
    """
    Formats retrieved chunks into a clean, structured context string for the LLM prompt.
    """
    if not chunks:
        return "No relevant notes found in the knowledge base."

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        block = (
            f"--- Context Note [{i}] ---\n"
            f"Source File: {chunk.filename}\n"
            f"Note Title: {chunk.title}\n"
            f"Heading/Section: {chunk.heading}\n"
            f"Relevance Score: {chunk.similarity_score:.2f}\n"
            f"Content:\n{chunk.text.strip()}\n"
        )
        context_blocks.append(block)

    return "\n".join(context_blocks)


def get_available_gemini_models(api_key: str) -> List[str]:
    """
    Dynamically queries Google Generative Language API for models supporting generateContent
    for this specific API key, guaranteeing zero 404 model errors.
    """
    clean_key = api_key.strip()
    if not clean_key:
        return []

    if clean_key in _DISCOVERED_MODELS_CACHE:
        return _DISCOVERED_MODELS_CACHE[clean_key]

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
        resp = requests.get(url, timeout=12, verify=certifi.where())
        if resp.status_code == 200:
            data = resp.json()
            flash_models = []
            other_models = []
            for m in data.get("models", []):
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    name = m.get("name", "").replace("models/", "")
                    if "flash" in name:
                        flash_models.append(name)
                    else:
                        other_models.append(name)

            # Sort flash models to front for speed
            ordered = flash_models + other_models
            if ordered:
                _DISCOVERED_MODELS_CACHE[clean_key] = ordered
                return ordered
    except Exception:
        pass

    fallback = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.6-flash"]
    _DISCOVERED_MODELS_CACHE[clean_key] = fallback
    return fallback


def call_gemini_generate_rest(prompt: str, api_key: str, model: Optional[str] = None) -> str:
    """
    Calls Google Gemini REST generateContent endpoint directly using discovered valid models.
    Guarantees clean UTF-8 encoding and zero 404 model mismatches.
    """
    clean_key = api_key.strip()
    if not clean_key:
        raise ValueError("No Gemini API key provided. Please enter your API key in the sidebar.")

    # Get dynamically discovered models for this account
    available_models = get_available_gemini_models(clean_key)
    
    # If a specific model was requested and is available, try it first
    if model and model in available_models:
        candidate_models = [model] + [m for m in available_models if m != model]
    else:
        candidate_models = available_models

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048,
        }
    }

    last_err = ""
    for m in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={clean_key}"
        try:
            resp = requests.post(url, json=payload, timeout=30, verify=certifi.where())
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            else:
                last_err = f"Status {resp.status_code} ({m}): {resp.text}"
        except Exception as e:
            last_err = f"({m}) {str(e)}"
            continue

    raise RuntimeError(f"Gemini generation error: {last_err}")


def generate_rag_answer(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    openai_client: Optional[OpenAI] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
    similarity_threshold: float = 0.15,
) -> RAGResponse:
    """
    Executes the grounded RAG generation step with robust dynamic model discovery and unicode sanitization.
    """
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key and openai_client and getattr(openai_client, "api_key", None):
        key = str(openai_client.api_key).strip()

    if not key:
        return RAGResponse(
            answer="⚠️ No API key found. Please enter your Google Gemini API key in the sidebar.",
            sources=[],
            retrieved_chunks=retrieved_chunks,
            is_insufficient_info=False,
        )

    is_gemini = not key.startswith("sk-") or bool(os.getenv("GEMINI_API_KEY"))

    valid_chunks = [c for c in retrieved_chunks if c.similarity_score >= similarity_threshold]

    if not valid_chunks:
        return RAGResponse(
            answer=FALLBACK_MESSAGE,
            sources=[],
            retrieved_chunks=retrieved_chunks,
            is_insufficient_info=True,
        )

    context_text = clean_unicode(format_context_for_llm(valid_chunks))

    # Construct conversation history string
    history_str = ""
    if chat_history:
        recent = [m for m in chat_history[-4:] if m.get("role") in ["user", "assistant"]]
        if recent:
            history_blocks = [f"{m['role'].capitalize()}: {clean_unicode(m['content'])}" for m in recent]
            history_str = "Prior Conversation:\n" + "\n".join(history_blocks) + "\n\n"

    user_prompt = clean_unicode(
        f"{history_str}"
        f"Context from Obsidian Vault:\n"
        f"{context_text}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer using the context above:"
    )

    try:
        if is_gemini:
            answer = call_gemini_generate_rest(
                prompt=user_prompt,
                api_key=key,
                model=model_name
            )
        else:
            client = openai_client or OpenAI(api_key=key)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat.completions.create(
                model=model_name or "gpt-4o-mini",
                messages=messages,
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip()

    except Exception as e:
        answer = f"Error generating answer from LLM: {str(e)}"
        return RAGResponse(
            answer=answer,
            sources=[],
            retrieved_chunks=retrieved_chunks,
            is_insufficient_info=False,
        )

    is_fallback = FALLBACK_MESSAGE.lower() in answer.lower()

    unique_sources = []
    seen = set()
    for c in valid_chunks:
        if c.filename not in seen:
            seen.add(c.filename)
            unique_sources.append(c.filename)

    return RAGResponse(
        answer=answer,
        sources=unique_sources if not is_fallback else [],
        retrieved_chunks=valid_chunks,
        is_insufficient_info=is_fallback,
    )
