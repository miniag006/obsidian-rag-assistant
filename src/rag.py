"""
RAG Generation & Context Grounding Pipeline
Constructs augmented prompts and queries the LLM with strict grounding rules.
Supports Google Gemini API (Free tier) and OpenAI API with multi-model fallback.
"""

from dataclasses import dataclass
import os
from typing import List, Optional
from openai import OpenAI

try:
    from src.vectorstore import RetrievedChunk, get_default_client
except ModuleNotFoundError:
    from vectorstore import RetrievedChunk, get_default_client


FALLBACK_MESSAGE = "I couldn't find enough information about this in your Obsidian knowledge base."

SYSTEM_PROMPT = """You are an intelligent Obsidian Knowledge Vault Assistant.
Your job is to answer the user's question accurately and concisely, STRICTLY based on the provided context notes retrieved from the user's Obsidian vault.

CRITICAL GROUNDING RULES:
1. Rely ONLY on the facts directly stated in the context notes below. Do NOT assume, extrapolate, or bring in outside knowledge that is not supported by the context.
2. If the provided context notes do NOT contain sufficient information to answer the question, you MUST respond with exactly:
   "I couldn't find enough information about this in your Obsidian knowledge base."
3. When answering, be clear, structured, and factual. Use bullet points or concise paragraphs where appropriate.
4. Always ground your claims. Do not make up citations or references."""


@dataclass
class RAGResponse:
    """Represents the complete result of a RAG query execution."""
    answer: str
    sources: List[str]  # List of unique source note filenames
    retrieved_chunks: List[RetrievedChunk]
    is_insufficient_info: bool


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


def generate_rag_answer(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    openai_client: Optional[OpenAI] = None,
    model_name: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
    similarity_threshold: float = 0.20,
) -> RAGResponse:
    """
    Executes the grounded RAG generation step with robust multi-model fallback.
    """
    if openai_client:
        client = openai_client
    else:
        client, _ = get_default_client()

    is_gemini = os.getenv("GEMINI_API_KEY") or str(client.base_url).startswith("https://generativelanguage.googleapis.com")

    # Build model candidate priority list
    if is_gemini:
        preferred = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        candidates = [preferred, "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-3.6-flash"]
    else:
        preferred = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        candidates = [preferred, "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

    # Deduplicate while preserving order
    seen_models = set()
    model_candidates = []
    for m in candidates:
        if m and m not in seen_models:
            seen_models.add(m)
            model_candidates.append(m)

    valid_chunks = [c for c in retrieved_chunks if c.similarity_score >= similarity_threshold]

    if not valid_chunks:
        return RAGResponse(
            answer=FALLBACK_MESSAGE,
            sources=[],
            retrieved_chunks=retrieved_chunks,
            is_insufficient_info=True,
        )

    context_text = format_context_for_llm(valid_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add short chat history if present (last 4 interactions)
    if chat_history:
        for msg in chat_history[-4:]:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})

    # User message with augmented context
    user_prompt = (
        f"Context from Obsidian Vault:\n"
        f"{context_text}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer using the context above:"
    )
    messages.append({"role": "user", "content": user_prompt})

    answer = ""
    last_error = None

    # Try model candidates in sequence
    for m in model_candidates:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip()
            if answer:
                break
        except Exception as e:
            last_error = e
            continue

    if not answer:
        err_text = f"Error generating answer from LLM: {str(last_error)}" if last_error else "Failed to generate answer."
        return RAGResponse(
            answer=err_text,
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
