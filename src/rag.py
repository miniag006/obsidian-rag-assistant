"""
RAG Generation & Context Grounding Pipeline
Constructs augmented prompts and queries the LLM with strict grounding rules.
Supports Google Gemini API (Free tier) and OpenAI API seamlessly.
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
    similarity_threshold: float = 0.25,
) -> RAGResponse:
    """
    Executes the grounded RAG generation step.
    """
    if openai_client:
        client = openai_client
    else:
        client, _ = get_default_client()

    # Determine default model (Gemini or OpenAI)
    if model_name:
        model = model_name
    elif os.getenv("GEMINI_API_KEY") or str(client.base_url).startswith("https://generativelanguage.googleapis.com"):
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
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
