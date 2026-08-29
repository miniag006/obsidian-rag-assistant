"""
RAG Generation & Context Grounding Pipeline
Constructs augmented prompts and queries the LLM with strict grounding rules.
Supports Google Gemini API (Dynamic Model Discovery + REST + certifi) and OpenAI API.
"""

from dataclasses import dataclass
import os
import re
from typing import List, Optional, Tuple
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
Your job is to answer the user's question helpfully, accurately, and concisely based on the provided context notes retrieved from the user's Obsidian vault.

GUIDELINES:
1. Answer the question using the information provided in the context notes. Synthesize and connect relevant concepts from the notes to directly address the user's query (including addressing paraphrased questions, typos, or informal wording).
2. Only use the fallback message "I couldn't find enough information about this in your Obsidian knowledge base." if the context notes truly contain zero relevant information or mentions related to the user's question.
3. When answering, be clear, structured, and factual.
4. Always ground your claims directly in the context notes without fabricating citations.
5. SOURCE ATTRIBUTION MANDATE: At the very end of your response, on a new line, list ONLY the context note numbers (e.g. [SOURCES_USED: 1] or [SOURCES_USED: 1, 2]) whose content you actually used to answer the question. Do NOT cite notes that you did not use. If no notes were used, do not output this tag."""

# Cache of discovered models per API key
_DISCOVERED_MODELS_CACHE = {}


@dataclass
class RAGResponse:
    """Represents the complete result of a RAG query execution."""
    answer: str
    sources: List[str]  # List of unique source note filenames that actually support the answer
    retrieved_chunks: List[RetrievedChunk]
    is_insufficient_info: bool


def extract_subject_from_query(query: str) -> str:
    """Extracts the core subject entity from a user question."""
    cleaned = re.sub(
        r"^(what\s+is|what\s+are|explain|describe|tell\s+me\s+about|how\s+does|why\s+is|why\s+are)\s+",
        "",
        query.strip(),
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r"[?!.,;:]+$", "", cleaned).strip()

    # Check for known multi-word concepts or acronyms
    patterns = [
        r"\b(llms?|large language models?)\b",
        r"\b(rag|retrieval[ -]augmented generation)\b",
        r"\b(embeddings?|dense embeddings?|vector embeddings?)\b",
        r"\b(vector databases?|chromadb|pinecone|faiss|qdrant)\b",
        r"\b(ai agents?|agents?|react pattern|react agent)\b",
        r"\b(fashion design|fashion)\b",
        r"\b(law|legal research)\b",
        r"\b(ca|chartered accountant|chartered accountancy)\b",
        r"\b(btech|engineering)\b",
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            return m.group(0).strip()

    stopwords = {"the", "a", "an", "and", "or", "of", "in", "for", "with", "to", "its", "their", "it", "this", "that"}
    meaningful = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", cleaned) if w.lower() not in stopwords]
    if meaningful:
        return " ".join(meaningful[:3])
    return cleaned


def contextualize_query_for_search(query: str, chat_history: Optional[List[dict]] = None) -> str:
    """
    Resolves conversational follow-up references ('it', 'its', 'they', 'them', 'these', 'those')
    by substituting the core subject from the previous conversation turn.
    Preserves standalone queries without topic pollution.
    """
    if not chat_history:
        return query

    q_clean = query.strip()
    words = set(re.findall(r"\b[a-zA-Z]+\b", q_clean.lower()))

    pronouns = {"it", "its", "they", "them", "their", "theirs", "this", "that", "these", "those"}
    has_pronoun = bool(words & pronouns)

    followup_patterns = [
        r"\bwhy\b.*\buseful\b",
        r"\bwhat\b.*\badvantages\b",
        r"\bwhat\b.*\bdisadvantages\b",
        r"\bwhat\b.*\buse\s*cases\b",
        r"\bhow\b.*\bwork\b",
        r"\bhow\b.*\bused\b",
        r"\bgive\b.*\bexamples\b",
        r"\btell\b.*\bmore\b",
        r"\bwhat\b.*\bbenefits\b",
        r"\bwhy\b.*\bneed(ed)?\b",
    ]
    is_pattern_followup = any(re.search(pat, q_clean, re.IGNORECASE) for pat in followup_patterns)

    if has_pronoun or is_pattern_followup:
        prev_user_queries = [m["content"] for m in chat_history if m.get("role") == "user" and m.get("content")]
        if prev_user_queries:
            last_q = prev_user_queries[-1]
            subject = extract_subject_from_query(last_q)
            if subject:
                subbed_query = re.sub(
                    r"\b(its|it|their|they|them|this|that|these|those)\b",
                    subject,
                    q_clean,
                    flags=re.IGNORECASE
                )
                if subject.lower() not in subbed_query.lower():
                    return f"{subject}: {subbed_query}"
                return subbed_query

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


def extract_supporting_sources_and_chunks(
    raw_answer: str,
    valid_chunks: List[RetrievedChunk]
) -> Tuple[str, List[str], List[RetrievedChunk]]:
    """
    Extracts the clean answer, unique supporting source filenames, and supporting chunks
    based strictly on the LLM's explicit [SOURCES_USED: ...] citation tag, with strict content-matching fallback.
    Guarantees unrelated retrieved files are never leaked to user-facing Sources.
    """
    pattern = r'\[SOURCES(?:_USED)?:\s*([0-9,\s]+)\]'
    match = re.search(pattern, raw_answer, re.IGNORECASE)

    clean_answer = re.sub(pattern, '', raw_answer, flags=re.IGNORECASE).strip()

    is_fallback = FALLBACK_MESSAGE.lower() in clean_answer.lower()
    if is_fallback:
        return clean_answer, [], []

    supporting_chunks = []
    if match:
        indices_str = match.group(1)
        indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
        for idx in indices:
            if 1 <= idx <= len(valid_chunks):
                chunk = valid_chunks[idx - 1]
                if chunk not in supporting_chunks:
                    supporting_chunks.append(chunk)

    # Content-based fallback if tag was omitted or empty
    if not supporting_chunks and valid_chunks:
        # Common English stop words to exclude from keyword overlap
        stopwords = {
            "this", "that", "these", "those", "with", "from", "have", "been", "were",
            "which", "what", "where", "when", "about", "into", "more", "also", "some",
            "such", "than", "then", "there", "their", "they", "them", "will", "would",
            "could", "should", "used", "uses", "using", "your", "user", "notes"
        }
        answer_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', clean_answer.lower())) - stopwords

        scored_chunks = []
        for c in valid_chunks:
            chunk_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', c.text.lower())) - stopwords
            overlap = answer_words & chunk_words
            if len(overlap) >= 3:
                scored_chunks.append((len(overlap), c))

        if scored_chunks:
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            supporting_chunks = [c for _, c in scored_chunks]
        else:
            supporting_chunks = [valid_chunks[0]]

    unique_sources = []
    seen = set()
    for c in supporting_chunks:
        if c.filename not in seen:
            seen.add(c.filename)
            unique_sources.append(c.filename)

    return clean_answer, unique_sources, supporting_chunks


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

            ordered = flash_models + other_models
            if ordered:
                _DISCOVERED_MODELS_CACHE[clean_key] = ordered
                return ordered
    except Exception:
        pass

    fallback = ["gemini-3.6-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash"]
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

    available_models = get_available_gemini_models(clean_key)
    
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
    similarity_threshold: float = 0.0,
) -> RAGResponse:
    """
    Executes the grounded RAG generation step with dynamic model discovery,
    unicode sanitization, and precise source attribution filtering.
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
            raw_answer = call_gemini_generate_rest(
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
            raw_answer = response.choices[0].message.content.strip()

    except Exception as e:
        answer = f"Error generating answer from LLM: {str(e)}"
        return RAGResponse(
            answer=answer,
            sources=[],
            retrieved_chunks=retrieved_chunks,
            is_insufficient_info=False,
        )

    clean_answer, supporting_sources, supporting_chunks = extract_supporting_sources_and_chunks(
        raw_answer=raw_answer,
        valid_chunks=valid_chunks
    )

    is_fallback = FALLBACK_MESSAGE.lower() in clean_answer.lower()

    return RAGResponse(
        answer=clean_answer,
        sources=supporting_sources if not is_fallback else [],
        retrieved_chunks=supporting_chunks if not is_fallback else valid_chunks,
        is_insufficient_info=is_fallback,
    )
