"""
ChromaDB Vector Store & Embeddings Manager
Handles chunk indexing, embedding generation via Gemini API / OpenAI API, and semantic similarity search.
"""

from dataclasses import dataclass
import os
from typing import List, Optional, Tuple
import certifi
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import requests

try:
    from src.chunker import NoteChunk
except ModuleNotFoundError:
    from chunker import NoteChunk


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved from the vector store with similarity scoring."""
    chunk_id: str
    filename: str
    relative_path: str
    title: str
    heading: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    distance: float
    similarity_score: float  # Normalized 0.0 - 1.0 score (higher is more relevant)


def get_default_client(api_key: Optional[str] = None) -> Tuple[OpenAI, str]:
    """
    Initializes an OpenAI-compatible client configured for Google Gemini API (Free)
    or OpenAI depending on the provided key.
    
    Returns:
        tuple[OpenAI, str]: (client_instance, default_embedding_model)
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    is_gemini = bool(os.getenv("GEMINI_API_KEY") or (key and not key.startswith("sk-")))

    if is_gemini:
        client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    else:
        client = OpenAI(api_key=key)
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    return client, embedding_model


def get_gemini_embeddings(texts: List[str], api_key: str, model: str = "text-embedding-004", batch_size: int = 32) -> List[List[float]]:
    """
    Fetches embeddings from Google Gemini API using requests with certifi CA bundle.
    Endpoint: https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents
    """
    embeddings: List[List[float]] = []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={api_key}"

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        requests_payload = []
        for text in batch:
            requests_payload.append({
                "model": f"models/{model}",
                "content": {"parts": [{"text": text.replace("\n", " ")}]}
            })

        payload = {"requests": requests_payload}
        try:
            resp = requests.post(url, json=payload, timeout=30, verify=certifi.where())
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API Error ({resp.status_code}): {resp.text}")
            
            data = resp.json()
            for emb_item in data.get("embeddings", []):
                embeddings.append(emb_item.get("values", []))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Gemini Embedding API: {str(e)}")

    return embeddings


def get_openai_embeddings(texts: List[str], client: OpenAI, model: str = "text-embedding-3-small", batch_size: int = 64) -> List[List[float]]:
    """
    Fetches embeddings from OpenAI API using the OpenAI client.
    """
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        cleaned_batch = [t.replace("\n", " ") for t in batch]
        response = client.embeddings.create(
            model=model,
            input=cleaned_batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings


class VaultVectorStore:
    """
    Manages ChromaDB vector indexing and retrieval for Obsidian vault chunks.
    Supports both Google Gemini API (text-embedding-004) and OpenAI API (text-embedding-3-small).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        openai_client: Optional[OpenAI] = None,
        embedding_model: Optional[str] = None,
        collection_name: str = "obsidian_vault",
        persist_directory: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.is_gemini = bool(os.getenv("GEMINI_API_KEY") or (self.api_key and not self.api_key.startswith("sk-")))
        
        if self.is_gemini:
            self.embedding_model = embedding_model or os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
            self.client = openai_client or OpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            self.client = openai_client or OpenAI(api_key=self.api_key)

        self.collection_name = collection_name
        
        # Initialize Ephemeral or Persistent ChromaDB client
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.EphemeralClient()
            
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings using the appropriate provider."""
        if not texts:
            return []
            
        if self.is_gemini:
            return get_gemini_embeddings(texts, api_key=self.api_key, model=self.embedding_model)
        else:
            return get_openai_embeddings(texts, client=self.client, model=self.embedding_model)

    def index_chunks(self, chunks: List[NoteChunk]) -> int:
        """
        Indexes a list of NoteChunks into the ChromaDB collection.
        Safely generates embeddings first, then creates the collection fresh.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.generate_embeddings(texts)

        if not embeddings or len(embeddings) != len(chunks):
            raise RuntimeError(f"Failed to generate embeddings for all {len(chunks)} chunks.")

        ids = [c.chunk_id for c in chunks]
        metadatas = [c.to_metadata_dict() for c in chunks]

        # Reset collection cleanly
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """
        Performs semantic similarity search for a query string against the indexed vault.
        """
        if not query.strip():
            return []

        count = self.collection.count()
        if count == 0:
            return []

        k = min(top_k, count)

        # Embed query vector
        query_embeddings = self.generate_embeddings([query])
        if not query_embeddings or not query_embeddings[0]:
            return []
            
        query_embedding = query_embeddings[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved: List[RetrievedChunk] = []

        if not results or not results["documents"] or not results["documents"][0]:
            return retrieved

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)
        ids = results["ids"][0] if "ids" in results and results["ids"] else [f"c_{i}" for i in range(len(docs))]

        for doc_text, meta, dist, cid in zip(docs, metas, distances, ids):
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))

            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(meta.get("chunk_id", cid)),
                    filename=str(meta.get("filename", "unknown.md")),
                    relative_path=str(meta.get("relative_path", "")),
                    title=str(meta.get("title", "")),
                    heading=str(meta.get("heading", "General")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc_text,
                    start_char=int(meta.get("start_char", 0)),
                    end_char=int(meta.get("end_char", len(doc_text))),
                    distance=float(dist),
                    similarity_score=round(similarity, 4),
                )
            )

        return retrieved

    def get_stats(self) -> dict:
        """Returns statistics on the current index."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
        }
