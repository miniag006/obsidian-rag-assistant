"""
ChromaDB Vector Store & Embeddings Manager
Handles chunk indexing, embedding generation via Gemini/OpenAI API, and semantic similarity search.
"""

from dataclasses import dataclass
import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI

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


def get_default_client(api_key: Optional[str] = None) -> tuple[OpenAI, str]:
    """
    Initializes an OpenAI-compatible client configured for Google Gemini API (Free)
    or OpenAI depending on the provided key.
    
    Returns:
        tuple[OpenAI, str]: (client_instance, default_embedding_model)
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # Check if this is a Gemini API key or configured for Gemini
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


class VaultVectorStore:
    """
    Manages ChromaDB vector indexing and retrieval for Obsidian vault chunks.
    Uses Google Gemini text-embedding-004 (or OpenAI) for generating vector embeddings.
    """

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        embedding_model: Optional[str] = None,
        collection_name: str = "obsidian_vault",
        persist_directory: Optional[str] = None,
    ):
        if openai_client:
            self.client = openai_client
            self.embedding_model = embedding_model or os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        else:
            self.client, default_model = get_default_client()
            self.embedding_model = embedding_model or default_model

        self.collection_name = collection_name
        
        # Initialize ChromaDB client (persistent if directory provided, otherwise ephemeral in-memory)
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
            
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings in batches.
        """
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            cleaned_batch = [t.replace("\n", " ") for t in batch]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

    def index_chunks(self, chunks: List[NoteChunk]) -> int:
        """
        Indexes a list of NoteChunks into the ChromaDB collection.
        Replaces any existing collection with fresh vectors and metadata.
        """
        if not chunks:
            return 0

        # Reset or recreate collection to avoid mixing vaults
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
            
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.to_metadata_dict() for c in chunks]

        # Generate embeddings
        embeddings = self.get_embeddings_batch(texts)

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

        # Embed query
        clean_query = query.replace("\n", " ")
        query_resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=[clean_query]
        )
        query_embedding = query_resp.data[0].embedding

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
