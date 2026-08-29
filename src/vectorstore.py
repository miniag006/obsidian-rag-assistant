"""
ChromaDB Vector Store & OpenAI Embeddings Manager
Handles chunk indexing, embedding generation via OpenAI API, and semantic similarity search.
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


class VaultVectorStore:
    """
    Manages ChromaDB vector indexing and retrieval for Obsidian vault chunks.
    Uses OpenAI's text-embedding-3-small model for generating vector embeddings.
    """

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "obsidian_vault",
        persist_directory: Optional[str] = None,
    ):
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        Generates OpenAI embeddings for a list of text strings in batches.
        """
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Replace newlines with spaces for optimal embedding model performance
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
        
        Returns:
            int: Number of chunks indexed.
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
        
        Args:
            query: The user query or question.
            top_k: Number of most relevant chunks to retrieve.
            
        Returns:
            List[RetrievedChunk]: Ranked retrieved chunks with metadata and similarity scores.
        """
        if not query.strip():
            return []

        count = self.collection.count()
        if count == 0:
            return []

        # Adjust top_k if total indexed chunks is smaller
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
            # For cosine distance in ChromaDB: distance = 1 - cosine_similarity
            # Therefore similarity = 1.0 - distance (or max(0.0, 1.0 - dist))
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
