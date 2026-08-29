# Vector Databases & ChromaDB

#ai/vectordb #databases #rag

## Overview
A **Vector Database** is a specialized storage engine optimized for indexing, storing, and querying high-dimensional vector embeddings alongside structured metadata. 

Unlike traditional relational databases that query by exact keyword matches or foreign key lookups, vector databases perform **Approximate Nearest Neighbor (ANN)** search to rapidly locate items based on semantic similarity.

## ChromaDB: Lightweight & Embedded Vector Store
**ChromaDB** is an open-source vector store designed for developer ergonomics, rapid prototyping, and production AI applications.

### Key Characteristics of ChromaDB:
1. **Embedded / In-Memory Mode**: Can run directly inside the Python application process without requiring external server processes or Docker containers.
2. **Persistent Storage**: Can persist embeddings, document chunks, and metadata to a local folder on disk (`PersistentClient`).
3. **Built-in Distance Metrics**: Supports cosine similarity, L2 Euclidean distance, and inner product (`ip`).
4. **Metadata Filtering**: Enables filtering search results by custom attributes (e.g., `filename`, `heading`, or `timestamp`).

## Indexing Algorithms in Vector Databases
To search millions of vectors in sub-millisecond time, vector databases rely on approximate indexing algorithms:

- **HNSW (Hierarchical Navigable Small World)**: A multi-layered graph structure where upper layers allow fast traversal across large vector distances, and lower layers refine local searches. ChromaDB uses HNSW for its default index.
- **IVF (Inverted File Index)**: Partitions the vector space into Voronoi cells and searches only within the clusters closest to the query vector.
- **Flat Index (Exact Search)**: Performs exhaustive brute-force distance calculation across all vectors. Accurate but scales linearly $O(N)$ with dataset size.

## Practical Usage in Obsidian RAG
In our Obsidian RAG Assistant:
1. A ChromaDB collection `obsidian_vault` is created.
2. Markdown chunks are stored with their embedding vectors and metadata (`filename`, `file_path`, `heading`, `chunk_index`, `start_char`, `end_char`).
3. When a user asks a question, ChromaDB queries the collection and returns the top-K matching documents and their similarity scores.

## Related Concepts
- [[Embeddings]]
- [[RAG]]
- [[LLMs]]
