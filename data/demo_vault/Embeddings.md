# Text Embeddings & Vector Representations

#ai/embeddings #math #nlp

## What are Embeddings?
In natural language processing, an **embedding** is a numerical representation of text (a word, sentence, or document chunk) mapped into a continuous high-dimensional vector space $\mathbb{R}^d$. 

Dense embeddings capture semantic meaning: pieces of text with similar conceptual meanings are positioned close to one another in the vector space, even if they use entirely different vocabulary.

Example:
- The vector for *"How to fix a leaky faucet"* is mathematically close to *"Plumbing repair guide"*, despite sharing almost no identical words.

## How Semantic Similarity is Measured
Vector similarity is commonly calculated using metric functions:

### 1. Cosine Similarity
Measures the cosine of the angle between two vectors $u$ and $v$:
$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
Values range from -1 to 1 (or 0 to 1 for normalized vectors). When vectors are normalized to unit length ($\|u\|=1$), cosine similarity simplifies directly to the dot product $u \cdot v$.

### 2. Euclidean Distance (L2)
Measures the straight-line geometric distance between two points in $d$-dimensional space:
$$d(u, v) = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$
Smaller distances indicate greater similarity.

### 3. Dot Product (Inner Product)
Reflects both the angle and the magnitude of the vectors:
$$\text{Dot Product}(u, v) = \sum_{i=1}^d u_i v_i$$

## Modern Embedding Models
- **OpenAI text-embedding-3-small**: Generates 1536-dimensional vectors with high semantic accuracy and low latency.
- **OpenAI text-embedding-3-large**: Offers 3072 dimensions for higher precision tasks.
- **Open-source alternatives**: BAAI/bge-large-en, sentence-transformers/all-MiniLM-L6-v2.

## Role in RAG Systems
In a [[RAG]] pipeline, embeddings serve as the indexing mechanism:
1. Every chunk of markdown notes is converted into an embedding during indexing.
2. The user's query is converted into an embedding at runtime.
3. The system performs vector search in a [[Vector_Databases]] to identify the nearest chunk vectors.

## Related Concepts
- [[Vector_Databases]]
- [[RAG]]
- [[LLMs]]
