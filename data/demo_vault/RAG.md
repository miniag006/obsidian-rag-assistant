# Retrieval-Augmented Generation (RAG)

#ai/rag #llm #nlp

## Overview
Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances the output of Large Language Models ([[LLMs]]) by referencing an authoritative external knowledge base before generating a response. 

Instead of relying solely on the static knowledge memorized during model pre-training, RAG dynamically retrieves relevant context from documents (such as Markdown files in an Obsidian vault or PDF manuals) and injects that context into the prompt sent to the LLM.

## Why RAG is Essential
1. **Hallucination Reduction**: Grounding LLM responses in real, verifiable source documents dramatically reduces fabricated facts or hallucinations.
2. **Access to Private & Up-to-Date Data**: Standard LLMs have a fixed knowledge cutoff date and lack access to personal or proprietary notes. RAG bridges this gap without requiring expensive model retraining.
3. **Source Traceability & Verification**: RAG systems can cite specific files, headings, and passages used to construct an answer, allowing users to verify facts.
4. **Cost Efficiency**: Updating a knowledge base with new information only requires computing embeddings for new notes, rather than fine-tuning billions of parameters.

## Core Stages of a RAG Pipeline
A standard RAG architecture consists of two major phases:

### 1. Ingestion Phase
- **Document Loading**: Scanning the knowledge repository (e.g., recursive `.md` loading with `pathlib`) and extracting raw text alongside document metadata.
- **Chunking**: Splitting large documents into smaller semantic units (typically 200–500 tokens) with slight overlap to preserve context across boundaries.
- **Embedding Generation**: Converting each chunk into a high-dimensional dense vector using models like OpenAI `text-embedding-3-small` ([[Embeddings]]).
- **Vector Storage**: Storing chunk vectors and their corresponding text and metadata inside a vector database like [[Vector_Databases]].

### 2. Retrieval & Generation Phase
- **Query Vectorization**: Encoding the user's incoming query into an embedding vector.
- **Similarity Search**: Performing approximate nearest neighbor (ANN) or cosine similarity search against the vector index to retrieve the top-K most relevant chunks.
- **Prompt Augmentation**: Constructing a system and user prompt that includes the retrieved passages as reference context.
- **Grounded Generation**: The LLM synthesizes an accurate answer strictly based on the provided context. If the retrieved context does not contain the answer, the LLM is instructed to explicitly state that the information is missing.

## Common Challenges in RAG
- **Chunk Size Trade-off**: Very small chunks may lack surrounding context; very large chunks may dilute retrieval precision and exceed the model's context window.
- **Retrieval Noise**: Irrelevant chunks retrieved due to keyword overlap can distract the generator.
- **Lost in the Middle**: LLMs often pay closer attention to information placed at the very beginning or end of long context windows rather than the middle.

## Related Concepts
- [[Embeddings]]
- [[Vector_Databases]]
- [[LLMs]]
- [[AI_Agents]]
