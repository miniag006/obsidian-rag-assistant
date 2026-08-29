# Large Language Models (LLMs)

#ai/llm #machine_learning #deep_learning

## Overview
Large Language Models (LLMs) are deep neural networks based on the Transformer architecture, trained on vast corpora of text data using self-supervised learning objectives (primarily next-token prediction). Examples include OpenAI's GPT-4o, Anthropic's Claude 3.5, and open-source models like Llama 3.

## Training Pipeline
The development of modern LLMs typically involves three stages:

### 1. Pre-training
- Models are trained on trillions of tokens from web text, books, code, and scientific literature.
- The core objective is causal language modeling (predicting the token $x_t$ given $x_1, \dots, x_{t-1}$).
- Pre-training builds deep linguistic comprehension, world knowledge, and reasoning capabilities, but raw base models simply continue text rather than follow conversational instructions.

### 2. Supervised Fine-Tuning (SFT)
- The base model is trained on curated datasets of instruction-response pairs.
- This aligns the model into a conversational assistant capable of answering questions, summarizing text, and writing code.

### 3. Reinforcement Learning from Human Feedback (RLHF)
- Models are refined using reward models that score outputs based on helpfulness, accuracy, and safety.
- Techniques like Direct Preference Optimization (DPO) and PPO align model behavior with human intent.

## Context Window and Tokenization
- **Tokens**: The basic units of text processed by LLMs (approximately 4 characters or 0.75 words in English).
- **Context Window**: The maximum number of tokens an LLM can process in a single inference call (e.g., 128k tokens for GPT-4o).
- While context windows have expanded, retrieval pipelines like [[RAG]] remain critical because processing massive contexts is expensive, slower, and prone to attention degradation.

## Hallucinations and Limitations
- **Hallucination**: The generation of factually incorrect or ungrounded assertions presented with high confidence.
- **Knowledge Cutoff**: The model has no knowledge of real-world events or private documents created after its training data collection date.
- **Lack of Verification**: Raw LLMs do not inherently know where their information originated, making citation and auditing difficult without external retrieval mechanisms.

## Related Concepts
- [[RAG]]
- [[Embeddings]]
- [[AI_Agents]]
