---
title: Introduction to Retrieval-Augmented Generation (RAG)
description: Learn the fundamentals of RAG and how it enhances LLM responses with external knowledge.
---

Retrieval-Augmented Generation (RAG) is a technique that combines the generative capabilities of Large Language Models (LLMs) with the precision of information retrieval systems.

## How RAG Works

1. **Indexing:** Documents are broken into chunks and converted into vector embeddings.
2. **Retrieval:** When a user asks a question, the system searches the index for the most relevant chunks.
3. **Generation:** The LLM uses the retrieved chunks as context to provide a grounded and accurate response.

## Why Use RAG?

- **Reduced Hallucinations:** The model relies on provided facts rather than its internal (and potentially outdated) training data.
- **Up-to-Date Information:** You can update the knowledge base without retraining the entire model.
- **Domain Specificity:** Easily adapt a general-purpose LLM to specialized fields like law or medicine.
