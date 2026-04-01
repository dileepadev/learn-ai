---
title: Introduction to Vector Stores
description: How AI models find information with vector databases.
---

Vector stores (or vector databases) are specialized types of databases that store data as high-dimensional vectors (or embeddings). These are essential for building Retrieval Augmented Generation (RAG) systems.

## What is a Vector Database?

Unlike traditional databases that search for exact text matches, a vector database searches for data by understanding semantic meaning.

### Key Components

1. **Embeddings:** Numerical representations of data (text, images, audio) that capture their meaning.
2. **Indexing:** Efficiently organizing vectors for fast searching and retrieval.
3. **Similarity Search:** Finding the closest vectors to a query, often using methods like Cosine Similarity or Euclidean Distance.

## Why Use Vector Databases?

Most Large Language Models (LLMs) can't search for information across vast amounts of data in real-time. Vector databases act as an "external memory" for LLMs.

### Popular Vector Databases

- **Pinecone:** A managed vector database for fast and scalable AI search.
- **Weaviate:** Open-source vector database with powerful search and classification features.
- **Chroma:** A powerful and lightweight vector database for AI and LLM projects.
- **Milvus:** An open-source vector database for enterprise-grade AI applications.

## Use Cases

- **Retrieval Augmented Generation (RAG):** Enhancing LLM responses with information retrieved from a vector database.
- **Semantic Search:** Building search engines that understand the intent and meaning of the user query.
- **Similarity Search:** Finding similar items, مانند similar images or similar products.
