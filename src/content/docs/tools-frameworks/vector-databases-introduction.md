---
title: Introduction to Vector Databases
description: Learn how vector databases store and search embeddings for modern AI applications.
---

Vector databases are specialized storage systems designed to store and efficiently search high-dimensional vector embeddings, which are numerical representations of data useful in AI/ML tasks.

## Why Vector Databases?

Traditional relational databases are great for structured data but fall short when it comes to similarity searches across text, images, or audio.

## Core Features

- **Indexing:** Algorithms like HNSW (Hierarchical Navigable Small Worlds) for fast nearest-neighbor searches.
- **Filtering:** Meta-data filtering for combining keyword searches with semantic similarity.
- **Scaling:** Distributed architectures for handling millions of embeddings.

## Popular Vector Databases

1. **Qdrant:** Highly performant with advanced filtering capabilities.
2. **Pinecone:** Serverless, cloud-native vector search service.
3. **Milvus:** Open-source, highly scalable vector database.
4. **Chroma:** Lightweight and easy to set up for local development.

## Use Cases

- **Retrieval-Augmented Generation (RAG):** Providing context to LLMs.
- **Recommendation Systems:** Finding similar products or users.
- **Semantic Search:** Understanding the intent behind a user's query.
