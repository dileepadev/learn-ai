---
title: Introduction to Vector Search
description: Understanding the fundamentals of vector search and its importance in modern AI.
---

Vector search is a technique used to find similar items based on their content rather than exact matches. It is a core component of many AI applications, including recommendation systems, image search, and Retrieval-Augmented Generation (RAG).

## How Vector Search Works

Instead of matching keywords, vector search transforms data (text, images, audio) into numerical representations called **embeddings**. These embeddings are high-dimensional vectors that capture the semantic meaning of the data.

1. **Embedding**: Data is passed through a model (like an LLM) to generate a vector.
2. **Indexing**: Vectors are stored in a specialized database called a vector database.
3. **Querying**: A search query is also converted into a vector.
4. **Similarity Search**: The system calculates the distance (e.g., Cosine Similarity or Euclidean Distance) between the query vector and the stored vectors to find the nearest neighbors.

## Why Use Vector Search?

- **Semantic Understanding**: Finds results that are contextually related even if they don't share keywords.
- **Multimodal**: Can compare different types of data (e.g., searching images using text).
- **Scalability**: Optimized algorithms like HNSW (Hierarchical Navigable Small World) allow for fast searching across millions of vectors.
