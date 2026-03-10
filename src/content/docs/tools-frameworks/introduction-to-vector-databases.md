---
title: Introduction to Vector Databases
description: Understanding how vector databases enable efficient similarity search for AI applications.
---

Vector databases are specialized databases designed to store and query high-dimensional vector embeddings efficiently.

## Why Vector Databases?

Traditional relational databases are great for exact matches, but they struggle with "semantic similarity." Vector databases allow you to find items that are "close" in meaning, even if they don't share identical keywords.

## Key Concepts

- **Embeddings:** Numerical representations of data (text, images, audio) in a high-dimensional space.
- **Distance Metrics:** Methods like Cosine Similarity or Euclidean Distance used to measure how close two vectors are.
- **Indexing:** Specialized structures like HNSW or IVF that make searching millions of vectors extremely fast.
