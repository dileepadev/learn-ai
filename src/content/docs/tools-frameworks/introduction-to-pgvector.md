---
title: Introduction to pgvector
description: Learn how pgvector adds vector similarity search to PostgreSQL for semantic retrieval and RAG systems.
---

pgvector is a PostgreSQL extension that enables vector storage and similarity search directly inside Postgres. It is commonly used for semantic search and retrieval-augmented generation (RAG) without introducing a separate vector database.

## Why pgvector

If your app already uses PostgreSQL, pgvector can simplify architecture by keeping transactional data and embeddings in one system.

Benefits include:

- Fewer moving parts
- Familiar SQL tooling
- Easier joins between metadata and vector results
- Simpler operations for small-to-medium workloads

## Core Concepts

### Embeddings

Text, images, or other inputs are transformed into numeric vectors using embedding models.

### Similarity Search

Queries can retrieve nearest vectors using distance operators (cosine, L2, inner product depending on setup).

### Hybrid Retrieval

pgvector can be combined with SQL filters and ranking logic, enabling semantic + structured retrieval in one query flow.

## Typical AI Use Cases

- Semantic document search
- RAG context retrieval
- Related-content recommendations
- Duplicate/similarity detection

## Operational Considerations

- Index choice affects performance and recall
- Embedding dimensions impact storage and speed
- Query patterns should be benchmarked with real workload sizes

For very large-scale, ultra-low-latency workloads, specialized vector stores may still be preferable.

## Best Practices

- Store source metadata with each embedding row
- Version embeddings when models change
- Track retrieval quality, not just latency
- Combine vector similarity with business constraints (tenant, time, permissions)

## Getting Started

1. Enable pgvector in your PostgreSQL environment
2. Create tables with vector columns
3. Generate and insert embeddings
4. Add indexes and benchmark query performance
5. Integrate retrieval with your application or RAG pipeline

pgvector is an excellent choice when you want semantic capabilities with the operational simplicity of staying inside PostgreSQL.
