---
title: Introduction to Pinecone - Managed Vector Database for Production RAG
description: Learn how Pinecone's fully managed vector database handles indexing, filtering, and scaling for production retrieval-augmented generation systems.
---

Pinecone is a fully managed vector database purpose-built for storing embeddings and running fast approximate nearest-neighbor search at production scale, without requiring you to operate the underlying infrastructure.

## Core Operations

Pinecone organizes vectors into indexes, each holding vectors of a fixed dimensionality with optional metadata attached to each vector for filtering:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("documents")

index.upsert(vectors=[
    {"id": "doc1", "values": embedding, "metadata": {"source": "handbook", "year": 2024}}
])

results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"year": {"$gte": 2023}}
)
```

Metadata filtering lets a query combine semantic similarity search with structured constraints — retrieve the five most relevant documents, but only those from a specific source or date range — in a single request rather than post-filtering results client-side.

## Serverless vs. Pod-Based Indexes

Pinecone's serverless indexes scale storage and compute automatically and charge based on actual usage, removing capacity planning for variable or unpredictable workloads. Pod-based indexes give more predictable performance and cost at high, steady query volumes by reserving dedicated compute, at the cost of needing to size and manage that capacity manually.

## Namespaces for Multi-Tenancy

Pinecone supports namespaces within a single index, letting you logically partition vectors — for example, one namespace per customer in a multi-tenant RAG application — while querying only within a specific namespace, which is both a performance optimization and a data isolation mechanism.

## Practical Guidance

Pinecone removes the operational burden of running and scaling a vector index yourself, which matters most once query volume or data size outgrows what a self-hosted option like Chroma or a Postgres extension like pgvector can comfortably handle on your own infrastructure. For smaller-scale or cost-sensitive projects, evaluate self-hosted alternatives first, since managed vector database pricing scales directly with vector count and query volume.
