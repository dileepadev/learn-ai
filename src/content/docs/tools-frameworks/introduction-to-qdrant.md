---
title: Introduction to Qdrant
description: An overview of Qdrant — a high-performance vector database built in Rust — covering its core concepts, filtering, payload storage, and how to use it for similarity search and RAG.
---

Qdrant is an open-source vector database and vector similarity search engine written in Rust. It is designed for production-scale similarity search with rich filtering, payload storage, and a developer-friendly REST and gRPC API. Qdrant is a popular choice for RAG applications, semantic search, and recommendation systems.

## Core Concepts

### Collections
A **collection** is the top-level organizational unit in Qdrant — equivalent to a table in a relational database. Each collection stores vectors of a fixed dimension and supports similarity search over them.

### Points
The fundamental data unit is a **point**, which consists of:
- **ID:** A unique identifier (integer or UUID).
- **Vector:** The embedding — a dense float array of fixed dimensionality.
- **Payload:** A JSON object of arbitrary metadata associated with the vector (e.g., `{"title": "...", "url": "...", "date": "2024-01-15"}`).

### Distance Metrics
Qdrant supports **Cosine**, **Dot Product**, and **Euclidean** distance. The choice depends on how your embeddings were trained — most text embeddings use cosine similarity.

## Getting Started

### Install and Run

```bash
# Docker (quickest)
docker run -p 6333:6333 qdrant/qdrant
```

### Python Client

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

# Create a collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Insert points
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 1536-dim embedding
            payload={"title": "Introduction to AI", "url": "https://..."},
        ),
    ],
)

# Search
results = client.search(
    collection_name="documents",
    query_vector=[0.15, 0.22, ...],  # query embedding
    limit=5,
)
```

## Filtered Vector Search

One of Qdrant's standout features is **filtering at query time** using payload metadata. Filters are applied efficiently using dedicated payload indexes, avoiding full post-processing scans.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="science")),
            FieldCondition(key="year", range=Range(gte=2022)),
        ]
    ),
    limit=10,
)
```

This retrieves the 10 most similar vectors that also match the filter — crucial for multi-tenant applications or domain-restricted search.

## Sparse Vectors and Hybrid Search

Qdrant supports **sparse vectors** alongside dense vectors in the same collection. This enables hybrid search: combining dense semantic similarity (from embeddings) with sparse keyword matching (from BM25 or SPLADE), then merging results with Reciprocal Rank Fusion (RRF).

Hybrid search outperforms either approach alone on most retrieval benchmarks, especially for queries containing specific terminology.

## Named Vectors

A single point can have **multiple named vectors**, each with its own dimension and distance metric. This allows:
- Storing both dense and sparse representations.
- Storing embeddings from different models for the same document.
- Supporting multiple search modalities (text + image) from the same collection.

## Quantization

For large-scale deployments, Qdrant supports vector quantization to reduce memory footprint:
- **Scalar quantization (SQ):** Compress float32 to int8 — 4× memory reduction with minimal accuracy loss.
- **Product quantization (PQ):** Higher compression; more accuracy trade-off.
- **Binary quantization:** Compress to binary; fastest search, highest compression, significant accuracy loss unless re-scoring is used.

## Qdrant vs. Other Vector Databases

| Feature | Qdrant | Pinecone | Weaviate | Chroma |
|---------|--------|----------|----------|--------|
| Open-source | ✓ | ✗ | ✓ | ✓ |
| Rust-based (speed) | ✓ | — | ✗ | ✗ |
| Filtering | Advanced | Basic | Advanced | Basic |
| Sparse vectors | ✓ | ✓ | ✓ (BM25) | ✗ |
| Self-hostable | ✓ | ✗ | ✓ | ✓ |
| Cloud managed | ✓ | ✓ | ✓ | Limited |

## Common Use Cases

- **RAG (Retrieval-Augmented Generation):** Store document embeddings, retrieve relevant chunks at query time to augment LLM context.
- **Semantic search:** Replace keyword search with meaning-based similarity.
- **Recommendation systems:** Find items similar to a user's history.
- **Duplicate detection:** Find near-duplicate documents or images.
- **Anomaly detection:** Flag embeddings that are far from all cluster centroids.

Qdrant's combination of Rust performance, advanced filtering, and a clean API makes it a strong choice for teams building production similarity search systems.
