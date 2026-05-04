---
title: "Introduction to FAISS"
description: "A comprehensive guide to FAISS (Facebook AI Similarity Search), a high-performance open-source library for efficient similarity search and clustering of dense vectors at massive scale, widely used in AI and information retrieval systems."
---

## What Is FAISS?

**FAISS** (Facebook AI Similarity Search) is an open-source library developed by Meta AI Research, written in C++ with Python bindings, for efficient **similarity search** and **clustering** of dense vectors. Given a query vector, FAISS finds the $k$ most similar vectors in a large database (nearest neighbor search), or clusters a large vector dataset into groups.

FAISS is the backbone of many production-scale vector search systems, including retrieval-augmented generation (RAG) pipelines, recommendation systems, image search, and document retrieval. It is designed to work efficiently on datasets ranging from thousands to billions of vectors, exploiting GPU acceleration for maximum throughput.

---

## Why FAISS? The Nearest Neighbor Search Problem

The fundamental problem FAISS solves is: given a set of $N$ vectors $\{x_1, \ldots, x_N\} \subset \mathbb{R}^d$ and a query $q \in \mathbb{R}^d$, find the $k$ vectors closest to $q$ under some distance metric.

**Brute-force search** computes all $N$ distances and takes $O(Nd)$ time per query. For $N = 10^9$ and $d = 768$, this is computationally prohibitive for interactive applications. FAISS provides approximate nearest neighbor (ANN) algorithms that trade a small amount of accuracy for orders-of-magnitude speedup.

### Distance Metrics Supported

- **L2 (Euclidean)**: $d(x,y) = \|x - y\|_2$ — default for many embedding models.
- **Inner Product**: $d(x,y) = -x \cdot y$ — used for cosine similarity with normalized vectors.
- **L1 (Manhattan)**: $d(x,y) = \|x - y\|_1$

For cosine similarity, normalize vectors to unit length before indexing, then use inner product search.

---

## Installation

```bash
# CPU-only (via conda — recommended for FAISS)
conda install -c pytorch faiss-cpu

# GPU version (CUDA required)
conda install -c pytorch faiss-gpu

# Via pip (may lack some optimizations)
pip install faiss-cpu
# or
pip install faiss-gpu
```

---

## Core Index Types

### Flat Indexes (Exact Search)

`IndexFlatL2` and `IndexFlatIP` perform exact brute-force search. No approximation — always finds true nearest neighbors.

```python
import faiss
import numpy as np

d = 128          # dimension
N = 100_000      # number of vectors

# Create random vectors
xb = np.random.randn(N, d).astype(np.float32)
xq = np.random.randn(1000, d).astype(np.float32)  # query vectors

# Build flat index
index = faiss.IndexFlatL2(d)
index.add(xb)

print(f"Vectors in index: {index.ntotal}")  # 100000

# Search: 5 nearest neighbors for each query
k = 5
distances, indices = index.search(xq, k)

print(distances.shape)  # (1000, 5)
print(indices.shape)    # (1000, 5)
print(indices[0])       # Indices of 5 nearest neighbors for first query
```

**Use case**: Small to medium datasets (< 1M vectors) where accuracy is critical.

### IVF Indexes (Inverted File Index)

`IndexIVFFlat` divides the vector space into Voronoi cells using k-means clustering, then searches only the `nprobe` nearest cells rather than the full dataset.

```python
import faiss
import numpy as np

d = 128
N = 1_000_000
nlist = 1000  # number of Voronoi cells

# Training data (representative sample of the dataset)
xb = np.random.randn(N, d).astype(np.float32)

quantizer = faiss.IndexFlatL2(d)  # the coarse quantizer
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# IVF indexes must be trained first
assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)
index.nprobe = 10  # search 10 nearest cells (higher = more accurate, slower)

xq = np.random.randn(1000, d).astype(np.float32)
distances, indices = index.search(xq, k=5)
```

**Trade-off**: `nprobe` controls the accuracy-speed trade-off. Higher `nprobe` → higher recall, higher latency. Typical recall@1 > 95% with `nprobe = sqrt(nlist)`.

**Use case**: Datasets of 1M–100M vectors where near-real-time search is required.

### IVF-PQ (Product Quantization)

`IndexIVFPQ` combines inverted file indexing with **product quantization** (PQ) for memory-efficient storage. PQ compresses each vector into a short code by partitioning the vector space into subspaces and quantizing each subspace independently.

For a $d$-dimensional vector with $m$ subquantizers each with $k^*$ centroids:
- **Storage**: $m \cdot \log_2(k^*) / 8$ bytes per vector (vs. $4d$ bytes for float32).
- **Compression ratio**: Typically 8–64×.

```python
import faiss
import numpy as np

d = 128
N = 10_000_000
nlist = 4096
M = 8        # number of subquantizers (d must be divisible by M)
nbits = 8    # bits per subquantizer code (k* = 2^nbits = 256 centroids)

xb = np.random.randn(N, d).astype(np.float32)

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

index.train(xb)
index.add(xb)
index.nprobe = 64

xq = np.random.randn(100, d).astype(np.float32)
distances, indices = index.search(xq, k=10)
```

**Use case**: 10M–1B vectors where memory is a constraint. Typical memory: ~32 bytes/vector (vs. 512 bytes for d=128 float32).

### HNSW (Hierarchical Navigable Small World)

`IndexHNSW` is a graph-based index where each vector is connected to its approximate nearest neighbors in a hierarchical graph. Search traverses the graph greedily from coarse to fine levels.

```python
import faiss
import numpy as np

d = 128
N = 1_000_000
M_hnsw = 32   # number of connections per node (higher = better recall, more memory)
efConstruction = 200  # construction-time ef parameter

index = faiss.IndexHNSWFlat(d, M_hnsw)
index.hnsw.efConstruction = efConstruction

xb = np.random.randn(N, d).astype(np.float32)
index.add(xb)  # No training step needed for HNSW

index.hnsw.efSearch = 64  # search-time parameter (higher = better recall, slower)
xq = np.random.randn(100, d).astype(np.float32)
distances, indices = index.search(xq, k=10)
```

HNSW typically achieves the best recall-per-query-time trade-off at 1–10M scale and requires no training step (unlike IVF methods).

---

## Index Selection Guide

| Index | Size | Recall | Memory | Training | GPU Support |
|-------|------|--------|--------|----------|-------------|
| `IndexFlatL2` | <1M | 100% | High | None | Yes |
| `IndexIVFFlat` | 1M–100M | 95–99% | High | Yes | Yes |
| `IndexIVFPQ` | 100M–1B | 85–97% | Low | Yes | Yes |
| `IndexHNSW` | 1M–10M | 96–99% | Medium | None | No |
| `IndexIVFPQR` | 1B+ | 90–99% | Low | Yes | Yes |

---

## Saving and Loading Indexes

```python
import faiss

# Save to disk
faiss.write_index(index, "my_index.faiss")

# Load from disk
loaded_index = faiss.read_index("my_index.faiss")
```

For large indexes (tens of GB), FAISS supports memory-mapped files for on-demand loading without reading the entire index into RAM.

---

## GPU Acceleration

FAISS supports GPU search for flat and IVF indexes, providing 5–100× speedup over CPU:

```python
import faiss

# Single GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # move to GPU 0

# Multi-GPU
gpu_index = faiss.index_cpu_to_all_gpus(index)  # distribute across all GPUs

distances, indices = gpu_index.search(xq, k=10)
```

GPU-accelerated flat search achieves >1 billion distance computations per second on modern hardware.

---

## Integration with Embedding Models

FAISS is commonly used with sentence embedding models for semantic search:

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
d = 384  # MiniLM embedding dimension

# Build document index
documents = [
    "FAISS is a library for similarity search.",
    "Machine learning is transforming science.",
    "The ocean covers 70% of Earth's surface.",
    "Neural networks learn hierarchical representations.",
]

doc_embeddings = model.encode(documents, convert_to_numpy=True)
doc_embeddings = doc_embeddings.astype(np.float32)

# Normalize for cosine similarity
faiss.normalize_L2(doc_embeddings)

index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity on normalized vectors
index.add(doc_embeddings)

# Query
query = "How does vector search work?"
q_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
faiss.normalize_L2(q_embedding)

distances, indices = index.search(q_embedding, k=3)
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. [{distances[0][i]:.4f}] {documents[idx]}")
```

---

## Use in RAG Pipelines

FAISS is a popular choice for the vector store in RAG (Retrieval-Augmented Generation) pipelines:

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build vector store from documents
vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)

# Retrieve top-3 relevant documents for a query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("How does similarity search work?")
```

LangChain and LlamaIndex both provide native FAISS integrations, handling embedding, indexing, metadata storage, and retrieval in a unified API.

---

## ID Mapping for Custom Identifiers

By default, FAISS assigns sequential integer IDs. For custom IDs, wrap with `IndexIDMap`:

```python
import faiss
import numpy as np

d = 128
base_index = faiss.IndexFlatL2(d)
index = faiss.IndexIDMap(base_index)

vectors = np.random.randn(10, d).astype(np.float32)
ids = np.array([1001, 1002, 1003, 1004, 1005, 2001, 2002, 2003, 2004, 2005])

index.add_with_ids(vectors, ids)

distances, indices = index.search(np.random.randn(1, d).astype(np.float32), k=3)
print(indices)  # Returns your custom IDs, e.g., [[2003, 1002, 1005]]
```

---

## Comparison: FAISS vs. Other Vector Databases

| Feature | FAISS | Pinecone | Weaviate | Qdrant | pgvector |
|---------|-------|----------|----------|--------|----------|
| Type | Library | Cloud service | Database | Database | Extension |
| Self-hosted | Yes | No | Yes | Yes | Yes |
| Persistence | Manual | Auto | Auto | Auto | Auto |
| Metadata filtering | No (manual) | Yes | Yes | Yes | Yes |
| GPU support | Yes | No | No | No | No |
| Scale | Billions | Millions | Millions | Millions | Millions |
| Best for | High perf research/prod | Managed cloud | Complex schemas | Filtering+search | Postgres users |

FAISS excels at raw performance and flexibility, especially for research prototypes, GPU-accelerated pipelines, and scenarios requiring fine-grained control over indexing strategy.

---

## Summary

FAISS is the gold standard library for high-performance similarity search at scale. From exact brute-force search on small datasets to billion-scale approximate nearest neighbor search with product quantization, FAISS offers a comprehensive set of indexes covering the full trade-off landscape of accuracy, memory, and speed. Its GPU support, Python bindings, and integration with the broader ML ecosystem make it the default choice for implementing vector search in research and production AI systems. Understanding FAISS indexes — Flat, IVF, PQ, HNSW — is fundamental knowledge for any practitioner building retrieval-augmented generation systems, recommendation engines, or large-scale semantic search applications.
