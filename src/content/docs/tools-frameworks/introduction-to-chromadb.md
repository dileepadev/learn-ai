---
title: Introduction to ChromaDB
description: Learn how to use ChromaDB, the open-source AI embedding database, for semantic search, collection management, document ingestion, and building local RAG applications.
---

**ChromaDB** (or simply **Chroma**) is an open-source, developer-friendly embedding database designed from the ground up to empower AI applications, large language model (LLM) agents, and Retrieval-Augmented Generation (RAG) pipelines.

Unlike heavy, complex distributed vector databases that require dedicated DevOps infrastructure, Chroma focuses on developer ergonomics: it can run **embedded directly inside your Python or JavaScript process** with zero external dependencies, or scale out as a standalone client-server service via Docker.

---

## ChromaDB Architecture

Chroma decouples embedding search, metadata filtering, and document storage into an integrated local or distributed stack:

```
                          ┌───────────────────────────┐
                          │   Client Application      │
                          │   (Python / TypeScript)   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Chroma Core Engine                                                          │
│                                                                             │
│  ┌───────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐ │
│  │ Vector Index (HNSW)   │  │ Metadata & ID Store  │  │ Embedding Model  │ │
│  │ Approximate Nearest   │  │ SQLite / Arrow       │  │ Default: all-    │ │
│  │ Neighbor Search       │  │ Filtering ($eq, $in) │  │ MiniLM-L6-v2     │ │
│  └───────────────────────┘  └──────────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                           Persistent Storage on Disk
```

### Core Architecture Components
1. **Vector Index (HNSW):** Uses the Hierarchical Navigable Small World (HNSW) graph algorithm for fast, high-recall approximate nearest neighbor (ANN) search over Euclidean ($L_2$), Cosine, or Inner Product (IP) distances.
2. **Metadata & Relational Store:** Utilizes SQLite and Apache Arrow to support structured filtering alongside vector distance queries.
3. **Built-In Embedding Functions:** Automatically vectors raw text using lightweight transformer models (`all-MiniLM-L6-v2`) if an explicit vector is not provided by the caller.

---

## Key Concepts

- **Collection:** The primary unit of organization in Chroma (equivalent to a table in SQL or a collection in MongoDB). Collections contain documents, vector embeddings, and associated metadata.
- **Documents:** The raw textual content (e.g., code snippets, PDF paragraphs, customer support tickets).
- **Embeddings:** The numerical floating-point vector representations of documents.
- **Metadatas:** Key-value dictionaries attached to each vector for relational filtering (e.g., `{"author": "alice", "year": 2024}`).
- **IDs:** Unique string identifiers for each record.

---

## Getting Started with Python

### Installation

```bash
pip install chromadb
```

### 1. In-Memory vs. Persistent Client

Chroma provides two modes of operation in Python:

```python
import chromadb

# Mode A: Ephemeral client (data vanishes when script terminates)
in_memory_client = chromadb.Client()

# Mode B: Persistent client (data saved to local directory)
persistent_client = chromadb.PersistentClient(path="./my_chroma_db")
```

### 2. Creating a Collection & Ingesting Documents

```python
# Create or get an existing collection
collection = persistent_client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"} # Options: "cosine", "l2", "ip"
)

# Ingest documents (Chroma automatically computes embeddings if omitted)
collection.add(
    documents=[
        "Transformers use self-attention to process entire sequences in parallel.",
        "Convolutional neural networks apply learnable filters over 2D spatial feature maps.",
        "Diffusion models generate images by iteratively reversing a Markovian noise process."
    ],
    metadatas=[
        {"category": "nlp", "difficulty": "intermediate"},
        {"category": "vision", "difficulty": "beginner"},
        {"category": "generative", "difficulty": "advanced"}
    ],
    ids=["doc_1", "doc_2", "doc_3"]
)

print(f"Total documents indexed: {collection.count()}")
```

### 3. Querying with Natural Language

You can query the collection directly with natural language text:

```python
results = collection.query(
    query_texts=["How do vision models analyze images?"],
    n_results=2,
    where={"difficulty": "beginner"} # Metadata filtering
)

for doc_id, text, distance in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
    print(f"Match [{doc_id}] (Cosine Distance: {distance:.4f}):\n{text}\n")
```

---

## Metadata Filtering Operators

Chroma supports rich metadata operators for filtering results:

| Operator | Syntax Example | Description |
| :--- | :--- | :--- |
| **$eq** | `{"category": {"$eq": "nlp"}}` | Equality match |
| **$ne** | `{"status": {"$ne": "archived"}}` | Not equal to value |
| **$gt / $gte** | `{"year": {"$gte": 2023}}` | Greater than / greater than or equal |
| **$lt / $lte** | `{"price": {"$lt": 50}}` | Less than / less than or equal |
| **$in / $nin** | `{"tag": {"$in": ["python", "ai"]}}` | Match any / none in array |
| **$and / $or** | `{"$and": [{"year": {"$gte": 2024}}, {"category": "nlp"}]}` | Logical conjunctions |

---

## Client-Server Deployment with Docker

For production applications or multi-container microservice deployments, Chroma runs as a dedicated HTTP server:

```bash
docker run -d -p 8000:8000 -v ./chroma_data:/chroma/chroma chromadb/chroma:latest
```

Connect to the remote instance from any service:

```python
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("ai_knowledge_base")
```

---

## When to Choose ChromaDB

- **Ideal For:** Fast prototyping, hackathons, local desktop applications, and small-to-medium enterprise RAG pipelines ($\le 5\text{ million vectors}$).
- **Key Strengths:** Zero infrastructure setup, native Python/JS bindings, automatic embedding generation, and simple SQLite backing.
- **When to Upgrade:** For petabyte-scale, multi-billion vector clusters with distributed shard replication across Kubernetes, consider dedicated cloud engines like **Milvus** or **Qdrant**.
