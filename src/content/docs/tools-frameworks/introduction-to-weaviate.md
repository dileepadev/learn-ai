---
title: Introduction to Weaviate
description: Master Weaviate, the AI-first vector database featuring built-in vectorization modules, hybrid keyword-vector search, GraphQL querying, and generative search integrations.
---

**Weaviate** is an open-source, AI-first vector search engine and database developed to store both data objects and vector embeddings. It bridges the gap between traditional search engines (like Elasticsearch) and specialized vector indexes (like FAISS) by providing native **hybrid search (BM25 + vector search)**, **graph-like cross-references**, and **modular vectorizer pipelines**.

With Weaviate, developers do not need to build manual embedding pipelines in Python before ingesting text, images, or audio. Weaviate can manage vectorization internally through plug-and-play **vectorizer modules** (`text2vec-openai`, `text2vec-cohere`, `text2vec-ollama`, `multi2vec-clip`).

---

## High-Level Architecture

Weaviate stores data in a dual-index design: every object is simultaneously registered in an **inverted index** (for keyword search and scalar filtering) and an **ANN vector index** (HNSW or Flat):

```
                                [ Incoming Data Object ]
                                            │
                                            ▼
                          ┌──────────────────────────────────┐
                          │    Vectorizer Module (Optional)  │
                          │  (OpenAI / Cohere / Local Model) │
                          └─────────────────┬────────────────┘
                                            │ Generates Embedding
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Weaviate Storage Engine                                                               │
│                                                                                       │
│  ┌───────────────────────────────┐         ┌────────────────────────────────────────┐ │
│  │ Inverted Index (BM25)         │         │ Vector Index (HNSW / Flat)             │ │
│  │ Keyword tokens, posting lists,│         │ Multi-layer proximity graph            │ │
│  │ scalar attributes, timestamps │         │ for Cosine / Dot / L2 distance search  │ │
│  └───────────────────────────────┘         └────────────────────────────────────────┘ │
│                                  │         │                                          │
│                                  ▼         ▼                                          │
│                    [ Hybrid Fusion Operator: α · Vector + (1 - α) · BM25 ]            │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                             [ Generative Search (RAG) ]
                             (Summarize, extract, prompt)
```

---

## Native Hybrid Search

Pure vector search excels at broad semantic concepts (e.g., matching *"feline companion"* to *"domestic cat"*), but struggles with exact alphanumeric strings, serial numbers, stock keeping units (SKUs), and rare named entities. Lexical BM25 search excels at exact matches, but fails on synonyms.

Weaviate resolves this with **native hybrid search**, parameterized by a balancing factor $\alpha \in [0, 1]$:

$$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{dense}} + (1 - \alpha) \cdot \text{Score}_{\text{sparse}}$$

- **$\alpha = 1.0$:** Pure vector similarity search.
- **$\alpha = 0.0$:** Pure BM25 keyword search.
- **$\alpha = 0.5$ (Recommended Default):** Equal weighting between dense semantics and sparse keyword occurrences.

Scores are automatically normalized using **Relative Score Fusion (RSF)** or **Reciprocal Rank Fusion (RRF)** before combining.

---

## Hands-On with Weaviate Python Client v4

The Weaviate Python Client (v4) features a modern, type-safe API with native gRPC support for high-throughput batching.

### Installation

```bash
pip install weaviate-client
```

### 1. Connecting to Weaviate

```python
import weaviate
from weaviate.classes.init import Auth

# Option A: Connect to local Docker instance
client = weaviate.connect_to_local(port=8080, grpc_port=50051)

# Option B: Connect to Weaviate Cloud (WCS)
# client = weaviate.connect_to_weaviate_cloud(
#     cluster_url="https://your-cluster.weaviate.network",
#     auth_credentials=Auth.api_key("your-api-key"),
#     headers={"X-OpenAI-Api-Key": "sk-..."}
# )
```

### 2. Creating a Collection with Auto-Vectorization

```python
import weaviate.classes.config as wvc

# Define a collection with built-in text2vec-ollama or text2vec-openai
articles = client.collections.create(
    name="TechnicalArticle",
    vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    generative_config=wvc.Configure.Generative.openai(
        model="gpt-4o-mini"
    ),
    properties=[
        wvc.Property(name="title", data_type=wvc.DataType.TEXT),
        wvc.Property(name="content", data_type=wvc.DataType.TEXT),
        wvc.Property(name="published_year", data_type=wvc.DataType.INT),
    ]
)
```

### 3. Inserting Documents (Auto-Embedding Generation)

```python
articles = client.collections.get("TechnicalArticle")

# Ingest records — Weaviate automatically calls the embedding model in background
with articles.batch.dynamic() as batch:
    batch.add_object(
        properties={
            "title": "State Space Models in Modern AI",
            "content": "Mamba and S4 architectures replace attention with linear recurrence.",
            "published_year": 2024
        }
    )
    batch.add_object(
        properties={
            "title": "Low Rank Adaptation (LoRA)",
            "content": "LoRA freezes pretrained weights and injects trainable rank decomposition matrices.",
            "published_year": 2023
        }
    )
```

### 4. Executing Hybrid and Generative Search (RAG in One Call)

Weaviate allows executing semantic search and RAG synthesis in a single unified API query:

```python
from weaviate.classes.query import MetadataQuery

response = articles.generate.hybrid(
    query="efficient fine-tuning with small matrices",
    alpha=0.75, # 75% vector, 25% keyword BM25
    limit=2,
    single_prompt="Summarize this technique in one bullet point: {content}",
    return_metadata=MetadataQuery(score=True)
)

for obj in response.objects:
    print(f"Title: {obj.properties['title']} (Score: {obj.metadata.score:.4f})")
    print(f"Generative Summary: {obj.generated}\n")
```

---

## Core Feature Comparison: Weaviate vs. Alternatives

| Feature | Weaviate | Pinecone | ChromaDB |
| :--- | :--- | :--- | :--- |
| **Open Source** | Yes (Apache 2.0) | No (Proprietary SaaS) | Yes (Apache 2.0) |
| **Self-Hosting** | Docker / Kubernetes / Local | Cloud Only | Python / Docker |
| **Built-In Vectorization**| Yes (OpenAI, Cohere, HuggingFace, Ollama)| No (Client pre-embeds)| Yes (Sentence Transformers) |
| **Native Hybrid Search** | Yes (BM25 + Vector Fusion) | Yes | Limited / Manual |
| **Generative Search (RAG)**| Yes (Direct single-call generation) | No | No |
| **Protocol** | High-performance gRPC + REST | REST / gRPC | REST / Python IPC |

---

## Key Takeaways

- Weaviate combines an inverted text index with HNSW vector graphs, providing seamless out-of-the-box hybrid search.
- Modular vectorizers eliminate the need to write custom embedding pipelines, translating raw text and images to vectors on ingestion.
- The v4 Python SDK leverages gRPC to deliver low-latency batching and direct generative RAG synthesis.
