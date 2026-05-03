---
title: Introduction to LanceDB
description: Get started with LanceDB — the open-source, serverless vector database built on the Lance columnar format. Learn embedded and cloud modes, Python API for CRUD and ANN search, IVF-PQ and HNSW indexing, full-text search, multimodal tables, LangChain and LlamaIndex integrations, and dataset versioning with time-travel queries.
---

**LanceDB** is an open-source vector database built on top of the **Lance** columnar file format — the same format designed for efficient random-access reads in ML training workloads. Unlike most vector databases that run as separate services, LanceDB operates in **embedded mode** with no server required: it runs in-process like SQLite, storing data as Lance files on local disk or cloud object storage (S3, GCS, Azure Blob). This architecture makes it exceptionally easy to get started and ideal for local development, edge deployment, and single-node retrieval applications.

## Installation and Setup

```bash
pip install lancedb
# Optional: for full-text search
pip install tantivy
# For LangChain/LlamaIndex integrations
pip install langchain-community llama-index-vector-stores-lancedb
```

## Core Concepts: The Lance Format

Lance stores data in a columnar format that provides:

- **Zero-copy reads**: ML training pipelines can memory-map Lance files and read random rows without deserializing entire column groups
- **Apache Arrow-compatible**: data is read directly as Arrow arrays, interoperating with pandas, Polars, and PyArrow
- **Versioned, append-only**: every write creates a new version; old versions remain accessible (time-travel queries)
- **Embedded vector index**: IVF-PQ and HNSW indexes stored alongside the data

## Basic Usage: CRUD and Vector Search

```python
import lancedb
import numpy as np
import pyarrow as pa
from sentence_transformers import SentenceTransformer

# ── Connect (creates directory if it doesn't exist) ────────────────────────
# Embedded mode: just a local path or s3:// / gs:// / az:// URI
db = lancedb.connect("./my_lancedb")

# For managed LanceDB Cloud:
# db = lancedb.connect("db://my-org/my-project", api_key=os.environ["LANCEDB_API_KEY"])

# ── Create a table with schema ─────────────────────────────────────────────
schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 384)),  # 384-dim embeddings
    pa.field("source", pa.string()),
    pa.field("timestamp", pa.timestamp("s")),
])

# Create empty table (or open existing)
table = db.create_table("documents", schema=schema, mode="overwrite")

# ── Insert data ────────────────────────────────────────────────────────────
encoder = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "The transformer architecture revolutionized NLP",
    "Diffusion models generate images by denoising",
    "Reinforcement learning trains agents via rewards",
    "Graph neural networks operate on structured data",
]

embeddings = encoder.encode(texts, normalize_embeddings=True)

data = [
    {
        "id": f"doc_{i}",
        "text": text,
        "vector": embedding.tolist(),
        "source": "ai_textbook",
        "timestamp": np.datetime64("now", "s"),
    }
    for i, (text, embedding) in enumerate(zip(texts, embeddings))
]

table.add(data)

# ── Vector search ──────────────────────────────────────────────────────────
query_text = "attention mechanisms in deep learning"
query_vector = encoder.encode([query_text], normalize_embeddings=True)[0]

results = (
    table.search(query_vector)
         .limit(3)
         .select(["id", "text", "source"])   # column projection
         .to_pandas()
)

print(results)
#         id                                           text       source
# 0    doc_0  The transformer architecture revolutionized NLP  ai_textbook
# 1    doc_2  Reinforcement learning trains agents via rewards  ai_textbook
# 2    doc_3   Graph neural networks operate on structured data  ai_textbook
```

## ANN Indexing: IVF-PQ and HNSW

For datasets beyond ~100K vectors, creating an Approximate Nearest Neighbor (ANN) index is essential:

```python
# ── IVF-PQ index: fast, memory-efficient, good for large datasets ──────────
table.create_index(
    metric="cosine",        # or "l2", "dot"
    index_type="IVF_PQ",
    num_partitions=256,     # number of Voronoi cells (IVF clusters)
    num_sub_vectors=48,     # PQ codebook count (must divide embedding dim)
    num_bits=8,             # bits per sub-quantizer (8 = 256 centroids)
    # IVF-PQ trades recall for speed; tune num_partitions ≈ sqrt(N)
)

# ── HNSW index: higher recall, more memory, faster query ──────────────────
table.create_index(
    metric="cosine",
    index_type="HNSW",
    m=16,                   # number of bidirectional connections per layer
    ef_construction=128,    # size of dynamic candidate list during construction
)

# ── Search with index ─────────────────────────────────────────────────────
results = (
    table.search(query_vector)
         .limit(10)
         .nprobes(32)         # IVF-PQ: number of clusters to search (higher = better recall)
         .refine_factor(4)    # re-rank top N*refine_factor candidates with exact distance
         .to_pandas()
)
```

## Pre-Filtering: Combining Vector Search with SQL Predicates

LanceDB supports pre-filtering (applied before ANN search) and post-filtering:

```python
from datetime import datetime, timedelta

# Filter documents from the last 30 days before vector search
cutoff = datetime.now() - timedelta(days=30)

results = (
    table.search(query_vector)
         .where(f"source = 'ai_textbook' AND timestamp > '{cutoff.isoformat()}'")
         .limit(5)
         .to_pandas()
)

# Pure SQL query (no vector search)
recent_docs = table.to_lance().to_table(
    filter="source = 'ai_textbook'",
    columns=["id", "text"]
).to_pandas()
```

## Full-Text Search

LanceDB supports BM25 full-text search via the Tantivy engine — enabling hybrid search (vector + keyword):

```python
# Create full-text search index on the "text" column
table.create_fts_index("text", replace=True)

# FTS query
fts_results = table.search("transformer attention").limit(5).to_pandas()

# Hybrid search: combine FTS + vector scores
hybrid_results = (
    table.search(query_vector, query_type="hybrid")
         .limit(5)
         .to_pandas()
)
```

## Multimodal Tables

Lance's columnar format handles arbitrary data types — images, audio, and text can live in the same table:

```python
import PIL.Image
import io

def image_to_bytes(image_path: str) -> bytes:
    """Serialize image to bytes for storage in Lance."""
    with open(image_path, "rb") as f:
        return f.read()

# Schema for a multimodal image-text table
multimodal_schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("image_bytes", pa.binary()),     # raw image bytes
    pa.field("text_vector", pa.list_(pa.float32(), 512)),   # CLIP text embedding
    pa.field("image_vector", pa.list_(pa.float32(), 512)),  # CLIP image embedding
    pa.field("category", pa.string()),
])

image_table = db.create_table("images", schema=multimodal_schema)

# Search images by text query using CLIP embeddings
def search_images_by_text(query: str, clip_model, n: int = 5):
    """Find images semantically matching a text query."""
    import torch
    text_inputs = clip_model.tokenize([query])
    with torch.no_grad():
        text_embed = clip_model.encode_text(text_inputs).numpy()[0]
    
    return (
        image_table.search(text_embed, vector_column_name="image_vector")
                   .limit(n)
                   .select(["id", "caption", "category"])
                   .to_pandas()
    )
```

## Versioning and Time-Travel

Every write to a LanceDB table creates an immutable version, enabling time-travel queries:

```python
# Check table version history
print(table.version)    # current version number
print(table.list_versions())  # list all versions with timestamps

# Add more data (creates new version)
table.add([{"id": "doc_10", "text": "New document", "vector": [...], ...}])
print(table.version)    # incremented

# Restore to a previous version (non-destructive: creates new version = old state)
table.restore(version=1)

# Query a specific historical version
old_table = db.open_table("documents", version=1)
old_results = old_table.search(query_vector).limit(3).to_pandas()
```

## LangChain and LlamaIndex Integration

```python
# ── LangChain integration ──────────────────────────────────────────────────
from langchain_community.vectorstores import LanceDB as LangChainLanceDB
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Build a RAG vector store from documents
texts = ["chunk 1 content", "chunk 2 content", "chunk 3 content"]
vectorstore = LangChainLanceDB.from_texts(
    texts,
    embedding=embeddings,
    connection=db,
    table_name="langchain_docs"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── LlamaIndex integration ─────────────────────────────────────────────────
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = LanceDBVectorStore(uri="./my_lancedb", table_name="llama_docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
query_engine = index.as_query_engine()
response = query_engine.query("What is attention in transformers?")
```

## LanceDB vs. Other Vector Databases

| Feature | LanceDB | Qdrant | Chroma | Pinecone |
| --- | --- | --- | --- | --- |
| Deployment | Embedded / cloud | Server / cloud | Embedded / server | Managed cloud only |
| Storage backend | Lance files (disk/S3) | Custom on-disk | DuckDB / Parquet | Proprietary |
| Versioning | Built-in time-travel | No | No | No |
| Multimodal native | Yes | Limited | No | No |
| Full-text search | Yes (Tantivy) | Yes | No | No |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | Proprietary |
| ML training integration | Excellent (Lance format) | Limited | Limited | None |

LanceDB is particularly well-suited for applications where the database lives alongside the application code (no infrastructure to manage), where ML training and inference share the same dataset (Lance format is read efficiently by PyTorch and JAX dataloaders), or where the dataset evolves over time and historical states need to be queryable.
