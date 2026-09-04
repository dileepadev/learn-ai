---
title: Introduction to Milvus
description: Discover Milvus, the cloud-native distributed vector database designed for massive-scale similarity search, covering its disaggregated architecture, IVF-PQ/HNSW indexes, and collection sharding.
---

**Milvus** is an open-source, cloud-native vector database created by Zilliz and graduated under the LF AI & Data Foundation. Specifically engineered to manage and query massive collections of unstructured data, Milvus routinely scales to **billions of high-dimensional vectors** with sub-10 millisecond search latencies.

While lightweight embedded vector stores work well for thousands or millions of documents on a single machine, enterprise-grade AI applications (such as e-commerce recommendation engines, national security biometric matching, and planetary-scale search) require true distributed horizontal scalability, high availability, and separation of storage and compute.

---

## Disaggregated Cloud-Native Architecture

Milvus adheres to a modern **cloud-native, disaggregated architecture** that cleanly decouples state from compute across four distinct layers:

```
                            [ Client Application (PyMilvus, Go, Java, REST) ]
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Access Layer (Stateless Proxies)                                                         │
│    Routes queries, validates requests, and aggregates distributed results                   │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Coordinator Services (etcd)                                                              │
│    Root Coord (DDL)  |  Data Coord (Allocation)  |  Query Coord (Shards)  |  Index Coord     │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Worker Node Cluster (Stateless Compute)                                                  │
│    • Query Nodes: In-memory ANN search on active segment data                                │
│    • Data Nodes: Consumes log stream and flushes immutable segments                          │
│    • Index Nodes: Builds CPU/GPU indexes (HNSW, IVF-PQ, DiskANN) asynchronously              │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Storage & Message Broker (Persistent State)                                              │
│    • Log Broker (Kafka / Apache Pulsar): Write-Ahead Log (WAL) for streaming data           │
│    • Object Storage (MinIO / AWS S3 / Google Cloud Storage): Immutable segment blobs        │
│    • Meta Store (etcd): Cluster topology and schema metadata                                │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Strengths
- **Independent Elasticity:** Query nodes can autoscale horizontally during high query-per-second (QPS) spikes without altering storage or ingestion nodes.
- **Fail-Fast Recovery:** Because worker nodes are completely stateless, a crashed query node can be replaced in seconds by re-mounting immutable segments from object storage.
- **Strict Data Consistency:** Every write passes through the log broker, guaranteeing configurable consistency levels (Strong, Bounded Staleness, Session, or Eventually Consistent).

---

## Indexing Algorithms in Milvus

A vector database is only as fast as its approximate nearest neighbor (ANN) indexes. Milvus supports a comprehensive suite of hardware-optimized index structures:

| Index Type | Search Speed | Memory Footprint | Recall Rate | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **FLAT** | Slow ($O(N)$ brute-force) | High (raw uncompressed) | $100\%$ (Exact) | Small collections ($<50\text{k}$) or exact baseline tests |
| **IVF-FLAT** | Fast | Moderate | High | Medium datasets; clusters space into Voronoi cells |
| **IVF-PQ** | Extremely Fast | Very Low (quantized) | Moderate | Massive datasets with limited RAM (compresses vectors $16\times$) |
| **HNSW** | Ultra-Fast | High | Very High ($\ge 98\%$) | Latency-critical applications where RAM is abundant |
| **SCaNN** | Ultra-Fast | Low to Moderate | Very High | Anisotropic vector quantization developed by Google Research |
| **DiskANN** | Fast | Extremely Low (SSD-based) | High | Billion-scale collections running on cost-effective NVMe drives |

---

## Hands-On with PyMilvus

### Installation

```bash
pip install pymilvus
```

### 1. Connecting and Defining a Schema

```python
from pymilvus import MilvusClient, DataType

# MilvusClient supports local Milvus Lite or remote distributed clusters
client = MilvusClient(uri="http://localhost:19530")

# 1. Define Schema
schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=True # Supports arbitrary JSON attributes
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64)

# 2. Define Index Parameters (HNSW)
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 200}
)

# 3. Create Collection
client.create_collection(
    collection_name="multimodal_catalog",
    schema=schema,
    index_params=index_params
)
```

### 2. Inserting Data and Conducting Vector Search

```python
import numpy as np

# Generate dummy vector data (128 dimensions)
data = [
    {
        "id": i,
        "vector": np.random.randn(128).astype(np.float32).tolist(),
        "category": "electronics" if i % 2 == 0 else "apparel",
        "stock": 100 + i
    }
    for i in range(1000)
]

# Insert records
client.insert(collection_name="multimodal_catalog", data=data)

# Perform ANN Search with Scalar Filtering
query_vector = np.random.randn(128).astype(np.float32).tolist()

search_results = client.search(
    collection_name="multimodal_catalog",
    data=[query_vector],
    filter="category == 'electronics' and stock > 500", # Hybrid filtering
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    output_fields=["category", "stock"]
)

for hits in search_results:
    for hit in hits:
        print(f"ID: {hit['id']} | Distance: {hit['distance']:.4f} | Entity: {hit['entity']}")
```

---

## Deployment Flavors

1. **Milvus Lite:** Runs as a lightweight local Python library without Docker or Kubernetes—ideal for rapid local testing and CI/CD pipelines.
2. **Milvus Standalone:** Packaged via Docker Compose, running all components in a single container host for medium-scale enterprise deployments ($\le 10\text{ million vectors}$).
3. **Milvus Distributed:** Deployed on Kubernetes via the official Helm Chart or Milvus Operator, offering auto-scaling and cross-region replication for enterprise workloads ($\ge 1\text{ billion vectors}$).
