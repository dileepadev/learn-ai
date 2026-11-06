---
title: Introduction to Vespa - Search and Vector Retrieval Engine
description: Learn how Vespa combines vector search, lexical ranking, and machine-learned ranking in one engine for large-scale, low-latency retrieval applications.
---

Vespa is an open-source engine for large-scale search, recommendation, and retrieval. Unlike purely vector-oriented databases, it combines full-text search, structured filtering, approximate nearest neighbor search, and on-node model inference in a single query path.

## What Sets It Apart

- **Hybrid retrieval**: combine BM25 and vector similarity in one query
- **Multi-phase ranking**: cheap first-phase scoring over many candidates, then expensive models over the top results
- **Real-time updates**: writes are searchable quickly, without full reindexing
- **Tensor support**: first-class multi-dimensional data, including multi-vector (late interaction) representations
- **Horizontal scaling**: data is distributed across content nodes automatically

## Schema Example

```text
schema doc {
  document doc {
    field title type string { indexing: index | summary }
    field body type string { indexing: index | summary }
    field embedding type tensor<float>(x[384]) {
      indexing: attribute | index
      attribute { distance-metric: angular }
    }
  }
  rank-profile hybrid {
    inputs { query(q) tensor<float>(x[384]) }
    first-phase {
      expression: bm25(title) + bm25(body) + closeness(field, embedding)
    }
  }
}
```

## Querying

```bash
vespa query 'yql=select * from doc where userQuery() or ({targetHits:100}nearestNeighbor(embedding, q))' \
  'query=refund policy' 'ranking=hybrid' 'input.query(q)=embed(refund policy)'
```

## When to Consider Vespa

It fits applications needing hybrid search, personalization, and ranking control at scale. For a small prototype, a simpler vector store is quicker to adopt; Vespa's strength appears as ranking complexity and data volume grow.
