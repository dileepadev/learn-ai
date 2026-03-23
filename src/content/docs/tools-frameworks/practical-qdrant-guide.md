---
title: Practical Qdrant Guide
description: How to set up and use Qdrant for vector search in production.
---

Qdrant is an open-source vector database optimized for similarity search. This guide covers installation, data ingestion, and querying.

## Quickstart

- Install Qdrant via Docker or use the hosted cloud offering
- Convert documents to embeddings and upsert vectors into Qdrant
- Use the API to run nearest-neighbor searches and filter by metadata

## Best Practices

- Use sharding and replicas for high availability
- Monitor index size and set appropriate payload schemas
- Tune distance metric and indexing parameters for your embeddings
