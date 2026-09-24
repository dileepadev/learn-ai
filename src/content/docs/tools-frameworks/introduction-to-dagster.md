---
title: Introduction to Dagster - Asset-Centric Data and ML Orchestration
description: Learn how Dagster's software-defined assets model data and ML pipelines around the data they produce, not just the tasks that produce it.
---

Dagster is a data and ML orchestration tool built around software-defined assets: instead of modeling a pipeline as a sequence of tasks, Dagster models it as a graph of the data artifacts (tables, files, models) those tasks produce.

## Software-Defined Assets

An asset declaration describes what a piece of data is and how to compute it, and Dagster infers the dependency graph from which assets each function reads:

```python
from dagster import asset

@asset
def raw_documents():
    return load_documents_from_source()

@asset
def document_embeddings(raw_documents):
    return embed(raw_documents)

@asset
def vector_index(document_embeddings):
    return build_index(document_embeddings)
```

Dagster sees that `document_embeddings` depends on `raw_documents` and `vector_index` depends on `document_embeddings` purely from the function arguments, and can materialize, track, and re-run only the assets affected by an upstream change.

## Why This Matters for ML Pipelines

Task-centric orchestrators answer "did this job run successfully?" Asset-centric orchestration answers "is this dataset or model up to date, and what does it depend on?" — a distinction that matters enormously for ML pipelines where you care about data freshness and lineage (which model was trained on which version of which dataset) as much as job completion. Dagster's asset catalog gives a visual, queryable map of these dependencies across an entire pipeline.

## Data Quality and Testing

Dagster supports asset checks — validation logic that runs against materialized data (row counts, null checks, schema conformance, or LLM-output quality checks) and surfaces failures directly in the asset catalog, integrating data quality monitoring into the orchestration layer rather than as a separate bolted-on system.

## Practical Guidance

Reach for Dagster over task-centric tools like Prefect or Airflow when data lineage and asset freshness tracking matter as much as execution scheduling — common in ML platforms that need to answer "what changed upstream that could explain this model's behavior change" quickly. The asset model has a steeper initial learning curve than plain task-based orchestration, so budget time to think in terms of data assets rather than procedural steps when migrating an existing pipeline.
