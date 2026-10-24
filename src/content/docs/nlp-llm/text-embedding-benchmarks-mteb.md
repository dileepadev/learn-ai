---
title: MTEB - Benchmarking Text Embedding Models
description: Learn how the Massive Text Embedding Benchmark evaluates embedding models across retrieval, classification, clustering, and other tasks, and how to use it to pick a model.
---

The Massive Text Embedding Benchmark (MTEB) is a standard suite for comparing text embedding models across many tasks and datasets. It was created because models evaluated on a single task, such as semantic similarity, often generalize poorly to others.

## Task Categories

| Category | What is measured | Main metric |
|----------|------------------|-------------|
| Retrieval | Find relevant documents for queries | NDCG@10 |
| Semantic textual similarity | Correlate similarity with human scores | Spearman correlation |
| Classification | Train a logistic regression on frozen embeddings | Accuracy |
| Clustering | Group texts without labels | V-measure |
| Pair classification | Decide whether two texts are duplicates or related | Average precision |
| Reranking | Reorder candidate lists | MAP |
| Summarization | Match summary quality judgments | Spearman correlation |

Extensions add multilingual coverage, domain-specific sets, and long documents.

## Running an Evaluation

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
evaluation = MTEB(tasks=["SciFact", "STSBenchmark"])
results = evaluation.run(model, output_folder="results")
```

## Reading the Leaderboard

- Check the **task type** that matches your use case; a model leading overall may not lead on retrieval
- Consider **model size, embedding dimension, and max sequence length**, which determine cost and latency
- Watch for **multilingual** needs
- Be skeptical of top scores: some models may have trained on benchmark datasets, inflating results

## Practical Selection Process

1. Shortlist models using MTEB for your task type and language
2. Build a small evaluation set from your own queries and documents
3. Compare shortlisted models on that set for recall and ranking metrics
4. Weigh accuracy against index size and inference cost

Lower-dimensional embeddings (or Matryoshka-truncated ones) can cut storage and search cost substantially with modest quality loss.

## Limitations

Benchmarks cannot capture your domain's vocabulary or query style, and aggregated averages hide per-task variation, so treat MTEB as a filter, not a final verdict.
