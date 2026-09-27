---
title: Content-Based Image Retrieval and Visual Search
description: Learn how image search works with learned embeddings and nearest neighbor indexes, from feature extraction through indexing, reranking, and evaluation.
---

Content-based image retrieval (CBIR) finds images similar to a query image (or text description) by comparing visual content instead of tags or filenames. It powers visual product search, duplicate detection, and photo library search.

## Core Pipeline

```text
offline:  images -> embedding model -> vectors -> ANN index
online:   query  -> embedding model -> vector  -> nearest neighbors -> (rerank) -> results
```

## Feature Representations

- **Handcrafted**: color histograms, SIFT, and bag-of-visual-words, which are largely historical
- **CNN / ViT embeddings**: global descriptors from pretrained backbones
- **Metric-learned embeddings**: networks trained with triplet or contrastive losses so similar items are close
- **Multimodal embeddings (CLIP, SigLIP)**: map images and text into one space, enabling text-to-image search and image-to-image search with the same index

## Metric Learning

A triplet loss pulls an anchor toward a positive and pushes it away from a negative:

```text
L = max(0, d(a, p) - d(a, n) + margin)
```

Hard-negative mining is crucial for good embeddings.

## Indexing for Scale

Exact search is too slow for millions of vectors. Approximate nearest neighbor (ANN) methods trade a little recall for large speedups:

| Method | Idea |
|--------|------|
| HNSW | Navigable small-world graph |
| IVF | Cluster vectors, search the nearest clusters |
| Product quantization | Compress vectors into compact codes |

```python
import faiss, numpy as np

d = 512
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.add(embeddings)                 # L2-normalized float32 (N, d)
scores, ids = index.search(query_vec, 10)
```

## Improving Results

- **Reranking**: re-score top candidates using local feature matching or a stronger model
- **Query expansion**: average the query with top results and search again
- **Region-level features** for object-centric search
- **Filtering** with metadata such as category, price, or date alongside vector search

## Evaluation

Use Recall@k, mean average precision (mAP), and NDCG on labeled query sets. Landmark datasets such as Oxford and Paris are standard benchmarks.

## Practical Concerns

- Domain shift: general embeddings may fail on specialized imagery (medical, industrial), so fine-tune on your data
- Near-duplicate and copyright detection often needs specialized perceptual hashing or copy-detection models
- Privacy and consent matter when indexing images of people
