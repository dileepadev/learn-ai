---
title: Text Clustering Techniques - Grouping Documents Without Labels
description: A practical overview of clustering documents using embeddings, from choosing a distance metric to picking the number of clusters.
---

Text clustering groups similar documents together without predefined labels, useful for exploring large unlabeled corpora, organizing support tickets, or discovering duplicate or near-duplicate content.

## Pipeline

A typical modern text clustering pipeline embeds each document with a sentence or document embedding model, reduces dimensionality if needed, and then applies a clustering algorithm.

```text
documents -> embedding model -> vectors -> (optional) dimensionality reduction -> clustering algorithm -> cluster labels
```

Cosine similarity, not Euclidean distance, is typically the right metric for embedding vectors, since embedding magnitude often carries little meaning compared to direction.

## Choosing a Clustering Algorithm

K-means requires specifying the number of clusters in advance and assumes roughly spherical, similar-sized clusters, which rarely matches real text distributions well but is fast and simple as a starting point. HDBSCAN discovers the number of clusters automatically and handles noise points (documents that don't belong to any cluster) explicitly, at the cost of more hyperparameters (minimum cluster size) to tune. Agglomerative clustering builds a hierarchy of merges and lets you cut the tree at different granularities without re-running the algorithm.

## Picking the Number of Clusters

For k-means, the elbow method (plotting within-cluster variance against `k`) and silhouette scores give rough guidance, but for text data these metrics are often noisy — validate by reading sample documents from each cluster rather than trusting a single numeric criterion.

## Labeling Clusters

Clustering only assigns group membership; labeling requires a separate step, such as extracting top TF-IDF terms per cluster, using an LLM to summarize a sample of documents from each cluster into a short label, or applying [[topic-modeling-with-lda]]-style techniques within each cluster for finer structure.

## Practical Guidance

Deduplicate and normalize text before clustering — near-identical documents can dominate a cluster and skew centroid-based algorithms. Re-embed with a domain-appropriate model if your text is highly specialized (legal, medical, code) since general-purpose embedding models cluster domain jargon poorly.
