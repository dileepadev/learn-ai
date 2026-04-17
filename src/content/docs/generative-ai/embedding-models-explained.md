---
title: Embedding Models Explained
description: Learn how text and multimodal embedding models work — covering BERT, sentence-transformers, late interaction models, and leading embeddings like E5, BGE, and Cohere Embed for retrieval and semantic search.
---

**Embedding models** convert text (or other data) into dense numerical vectors — **embeddings** — that capture semantic meaning. Similar content maps to nearby points in the embedding space; dissimilar content maps to distant points. They are the foundational backbone of semantic search, RAG pipelines, clustering, classification, and recommendation systems.

## What Is an Embedding?

An embedding is a real-valued vector $e \in \mathbb{R}^d$ (typically $d = 384$ to $4096$) that represents the semantic content of an input. The key property:

$$\cos(e_1, e_2) = \frac{e_1 \cdot e_2}{\|e_1\| \|e_2\|}$$

High cosine similarity → semantically close. This geometric structure enables fast nearest-neighbor search across millions of documents.

## From Word Embeddings to Sentence Embeddings

### Word2Vec and GloVe

Early embedding models produced **word-level** vectors — each word maps to a fixed vector regardless of context. While revolutionary, they could not handle polysemy (the word "bank" means something different in "river bank" vs. "bank account").

### BERT and Contextual Embeddings

**BERT** (Devlin et al., 2018) introduced **contextual embeddings** — the same word receives a different vector depending on its surrounding context. BERT encodes a sentence with a Transformer, producing per-token vectors.

**Problem:** BERT was designed for token-level tasks (NER, fill-mask). For sentence-level similarity, averaging all token vectors or using the `[CLS]` token performs poorly because BERT was not trained to produce meaningful sentence-level representations.

### Sentence-BERT (SBERT)

**Sentence-BERT** (Reimers & Gurevych, 2019) fine-tuned BERT using a **Siamese network** with a contrastive objective on Natural Language Inference (NLI) pairs:

- Semantically entailing sentence pairs → trained to produce similar embeddings.
- Contradicting pairs → trained to produce dissimilar embeddings.

A mean-pooling strategy over all token vectors produces the final sentence embedding. The result: a model that produces semantically meaningful sentence-level representations ~20–80× faster than comparing sentences with cross-encoders.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "A dog is playing in the park.",
    "A puppy runs through the garden.",
    "The stock market fell dramatically today."
]

embeddings = model.encode(sentences)
# Shape: (3, 384)
```

## Bi-Encoders vs. Cross-Encoders

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| **How** | Encodes query and document independently | Processes query + document jointly |
| **Speed** | Fast — embed once, search with ANN | Slow — must re-run for every pair |
| **Accuracy** | Good | Higher (sees interaction between query and doc) |
| **Use case** | First-stage retrieval (millions of docs) | Re-ranking top-K results |

The standard **two-stage retrieval** pipeline: bi-encoder retrieves top-K candidates → cross-encoder re-ranks them for final answer.

## Training Objectives for Embedding Models

Modern high-quality embedding models use several training techniques:

### Contrastive Learning (SimCSE)

**SimCSE** creates positive pairs by passing the same sentence through the model twice with different dropout masks — no label data required. Negatives are other sentences in the batch.

### Multiple Negatives Ranking (MNR) Loss

For a batch of $(q_i, d_i^+)$ pairs, treat all other documents $d_j^+$ ($j \neq i$) as hard negatives:

$$L = -\log \frac{e^{\cos(q_i, d_i^+)/\tau}}{\sum_j e^{\cos(q_i, d_j^+)/\tau}}$$

Scales effectively with batch size — larger batches provide more negatives.

### Hard Negative Mining

Mining **hard negatives** — documents that are similar but not relevant — significantly improves embedding quality for retrieval tasks. Hard negatives come from BM25 top results that aren't relevant, or from a weaker embedding model's retrievals.

### Matryoshka Representation Learning (MRL)

MRL trains embeddings so that the **first $k$ dimensions** of a $d$-dimensional vector are themselves a meaningful $k$-dimensional embedding. This enables:

- Using a single model at multiple dimensions ($64$, $128$, $256$, $1024$) trading off between speed and accuracy.
- Compressing embeddings for memory-constrained deployments.

## Leading Embedding Models (2025)

| Model | Dims | Params | Strengths |
|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | 3072 | — | Strong general-purpose |
| `text-embedding-3-small` (OpenAI) | 1536 | — | Fast, cost-effective |
| `embed-v4.0` (Cohere) | 1024 | — | Multilingual, binary quantization |
| `E5-large-v2` (Microsoft) | 1024 | 335M | Strong retrieval |
| `BGE-M3` (BAAI) | 1024 | 568M | Multi-lingual, multi-granularity |
| `all-MiniLM-L6-v2` (SBERT) | 384 | 22M | Fast, lightweight |
| `nomic-embed-text` | 768 | 137M | Long context (8192 tokens), open |
| `Jina-embeddings-v3` | 1024 | 570M | Task-specific LoRA adapters |

## Instruction-Tuned Embeddings

Models like **E5** and **Instructor** accept a task instruction prefix that customizes the embedding for specific use cases:

```python
# E5 style
query = "query: How does photosynthesis work?"
document = "passage: Photosynthesis is the process..."

# Instructor style
model.encode([["Represent the science question for retrieval:", query]])
```

This allows a single model to produce specialized embeddings for retrieval, clustering, classification, or reranking tasks.

## Late Interaction: ColBERT

**ColBERT** (Khattab & Zaharia, 2020) is a hybrid approach: it encodes query and document independently (like a bi-encoder) but computes similarity via **late interaction** — comparing all query token vectors against all document token vectors:

$$\text{score}(q, d) = \sum_{i \in q} \max_{j \in d} e_{q_i} \cdot e_{d_j}$$

This is more expressive than single-vector bi-encoders but cheaper than full cross-encoders. ColBERT achieves near cross-encoder quality at near bi-encoder speed using compressed token vectors.

## Multimodal Embeddings

Embedding models have been extended to jointly encode multiple modalities:

- **CLIP** — Text and image embeddings in a shared space; enables image retrieval by text query.
- **ImageBind** (Meta) — Six modalities (text, image, audio, depth, thermal, IMU) in a single embedding space.
- **VoyagerEmbed / MMTE** — Embed interleaved text and image inputs for multi-modal RAG.

## Embedding Quality: MTEB Benchmark

The **Massive Text Embedding Benchmark (MTEB)** is the standard evaluation for embedding models, covering 56 datasets across 8 tasks:

- Retrieval, clustering, classification, reranking, semantic textual similarity, summarization, bitext mining, pair classification.

The MTEB leaderboard at [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is the authoritative source for model comparison.

## Practical Tips

- **Match embedding dimensions to use case** — Smaller dims are faster for large-scale retrieval; larger dims have higher ceiling accuracy.
- **Use the same model for queries and documents** — Mixing models produces incompatible embedding spaces.
- **Normalize embeddings** — Cosine similarity requires unit-normalized vectors; most libraries do this by default.
- **Binary quantization** — Quantize float32 embeddings to 1-bit (Hamming distance) for 32× memory reduction with ~5% accuracy drop — Cohere's Embed v4 supports this natively.
- **Long documents** — Most embedding models have a 512-token limit. For longer documents, chunk and embed chunks; optionally embed a summary.
