---
title: Cross-Encoders and Neural Reranking
description: Explore two-stage retrieval architectures, cross-encoder self-attention over query-document pairs, Reciprocal Rank Fusion (RRF), and production reranking pipelines.
---

In modern search systems and Retrieval-Augmented Generation (RAG) pipelines, relying solely on dense vector search (bi-encoders) or lexical search (BM25) often yields noisy results. Dense embeddings compress an entire multi-paragraph document into a single fixed-length vector (e.g., 768 or 1536 floating-point numbers), which inevitably loses fine-grained nuances, numerical constraints, and specific keyword alignments.

To achieve maximum retrieval accuracy without sacrificing sub-second response times, production systems deploy a **Two-Stage Retrieval Architecture**:
1. **Stage 1 (First-Stage Retrieval):** High-recall, low-latency search across millions of documents using dense vector indexes and lexical search to extract the top $K \approx 50\text{--}100$ candidates.
2. **Stage 2 (Neural Reranking):** High-precision **Cross-Encoders** evaluate the top $K$ candidates with deep, full-attention cross-scoring to select the top $N \approx 3\text{--}5$ most relevant passages.

---

## Two-Stage Architecture Overview

```
Corpus (Millions of Docs)
           │
           ├──────────────────────────────┬──────────────────────────────┐
           ▼                                                             ▼
[ Lexical Search (BM25 / SPLADE) ]                             [ Vector Search (Bi-Encoder) ]
           │                                                             │
           └──────────────────────────────┬──────────────────────────────┘
                                          ▼
                         Top 100 Combined Candidates
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Stage 2: Cross-Encoder Reranker    │
                      │  Joint Cross-Attention: Query + Doc   │
                      └───────────────────┬───────────────────┘
                                          ▼
                           Top 3-5 Highly Accurate Passages
                                          │
                                          ▼
                             [ LLM Context Window ]
```

---

## Why Bi-Encoders Miss Critical Details

In a bi-encoder (such as standard Sentence Transformers), the query vector $\mathbf{q} = f(\text{query})$ and document vector $\mathbf{d} = f(\text{doc})$ are generated in total isolation:

$$\text{Score}_{\text{bi-encoder}} = \mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^D q_i d_i$$

There is no token-to-token cross-attention between the query words and the document words. This creates an **information bottleneck**:
- If a query asks: *"Can an H-1B visa holder invest in passive index funds without violating status?"*
- A bi-encoder may match documents generally discussing *"H-1B visa status"* or *"investing in index funds"*, but fail to capture the critical legal nuance of *"passive investing"* vs. *"active employment"*.

---

## How Cross-Encoders Work

A **Cross-Encoder** passes the query and document into a transformer model **simultaneously as a single concatenated input sequence**:

$$\mathbf{x} = \text{[CLS]} \, q_1, q_2, \dots, q_m \, \text{[SEP]} \, d_1, d_2, \dots, d_k \, \text{[SEP]}$$

```
Input Tokens: [CLS] Query Tokens [SEP] Document Tokens [SEP]
                    ▲                     ▲
                    └────── All-to-All ───┘
                    Full Cross-Attention Matrix in Every Layer
                              │
                              ▼
                      [ [CLS] Hidden State ]
                              │
                              ▼
                   [ Single Linear Output Head ]
                              │
                              ▼
                    Relevance Score s ∈ [0, 1]
```

### Advantages of Full Cross-Attention:
1. **Word-Level Interaction:** Every single token in the query attends directly to every token in the candidate document across all 12–24 transformer layers.
2. **Contextual Disambiguation:** Negations, qualifications (*"except"*, *"not"*, *"unless"*), and subtle conditions are parsed in context.
3. **Calibrated Confidence:** The output head produces an absolute relevance logit or probability score, making threshold filtering straightforward.

---

## Late Interaction: The ColBERT Alternative

Between fast bi-encoders and computationally heavy cross-encoders lies **Late Interaction**, pioneered by **ColBERT** (Khattab & Zaharia):

```
Query Tokens    ──► [ BERT ] ──► Sequence of Token Vectors: {q_1, q_2, ..., q_m}
                                          │
                                          ├──► MaxSim Operator: For each q_i, find max dot-product
                                          │    with any d_j, then sum across all query tokens.
                                          ▼
Document Tokens ──► [ BERT ] ──► Sequence of Token Vectors: {d_1, d_2, ..., d_k}
```

The relevance score is computed as:

$$\text{Score}_{\text{ColBERT}}(Q, D) = \sum_{i \in Q} \max_{j \in D} \left( \mathbf{E}_{q_i} \cdot \mathbf{E}_{d_j}^\top \right)$$

ColBERT retains token-level granularity while allowing document token matrices to be precomputed and compressed using vector quantization (PLAID), delivering near-cross-encoder accuracy at speeds orders of magnitude faster.

---

## Hybrid Search & Reciprocal Rank Fusion (RRF)

When combining candidates from multiple Stage 1 retrievers (e.g., BM25 keyword search + Dense Vector search) before feeding them to the cross-encoder, **Reciprocal Rank Fusion (RRF)** is the industry standard algorithm:

$$\text{RRF}(d) = \sum_{m \in \mathcal{M}} \frac{1}{k + r_m(d)}$$

where:
- $\mathcal{M}$ is the set of retrieval systems (e.g., $\{\text{BM25}, \text{Dense Vector}\}$).
- $r_m(d)$ is the rank position of document $d$ in system $m$ (1-indexed).
- $k$ is a smoothing constant (typically $k = 60$).

RRF merges rankings purely based on positional rank without requiring score normalization across fundamentally different score distributions.

---

## Practical Reranking with BGE-Reranker

```python
from sentence_transformers import CrossEncoder

# Load a production cross-encoder model
reranker = CrossEncoder("BAAI/bge-reranker-large")

query = "What is the primary cause of overfitting in deep neural networks?"
retrieved_passages = [
    "Overfitting occurs when a model learns noise and detail in training data to the extent that it negatively impacts performance on new data, often caused by excessive parameters relative to observations.",
    "Neural networks utilize backpropagation and stochastic gradient descent to iteratively update weights based on loss gradients.",
    "Underfitting happens when a model is too simple to capture the underlying trend of the data."
]

# Construct (query, doc) pairs
pairs = [[query, doc] for doc in retrieved_passages]

# Predict relevance scores (unnormalized logits or sigmoid probabilities)
scores = reranker.predict(pairs)

# Sort passages by reranker confidence
ranked_results = sorted(zip(scores, retrieved_passages), reverse=True)
for score, passage in ranked_results:
    print(f"[{score:.4f}] {passage[:80]}...")
```

---

## Key Takeaways

- Bi-encoders excel at fast candidate generation ($O(1)$ indexed lookup), while cross-encoders excel at precision scoring via all-to-all cross-attention.
- Deploying a cross-encoder on the top 50 retrieved documents dramatically reduces hallucinations in RAG systems by filtering out semantically irrelevant noise.
- Late interaction models like ColBERT provide a compelling middle ground for high-throughput enterprise search systems.
