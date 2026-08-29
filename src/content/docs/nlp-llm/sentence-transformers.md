---
title: Sentence Transformers and Dense Embeddings
description: Learn how Sentence-BERT (SBERT) and bi-encoder architectures generate high-quality dense vector representations for semantic search, clustering, and retrieval-augmented generation.
---

When BERT was released in 2018, it set new benchmarks across natural language processing tasks. However, finding the most semantically similar sentence pair within a collection of $10{,}000$ sentences required feeding all possible combinations into BERT—amounting to approximately $50$ million inference forward passes, taking over **65 hours of computation**.

**Sentence-BERT (SBERT)**, introduced by Reimers and Gurevych in 2019, revolutionized this workflow. By adopting a **siamese / bi-encoder network architecture**, SBERT maps variable-length sentences into fixed-dimensional dense vector spaces where semantic similarity corresponds directly to cosine distance. The same $10{,}000$-sentence search was reduced from 65 hours to **under 5 seconds**.

---

## Bi-Encoders vs. Cross-Encoders

```
Cross-Encoder (High Accuracy, Prohibitive Latency):
[ Query ] + [ Document ] ──► [ Joint Transformer Self-Attention ] ──► Relevance Score (0 to 1)
Cost: Must evaluate every query-document pair online. Cannot precompute or index.

Bi-Encoder / Sentence Transformer (Ultra-Fast Search, Scalable):
[ Document ] ──► [ Document Encoder ] ──► Document Vector d ──► Pre-indexed in Vector DB
                                                                      ▲
                                                                      │ Cosine Similarity:
                                                                      │ cos(q, d) = (q · d) / (|q||d|)
                                                                      ▼
[ Query ]    ──► [ Query Encoder ]    ──► Query Vector q    ──────────┘
```

| Dimension | Cross-Encoder | Bi-Encoder (Sentence Transformer) |
| :--- | :--- | :--- |
| **Input Structure** | Pair concatenated: `[CLS] Q [SEP] D [SEP]` | Query and Document encoded independently |
| **Attention** | Cross-attention across all words in both texts | Self-attention isolated within each text |
| **Computational Complexity** | $O(N \cdot M)$ full transformer passes | $O(N + M)$ passes + vector similarity lookup |
| **Indexability** | Cannot be pre-indexed | Embeddings can be precomputed and stored in vector DBs |
| **Use Case** | Re-ranking top-50 candidate passages | Candidate retrieval across millions of documents |

---

## Architecture of a Sentence Transformer

A Sentence Transformer passes a sequence through a transformer backbone (such as BERT, RoBERTa, or ModernBERT) and aggregates the resulting token embeddings into a single fixed-length vector through a **pooling layer**:

```
Tokenized Sentence:  "Artificial Intelligence is transforming search"
                              │
                              ▼
                   [ Transformer Encoder ]
                              │
                    Token Hidden States: [h_1, h_2, ..., h_n]
                              │
                              ▼
                     [ Pooling Layer ]
                              │
                              ▼
                    Dense Sentence Vector u ∈ R^768
```

### Pooling Strategies
Given token output representations $\mathbf{h}_1, \dots, \mathbf{h}_n \in \mathbb{R}^d$:
1. **Mean Pooling (Default & Recommended):** Computes the average of all contextual token embeddings (accounting for attention masks to ignore padding):
   $$\mathbf{u} = \frac{\sum_{i=1}^n m_i \mathbf{h}_i}{\sum_{i=1}^n m_i}$$
2. **`[CLS]`-Token Pooling:** Uses the hidden state of the first special token $\mathbf{h}_{\text{[CLS]}}$. While natural in standard BERT classification, it often yields inferior sentence representations without specialized contrastive pretraining.
3. **Max Pooling:** Takes the element-wise maximum across all token representations along each hidden dimension.

---

## Training Objectives and Loss Functions

Sentence transformers are trained using contrastive objectives that pull semantically related pairs together while pushing unrelated pairs apart.

### 1. Multiple Negatives Ranking Loss (MNRL)
MNRL is the gold standard loss function for training modern dense retrieval models (e.g., BGE, E5). Given a mini-batch of $B$ positive pairs $(q_i, d_i^+)$, the remaining $B - 1$ documents in the mini-batch act as **in-batch negative samples** for query $i$:

$$\mathcal{L}_{\text{MNRL}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\left(\tau \cdot \cos(\mathbf{q}_i, \mathbf{d}_i^+)\right)}{\sum_{j=1}^B \exp\left(\tau \cdot \cos(\mathbf{q}_i, \mathbf{d}_j^+)\right)}$$

where $\tau$ is a temperature scaling factor. This allows training with hundreds of negatives per step without additional encoder passes.

### 2. Triplet Loss
Given an anchor text $a$, a positive text $p$, and a negative text $n$:

$$\mathcal{L}_{\text{triplet}} = \max\left(0, \; \|\mathbf{u}_a - \mathbf{v}_p\|_2 - \|\mathbf{u}_a - \mathbf{v}_n\|_2 + \epsilon\right)$$

where $\epsilon$ is a safety margin enforcing that the anchor is closer to the positive example than to the negative example by at least $\epsilon$.

---

## Hard Negatives and Mine-and-Refine Pipelines

Using purely random in-batch negatives causes saturation: random negatives are easily distinguished from positive matches. To build state-of-the-art retrieval models, training pipelines incorporate **Hard Negatives**:

1. **Initial BM25 / Bi-Encoder Mining:** For each query, retrieve the top 100 documents using lexical search (BM25) or a baseline vector model.
2. **Cross-Encoder Filtering:** A powerful cross-encoder scores the retrieved documents. Any document that ranks high according to BM25 but receives a low relevance score from the cross-encoder is a **Hard Negative** (lexically similar, but semantically irrelevant).
3. **Bi-Encoder Fine-Tuning:** The sentence transformer is trained using triplets containing these mined hard negatives, forcing it to learn fine-grained conceptual distinctions.

---

## Practical Implementation with `sentence-transformers`

```python
from sentence_transformers import SentenceTransformer, util

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Sentences to encode
corpus = [
    "A neural network is a computational model inspired by biological brains.",
    "Convolutional layers process visual imagery through local spatial kernels.",
    "Baking sourdough bread requires flour, water, salt, and wild yeast."
]
query = "How do artificial neural nets work?"

# Compute dense embeddings with normalized unit lengths
corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
query_embedding = model.encode(query, normalize_embeddings=True)

# Compute cosine similarities via dot product
scores = util.dot_score(query_embedding, corpus_embeddings)[0]

# Output most similar match
for score, doc in zip(scores, corpus):
    print(f"Score: {score:.4f} | Document: {doc}")
```

---

## Key Takeaways

- Sentence Transformers decouple query and document encoding, enabling pre-indexing of millions of vectors and sub-millisecond retrieval.
- Mean pooling across contextualized token states consistently outperforms naive `[CLS]` token extraction for semantic representations.
- Modern high-performance embeddings (BGE, E5, GTE) rely on Multiple Negatives Ranking Loss combined with hard negative mining to achieve high precision in RAG pipelines.
