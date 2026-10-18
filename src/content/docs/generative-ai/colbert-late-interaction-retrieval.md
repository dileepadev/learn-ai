---
title: ColBERT and Late Interaction Retrieval
description: Learn how ColBERT keeps one embedding per token and scores queries with MaxSim, trading index size for much finer-grained matching than single-vector retrieval.
---

ColBERT (Contextualized Late Interaction over BERT) is a retrieval architecture that sits between fast single-vector bi-encoders and accurate but slow cross-encoders.

## Three Retrieval Styles

| Approach | Representation | Interaction | Cost |
|----------|----------------|-------------|------|
| Bi-encoder | 1 vector per text | Dot product | Cheapest |
| Cross-encoder | None (joint encoding) | Full attention | Very expensive, rerank only |
| Late interaction | 1 vector per token | MaxSim | Middle |

## How ColBERT Works

Queries and documents are encoded **independently** into matrices of token embeddings. Document embeddings can therefore be precomputed and indexed. At search time, relevance is computed by late interaction:

```text
score(q, d) = sum over query tokens i of  max over doc tokens j of  (q_i . d_j)
```

Each query token finds its best-matching document token (MaxSim), and the maxima are summed. This lets individual query terms match specific parts of a document, which captures fine-grained relevance that a single pooled vector blurs.

## Implementation Sketch

```python
import torch

def maxsim(q_emb, d_emb):
    # q_emb: (Lq, D)   d_emb: (Ld, D), both L2-normalized
    sim = q_emb @ d_emb.T            # (Lq, Ld)
    return sim.max(dim=1).values.sum()
```

## Making It Practical

Storing a vector per token is large. ColBERTv2 reduces the footprint with residual compression (clustering token embeddings and storing quantized residuals) and uses centroid-based candidate generation (as in PLAID) to avoid scoring every document.

## Strengths

- Strong out-of-domain accuracy
- Interpretable: you can see which tokens matched
- Multi-vector text and image variants (such as ColPali for document pages) extend the idea

## Costs

Index size and engineering complexity exceed single-vector search. Many teams use ColBERT-style models as a high-quality reranking stage over candidates from a cheaper first-stage retriever.
