---
title: Dense Passage Retrieval - Dual-Encoder Retrieval Training
description: Learn how dense passage retrieval trains query and passage encoders with in-batch negatives and hard negatives to power semantic search and open-domain question answering.
---

Dense Passage Retrieval (DPR) uses two neural encoders to map questions and passages into the same vector space, so relevance becomes vector similarity. It underpins semantic search and the retrieval stage of many question answering and RAG systems.

## Architecture

```text
question  -> question encoder -> q vector
passage   -> passage encoder  -> p vector
relevance  = dot_product(q, p)
```

Passage vectors are computed offline and stored in an approximate nearest neighbor index (such as FAISS or HNSW), so query-time work is one encoder pass plus a vector search.

## Training Objective

Given a question, one positive passage, and several negatives, minimize the negative log-likelihood of the positive using a softmax over similarities:

```text
L = -log( exp(sim(q, p+)) / ( exp(sim(q, p+)) + sum_j exp(sim(q, p-_j)) ) )
```

## Negatives Matter

- **In-batch negatives**: other questions' positives in the same batch serve as negatives; large batches are cheap and effective
- **BM25 hard negatives**: passages that match keywords but do not answer the question, which teach finer distinctions
- **Mined hard negatives**: retrieved by an earlier version of the model itself, but risk false negatives when unlabeled passages are actually relevant

## Training Sketch

```python
import torch
import torch.nn.functional as F

def in_batch_loss(q_emb, p_emb, temperature=0.05):
    # q_emb, p_emb: (B, D); the positive for q[i] is p[i]
    q = F.normalize(q_emb, dim=-1)
    p = F.normalize(p_emb, dim=-1)
    logits = q @ p.T / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)
```

## Strengths and Limits

Dense retrieval captures paraphrase and semantic similarity that lexical methods miss. It can underperform BM25 on rare entities, exact identifiers, and out-of-domain text, and a single vector may not capture all aspects of a long passage. Hybrid retrieval and cross-encoder reranking address both gaps.

## Modern Descendants

Contrastively pretrained general-purpose embedding models, multi-vector approaches like ColBERT, and instruction-aware embedders extend the same dual-encoder idea with much larger and cleaner training data.
