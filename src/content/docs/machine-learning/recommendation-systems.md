---
title: Recommendation Systems
description: Build modern recommendation systems — from collaborative filtering and matrix factorization to two-tower neural models, session-based transformers, and real-time serving — covering implicit feedback, approximate nearest neighbor retrieval, feature crosses, and multi-objective ranking for commercial recommenders.
---

Recommendation systems are the most economically impactful deployed machine learning systems in existence. Netflix, YouTube, Spotify, Amazon, and TikTok attribute substantial revenue to their recommendation engines. Modern recommenders are complex multi-stage systems combining collaborative filtering, content-based features, and deep learning — built on top of subtle mathematical foundations that determine what billions of users see every day.

## The Recommendation Problem

Given a set of users $\mathcal{U}$, items $\mathcal{I}$, and observed interactions $\mathcal{O} \subseteq \mathcal{U} \times \mathcal{I}$, predict the score or probability of each user-item pair $(u, i) \notin \mathcal{O}$.

**Explicit feedback**: ratings (1–5 stars), thumbs up/down. Sparse and noisy.

**Implicit feedback**: clicks, watches, purchases, dwell time. Dense but noisy — a watched video is not necessarily enjoyed.

## Matrix Factorization

Matrix factorization (MF) decomposes the user-item interaction matrix $\mathbf{R} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}$ into user and item embedding matrices:

$$\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i + b_u + b_i + \mu$$

where $\mathbf{p}_u \in \mathbb{R}^d$ is the user embedding, $\mathbf{q}_i \in \mathbb{R}^d$ is the item embedding, $b_u, b_i$ are user/item biases, and $\mu$ is the global mean.

### Weighted Matrix Factorization for Implicit Feedback

For implicit feedback, we observe binary preference ($r_{ui} = 1$ if interacted, 0 otherwise) with confidence weights (e.g., log of interaction count):

$$c_{ui} = 1 + \alpha \cdot n_{ui}$$

$$\mathcal{L}_{\mathrm{WMF}} = \sum_{u,i} c_{ui}(r_{ui} - \mathbf{p}_u^T \mathbf{q}_i)^2 + \lambda(\|\mathbf{P}\|_F^2 + \|\mathbf{Q}\|_F^2)$$

```python
import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        p = self.user_emb(user_ids)       # (B, D)
        q = self.item_emb(item_ids)       # (B, D)
        b_u = self.user_bias(user_ids)    # (B, 1)
        b_i = self.item_bias(item_ids)    # (B, 1)
        dot = (p * q).sum(dim=1, keepdim=True)
        return (dot + b_u + b_i + self.global_bias).squeeze(1)
```

## Bayesian Personalized Ranking (BPR)

For implicit feedback, optimizing pairwise rankings directly outperforms pointwise MSE. BPR (Rendle et al., 2009) maximizes the probability that interacted items rank above unobserved items:

$$\mathcal{L}_{\mathrm{BPR}} = -\sum_{(u, i, j) \in \mathcal{D}_S} \log \sigma(\hat{r}_{ui} - \hat{r}_{uj})$$

where $(u, i, j)$ means user $u$ interacted with item $i$ but not $j$. Negative items $j$ are sampled from unobserved interactions.

```python
def bpr_loss(model, user_ids, pos_item_ids, neg_item_ids):
    pos_scores = model(user_ids, pos_item_ids)
    neg_scores = model(user_ids, neg_item_ids)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
```

## Two-Tower Neural Models

Industrial recommenders use **two-tower** (dual-encoder) models: a user tower and an item tower, each encoding their respective features into a shared embedding space. The score is the dot product of the two embeddings:

$$\text{score}(u, i) = f_\theta(\mathbf{x}_u) \cdot g_\phi(\mathbf{x}_i)$$

```python
class TwoTowerModel(nn.Module):
    def __init__(self, user_feature_dim: int, item_feature_dim: int, embedding_dim: int = 256):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, embedding_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, user_features, item_features):
        u_emb = self.user_tower(user_features)           # (B, D)
        i_emb = self.item_tower(item_features)           # (B, D)
        # L2 normalize for cosine similarity
        u_emb = torch.nn.functional.normalize(u_emb, dim=-1)
        i_emb = torch.nn.functional.normalize(i_emb, dim=-1)
        return (u_emb * i_emb).sum(dim=-1)              # dot product

    def get_user_embedding(self, user_features):
        return torch.nn.functional.normalize(self.user_tower(user_features), dim=-1)

    def get_item_embedding(self, item_features):
        return torch.nn.functional.normalize(self.item_tower(item_features), dim=-1)
```

**In-batch negatives**: during training, use all other items in the batch as negatives — efficient and effective when batch size is large:

```python
def in_batch_softmax_loss(user_embs, item_embs, temperature: float = 0.05):
    """
    Contrastive loss treating all other items in batch as negatives.
    Diagonal entries are the positive pairs.
    """
    logits = (user_embs @ item_embs.T) / temperature   # (B, B)
    labels = torch.arange(len(user_embs), device=user_embs.device)
    return torch.nn.functional.cross_entropy(logits, labels)
```

## Approximate Nearest Neighbor Retrieval

At inference time, the user tower computes one embedding and the system must find the top-K most similar items among millions. Exact search is $O(|\mathcal{I}| \cdot d)$ — too slow for real-time serving. **ANN (Approximate Nearest Neighbor)** indexes trade a small accuracy loss for orders-of-magnitude speedup:

- **FAISS** (Facebook): hierarchical navigable small world (HNSW) graphs and inverted file (IVF) indexes — the standard for production retrieval
- **ScaNN** (Google): anisotropic quantization optimized for maximum inner product search

```python
import faiss
import numpy as np

# Build index offline (after computing item embeddings)
embedding_dim = 256
item_embeddings = np.random.randn(1_000_000, embedding_dim).astype(np.float32)

index = faiss.IndexHNSWFlat(embedding_dim, 32)   # HNSW with M=32 neighbors
index = faiss.IndexFlatIP(embedding_dim)          # Exact inner product (for small catalogs)
index.add(item_embeddings)

# Online retrieval
user_embedding = np.random.randn(1, embedding_dim).astype(np.float32)
scores, item_indices = index.search(user_embedding, k=100)
```

## Deep Structured Semantic Models and Feature Crosses

Beyond dot products, **deep cross networks** (DCN-v2, DeepFM) explicitly model feature interactions for ranking:

```python
class DeepCrossNetwork(nn.Module):
    """DCN-v2: Cross layers + Deep layers for click-through rate prediction."""

    def __init__(self, input_dim: int, num_cross_layers: int = 3, deep_dims=(256, 128, 64)):
        super().__init__()
        # Cross layers: learn explicit feature interactions
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim)) for _ in range(num_cross_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_cross_layers)
        ])
        # Deep tower: implicit high-order interactions
        layers = []
        dims = [input_dim] + list(deep_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        self.deep = nn.Sequential(*layers)
        # Final prediction
        self.output = nn.Linear(input_dim + deep_dims[-1], 1)

    def forward(self, x):
        # Cross network
        x_cross = x
        for W, b in zip(self.cross_weights, self.cross_biases):
            x_cross = x @ (W * x_cross).T + b + x_cross  # simplified cross layer

        # Deep network
        x_deep = self.deep(x)

        return torch.sigmoid(self.output(torch.cat([x_cross, x_deep], dim=-1)))
```

## Session-Based Recommendation with Transformers

When user history is sparse or cold-start, session-based models recommend based on the current session's sequence of interactions. **BERT4Rec** and **SASRec** apply masked/causal self-attention to item sequences:

```python
from transformers import BertConfig, BertModel


class BERT4Rec(nn.Module):
    def __init__(self, num_items: int, max_seq_len: int = 50, d_model: int = 256):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        config = BertConfig(
            hidden_size=d_model, num_attention_heads=4,
            num_hidden_layers=2, intermediate_size=1024,
            max_position_embeddings=max_seq_len,
        )
        self.bert = BertModel(config)
        self.head = nn.Linear(d_model, num_items)

    def forward(self, item_seq, masked_positions=None):
        embs = self.item_emb(item_seq)
        outputs = self.bert(inputs_embeds=embs)
        logits = self.head(outputs.last_hidden_state)
        return logits
```

## Multi-Objective Ranking

Commercial recommenders optimize multiple objectives simultaneously (clicks, watch time, shares, revenue) using weighted multi-task heads or Pareto-constrained optimization. YouTube's ranking system (Zhao et al., 2019) uses a multi-gate mixture-of-experts architecture where different expert networks specialize in different objectives.

A simple multi-objective loss:

```python
def multi_objective_loss(
    click_logit, watch_time_logit, share_logit,
    click_label, watch_time_label, share_label,
    weights=(1.0, 0.5, 0.3),
):
    click_loss = torch.nn.functional.binary_cross_entropy_with_logits(click_logit, click_label)
    watch_loss = torch.nn.functional.mse_loss(watch_time_logit, watch_time_label)
    share_loss = torch.nn.functional.binary_cross_entropy_with_logits(share_logit, share_label)
    return weights[0] * click_loss + weights[1] * watch_loss + weights[2] * share_loss
```

## Industrial System Architecture

A production recommender has multiple stages:

```text
User Request
     ↓
Candidate Generation (millions → thousands)
  Two-tower ANN retrieval
     ↓
Filtering (business rules, already-seen, safety)
     ↓
Ranking (thousands → tens)
  Deep cross network / transformer ranker
     ↓
Post-ranking / Re-ranking (diversity, freshness)
     ↓
Final Recommendations Served
```

## Summary

Modern recommendation systems combine principled mathematical foundations with industrial-scale engineering:

- **Matrix factorization** with BPR loss provides a strong baseline for collaborative filtering on implicit feedback
- **Two-tower models** with in-batch negatives scale to millions of items through ANN retrieval at serving time
- **Deep cross networks** explicitly model feature interactions for click-through rate and conversion prediction
- **BERT4Rec and SASRec** use transformer self-attention over item sequences for session-based cold-start recommendations
- **Multi-objective ranking** balances competing signals (engagement, quality, revenue) through multi-task learning
- Production systems are multi-stage pipelines: generation → filtering → ranking → re-ranking, each using models of increasing complexity and smaller candidate sets
