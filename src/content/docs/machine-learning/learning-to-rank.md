---
title: Learning to Rank - Pointwise, Pairwise, and Listwise Methods
description: Learn how learning-to-rank models order items for search and recommendation, comparing pointwise, pairwise, and listwise losses and ranking metrics like NDCG.
---

Learning to rank (LTR) trains models to order a list of items, such as documents for a query, products in a feed, or candidates for a recommender, so that the most relevant items appear first. What matters is the ordering, not the absolute score.

## Training Data

Each training instance is a *query* with candidate items, each having a feature vector and a relevance label (for example 0 to 4), derived from human judgments or click data.

## Three Families

### Pointwise
Treat each item independently and predict its relevance with regression or classification. Simple, but ignores that ranking is relative within a list.

### Pairwise
Learn which of two items should be ranked higher. RankNet uses a logistic loss on score differences:

```text
P(i > j) = sigmoid(s_i - s_j)
loss = -log P(i > j)   for pairs where label_i > label_j
```

### Listwise
Optimize a loss defined over the whole list. LambdaRank and LambdaMART weight pair gradients by the change in NDCG that swapping the pair would cause, directly focusing on the top of the ranking.

## Ranking Metrics

- **Precision@k / Recall@k**
- **MRR**: reciprocal rank of the first relevant item
- **NDCG@k**: discounted cumulative gain normalized by the ideal ordering

```text
DCG@k = sum over i=1..k of (2^rel_i - 1) / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

## Practical Example with LightGBM

```python
import lightgbm as lgb

ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05)
ranker.fit(
    X_train, y_train,
    group=train_group_sizes,          # number of items per query
    eval_set=[(X_val, y_val)], eval_group=[val_group_sizes],
    eval_at=[5, 10],
)
scores = ranker.predict(X_test)
```

## Biases in Click Data

Clicks are biased by position (users click top results more). Techniques such as inverse propensity weighting and unbiased LTR correct for this; otherwise the model learns to reproduce the existing ranking.

## Typical System Design

A fast candidate-retrieval stage (BM25 or embedding search) returns hundreds of items, then a learned ranker rescores them, sometimes followed by a neural cross-encoder on the top few. LLM-based rerankers now serve as the final stage in many search pipelines.
