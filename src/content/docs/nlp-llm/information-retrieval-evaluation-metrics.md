---
title: Information Retrieval Evaluation Metrics
description: Learn how to evaluate search and retrieval systems with Recall@k, MRR, MAP, and NDCG, and how to build relevance judgments for your own corpus.
---

Evaluating retrieval means measuring how well a system surfaces relevant documents near the top of a ranked list. The right metric depends on whether you care about finding any relevant item, all of them, or good ordering.

## Needed: Relevance Judgments

An evaluation set contains queries and, for each, labels indicating which documents are relevant (binary or graded). Labels come from human annotators, click logs, or LLM judges validated against humans.

## Common Metrics

### Precision@k and Recall@k
```text
Precision@k = (relevant items in top k) / k
Recall@k    = (relevant items in top k) / (all relevant items)
```

Recall@k is the key metric for the first stage of a RAG pipeline, because missing documents cannot be recovered later.

### Mean Reciprocal Rank (MRR)
```text
RR  = 1 / rank of first relevant item
MRR = mean of RR over queries
```
Good when one correct answer is enough.

### Mean Average Precision (MAP)
Average the precision at each rank where a relevant item appears, then average across queries. Rewards placing all relevant documents high.

### NDCG@k
Handles graded relevance and position discount:
```text
DCG@k  = sum over i=1..k of (2^rel_i - 1) / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

## Implementation

```python
import math

def ndcg_at_k(rels, k):
    def dcg(r): return sum((2**x - 1) / math.log2(i + 2) for i, x in enumerate(r[:k]))
    ideal = dcg(sorted(rels, reverse=True))
    return dcg(rels) / ideal if ideal > 0 else 0.0

def recall_at_k(retrieved_ids, relevant_ids, k):
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)
```

## Choosing a Metric

| Goal | Metric |
|------|--------|
| First-stage recall for RAG | Recall@k |
| Single correct answer | MRR |
| Graded relevance, ranked list quality | NDCG@k |
| Overall ranking of all relevant docs | MAP |

## Practical Cautions

- Unjudged documents are usually treated as non-relevant, which penalizes systems that find valid but unlabeled results
- Report results with confidence intervals; small query sets are noisy
- Retrieval metrics do not guarantee end-to-end answer quality, so also evaluate generated answers for faithfulness and correctness
- Public benchmarks (such as BEIR) are useful, but build a test set from your own queries
