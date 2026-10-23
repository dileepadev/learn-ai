---
title: BM25 - Probabilistic Lexical Retrieval
description: Learn how BM25 scores documents using term frequency saturation, inverse document frequency, and length normalization, and why it remains a strong baseline next to dense retrieval.
---

BM25 (Best Matching 25) is the dominant lexical ranking function in search engines. Despite decades of age, it is a hard baseline to beat on many retrieval benchmarks and is routinely combined with embeddings in hybrid search.

## The Scoring Function

For query Q with terms `q_i` and document D:

```text
score(D, Q) = sum over q_i of  IDF(q_i) * ( f(q_i, D) * (k1 + 1) ) /
              ( f(q_i, D) + k1 * (1 - b + b * |D| / avgdl) )
```

- `f(q_i, D)`: term frequency in the document
- `|D|`, `avgdl`: document length and average document length
- `k1`: controls term-frequency saturation (typically 1.2 to 2.0)
- `b`: controls length normalization (typically 0.75)

## Inverse Document Frequency

```text
IDF(q) = log( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )
```

Rare terms receive more weight than common ones.

## What Makes It Better than TF-IDF

- **Saturation**: the tenth occurrence of a word adds much less than the first, preventing keyword stuffing from dominating
- **Length normalization**: long documents are penalized in a tunable way instead of raw counts

## Example

```python
from rank_bm25 import BM25Okapi

corpus = ["the cat sat on the mat", "dogs chase cats", "stock markets fell today"]
tokenized = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

scores = bm25.get_scores("cat on mat".split())
```

## Strengths and Weaknesses

| Strength | Weakness |
|----------|----------|
| Exact keyword, ID, and rare-term matching | No synonym or paraphrase understanding |
| No training required, fast, interpretable | Sensitive to tokenization and stemming |
| Works well out of domain | Vocabulary mismatch between query and document |

## Hybrid Search

Production RAG systems often run BM25 and vector search in parallel and merge results with reciprocal rank fusion:

```text
RRF(d) = sum over rankers of 1 / (k + rank(d))      # k typically 60
```

Sparse neural variants such as SPLADE keep the inverted-index efficiency while learning term expansion.
