---
title: HyDE - Hypothetical Document Embeddings for Retrieval
description: Learn how HyDE improves zero-shot dense retrieval by embedding an LLM-generated hypothetical answer instead of the raw query.
---

HyDE (Hypothetical Document Embeddings) is a retrieval technique that bridges the gap between short queries and long documents. Instead of embedding the query directly, an LLM writes a fake answer, and that answer is embedded and used for the search.

## The Problem

Queries and relevant documents often look very different: a short question versus a detailed passage. Embedding models trained on matched pairs help, but in new domains the query embedding may land far from the embeddings of its answer passages.

## How HyDE Works

```text
query -> LLM writes a hypothetical passage that answers it
      -> embed the passage
      -> nearest-neighbor search over the real document index
```

The hypothetical document may contain factual errors, but it has the right *form* and vocabulary, so its embedding lies near real relevant documents. The retriever's encoder acts as a filter that discards hallucinated specifics.

## Implementation

```python
def hyde_retrieve(query, llm, embedder, index, k=5):
    prompt = f"Write a short passage that answers the question:\n{query}"
    fake_doc = llm(prompt)
    vec = embedder.encode(fake_doc)
    return index.search(vec, k)
```

Variants average the query embedding with several sampled hypothetical documents for robustness.

## When It Helps

- Zero-shot or low-resource domains without labeled retrieval data
- Short, underspecified, or conversational queries
- Specialized vocabulary where the LLM can supply domain terms

## When It Hurts

- Added latency and cost from an extra LLM call per query
- If the LLM knows nothing about the topic, the fake passage may drift toward the wrong region
- Queries about private or very recent facts the model cannot guess

## Practical Advice

Combine HyDE with lexical search and reranking, and evaluate on your own queries; gains are largest where the baseline embedding model is weak.
