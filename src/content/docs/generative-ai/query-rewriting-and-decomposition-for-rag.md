---
title: Query Rewriting and Decomposition for RAG
description: Learn how rewriting, expanding, and decomposing user queries before retrieval improves recall and answer quality in retrieval-augmented generation systems.
---

User questions are often poorly suited to direct retrieval: they may be ambiguous, depend on earlier conversation turns, or bundle several information needs. Transforming the query before search is one of the highest-leverage improvements to a RAG system.

## Common Query Problems

- **Context dependence**: "What about its pricing?" has no standalone meaning
- **Vague wording**: the user's phrasing differs from the vocabulary in documents
- **Multi-part questions**: "Compare A and B on cost and latency" needs several lookups
- **Over-specific queries**: too many details harm lexical matching

## Techniques

### 1. Contextual Rewriting
Use the chat history to produce a standalone question.

```text
History: "Tell me about the Pro plan."  ->  Follow-up: "What about its limits?"
Rewritten: "What are the usage limits of the Pro plan?"
```

### 2. Query Expansion / Multi-Query
Generate several paraphrases or related queries, retrieve for each, and merge results (for example with reciprocal rank fusion).

### 3. Decomposition
Split a complex question into sub-questions answered separately, then synthesize:

```text
"Which is cheaper for 1M requests, provider A or B, and which has lower latency?"
 -> "What is the price of provider A per 1M requests?"
 -> "What is the price of provider B per 1M requests?"
 -> "What is the latency of provider A?"
 -> "What is the latency of provider B?"
```

### 4. Step-Back Prompting
Ask a more general question first to retrieve background principles, then answer the specific one.

### 5. HyDE
Generate a hypothetical answer and embed it for search.

## Implementation Sketch

```python
def rag_answer(question, history, llm, retriever):
    standalone = llm(f"Rewrite as a standalone question:\n{history}\n{question}")
    subqs = llm(f"List sub-questions needed to answer, one per line:\n{standalone}").splitlines()
    docs = {d.id: d for q in subqs for d in retriever(q, k=4)}.values()
    return llm(f"Answer using the context.\nContext:\n{list(docs)}\nQuestion: {standalone}")
```

## Trade-offs

Each transformation adds latency and cost and can introduce drift if the rewrite changes intent. Use a small, fast model for rewriting, log rewritten queries, and evaluate retrieval recall with and without each step.
