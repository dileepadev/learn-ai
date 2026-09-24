---
title: Introduction to the Cohere API
description: Learn how Cohere's API provides generation, embedding, reranking, and RAG-focused endpoints, and where it fits among enterprise LLM providers.
---

Cohere is an LLM provider focused heavily on enterprise and retrieval-augmented use cases, offering generation, embedding, classification, and reranking as distinct first-class API endpoints rather than treating retrieval as an afterthought bolted onto a chat API.

## Core Endpoints

```python
import cohere

co = cohere.Client("COHERE_API_KEY")

response = co.chat(
    model="command-r-plus",
    message="Summarize the key risks in this contract.",
    documents=[{"title": "Contract", "text": contract_text}]
)

embeddings = co.embed(
    texts=["a support ticket about a billing issue"],
    model="embed-english-v3.0",
    input_type="search_document"
)

reranked = co.rerank(
    query="billing dispute",
    documents=candidate_documents,
    model="rerank-english-v3.0"
)
```

The `chat` endpoint accepts a `documents` parameter directly, letting the model ground its response in supplied source text and return citations tied to specific documents — a RAG-oriented pattern built into the API rather than something you assemble yourself from a generic completion endpoint.

## Embeddings with Input Type

Cohere's embedding models require specifying an `input_type` (`search_document`, `search_query`, `classification`, `clustering`), which changes how the model encodes the same text depending on its intended use. Search queries and the documents they'll be matched against are embedded asymmetrically, since a short question and a long passage answering it are structurally different kinds of text, and Cohere's models are trained to account for that asymmetry directly.

## Rerank as a Dedicated Endpoint

Rather than relying solely on embedding similarity for retrieval ranking, Cohere's Rerank endpoint takes a query and a list of candidate documents already retrieved by a first-pass search and reorders them using a cross-encoder-style model that scores each query-document pair jointly, typically improving top-result relevance over embedding similarity alone.

## Practical Guidance

Reach for Cohere specifically when your application is retrieval-heavy — built-in document grounding with citations and a dedicated reranker reduce the custom RAG plumbing you'd otherwise build yourself. Benchmark Cohere's generation quality against other providers for your specific task, since strengths vary: Cohere's differentiators are retrieval-oriented tooling and multilingual embedding support, not necessarily leading raw generation benchmarks across every task type.
