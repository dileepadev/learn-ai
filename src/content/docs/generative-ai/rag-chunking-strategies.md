---
title: RAG Chunking Strategies
description: A practical guide to chunking for Retrieval-Augmented Generation, including chunk size, overlap, semantic segmentation, and evaluation tradeoffs.
---

**Chunking** is one of the highest-leverage decisions in RAG systems. Before retrieval ranking, reranking, or generation quality can help, the corpus is already constrained by how documents were split.

## What Chunking Controls

Chunking determines:

- What information can be retrieved together.
- How much context each embedding represents.
- Whether key facts are fragmented across chunks.
- Retrieval recall/precision tradeoffs.

Bad chunking often looks like retrieval failure, even when the retriever is working as designed.

## Common Chunking Methods

### Fixed-Length Chunking

Split by token count (for example 300-800 tokens) with overlap.

**Pros:** simple, fast, predictable indexing.  
**Cons:** can split logical units in the middle of definitions, tables, or code blocks.

### Structure-Aware Chunking

Split by headings, paragraphs, lists, and table boundaries.

**Pros:** preserves meaning and improves citation quality.  
**Cons:** produces variable chunk sizes; can exceed optimal embedding window length.

### Semantic Chunking

Use embedding similarity shifts to detect topic boundaries.

**Pros:** better topical coherence than fixed rules.  
**Cons:** higher indexing cost and more tuning complexity.

### Hybrid Chunking

Use document structure first, then enforce token bounds with smart fallbacks.

This is often the best production default.

## Chunk Size and Overlap

There is no universal best size, but practical starting points are:

- 300-500 tokens for dense factual documentation.
- 600-900 tokens for narrative or conceptual text.
- 50-150 tokens for FAQ or short answer corpora.

Overlap (10-20%) helps preserve cross-boundary meaning, but excessive overlap increases storage and retrieval duplicates.

## Content-Type-Specific Strategies

### Code Repositories

- Preserve function/class boundaries.
- Keep docstrings and signatures in the same chunk.
- Add file path and symbol metadata.

### Policies and Legal Text

- Split at section/subsection levels.
- Keep references and definitions attached to governing clauses.
- Store version/date metadata for compliance traceability.

### Product Docs

- Keep steps of a procedure together.
- Avoid splitting prerequisite sections from action steps.

## Metadata Is Part of Chunking

A chunk is more than text. Attach metadata such as:

- document ID and title
- section path
- timestamp/version
- access control labels
- domain tags

Strong metadata supports filtered retrieval and safer grounding.

## How to Evaluate Chunking

Run chunking A/B tests with fixed retriever and generator:

- Recall@k on benchmark questions
- Context precision (percentage of retrieved chunks actually used)
- Answer groundedness/citation correctness
- Latency and vector store size impact

If answer quality changes significantly while models stay constant, chunking is likely the root cause.

## Frequent Mistakes

- Using one chunk policy for all document types.
- Ignoring tables and code blocks in split logic.
- High overlap that floods top-k with near duplicates.
- Reindexing without version controls, creating mixed old/new chunks.

## Summary

RAG chunking is an information architecture decision, not a preprocessing footnote. The best strategy balances semantic coherence, retrieval efficiency, and maintainability, usually through a hybrid structure-aware pipeline with measured token boundaries.

