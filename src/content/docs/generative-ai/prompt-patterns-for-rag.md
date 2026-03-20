---
title: "Prompt Patterns for RAG"
description: "Effective prompt patterns when building retrieval-augmented generation pipelines."
date: "2026-03-20"
tags: ["generative-ai", "rag", "prompts"]
---

Retrieval-Augmented Generation (RAG) combines a retrieval step with an LLM to ground outputs in external documents. Good prompt patterns make RAG systems more reliable and easier to evaluate.

## Core patterns

- **Context + Instruction:** Include the retrieved passages first, then a clear instruction: "Given the context below, answer concisely and cite the source IDs used."
- **Concise Fusion:** When many passages are returned, ask the model to synthesize only the most relevant 2–3 points.
- **Fallback Behavior:** Explicitly instruct what to do when the context doesn't contain an answer: return "No answer in provided context.".

## Formatting and citation

- Request inline citations like `(doc-123)` and a short sources list at the end.
- Limit token use by asking for summaries rather than verbatim repeats of context.

## Evaluation tips

- Test with out-of-context questions to confirm the model refuses to hallucinate.
- Measure faithfulness by comparing generated citations to the actual retrieved passages.

Practical adoption of these patterns reduces hallucinations and improves traceability for RAG applications.
