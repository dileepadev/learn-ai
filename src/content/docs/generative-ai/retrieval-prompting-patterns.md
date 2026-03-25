---
title: Introduction to Retrieval Prompting Patterns
description: Learn common strategies for integrating retrieved information effectively into LLM prompts.
---

Retrieval Prompting Patterns are standardized ways of formatting how retrieved knowledge is presented to a large language model (LLM) within a Retrieval-Augmented Generation (RAG) system.

## Popular Prompting Patterns

1. **Context-First:** The retrieved information is placed at the top of the prompt to provide the model with "grounding" before the instruction.
2. **Context-Instruction-Query:** A three-part structure used to clearly separate knowledge from the user's specific request.
3. **Few-Shot RAG:** Providing examples that include both the retrieved background and a high-quality demonstration answer.

## Key Principles

- **Clarity:** Use delimiters (e.g., `### Context ###`) to help the model distinguish between your instructions and the retrieved data.
- **Relatability:** Explicitly tell the model "Use ONLY the provided context" to reduce hallucinations.
- **Relevance:** Only include the most relevant chunks to avoid distracting the model with unrelated information ("Lost-in-the-Middle" problem).

## Common Prompt Template example

```text
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know.

Context:
{context}

Question: {question}
Helpful Answer:
```
