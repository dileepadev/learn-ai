---
title: "Introduction to Multi-Modal RAG"
description: "Beyond text: How to build RAG pipelines that retrieve and reason over images, tables, and documents simultaneously."
---

Most RAG systems focus exclusively on text. However, real-world data is often multi-modal, containing images, charts, and complex formatting. **Multi-Modal RAG** expands the retrieval-augmented generation pattern to handle these diverse data types.

## How Multi-Modal RAG Works

The process involves two main components:

1. **Multi-Modal Embeddings**: Models like CLIP or Contrastive Language-Image Pretraining that can represent both text and images in the same vector space.

2. **Multi-Modal LLMs**: Models (like GPT-4o or Claude 3.5 Sonnet) that can process both text and visual inputs as part of their prompt.

## The Pipeline

1. **Ingestion**: Documents are parsed into text chunks, images are extracted, and charts are often converted into text summaries or kept as images.
2. **Retrieval**: Based on a user query, the system retrieves relevant text AND relevant visual elements (images/charts).
3. **Generation**: The Multi-Modal LLM receives the text query, retrieved text context, and retrieved images to formulate a comprehensive answer.

## Key Challenges

- **Alignment**: Ensuring that text and images are correctly related in the vector space.
- **Context Length**: Visual data consumed as tokens can quickly fill up an LLM's context window.
- **Parsing**: Accurately extracting data from non-selectable text within images or complex PDF layouts.
