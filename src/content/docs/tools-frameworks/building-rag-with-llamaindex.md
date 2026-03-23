---
title: Building RAG with LlamaIndex
description: Practical tutorial on building Retrieval-Augmented Generation applications using LlamaIndex.
---

This guide walks through building a simple RAG (Retrieval-Augmented Generation) application using LlamaIndex to connect external data to an LLM.

## Steps

1. Ingest your documents into an index
2. Create an embedding model and embed documents
3. Configure a retriever to fetch relevant context
4. Combine retrieved context with prompts to the LLM

## Example

- Use LlamaIndex connectors to load PDFs or a database
- Build a vector index and run similarity search for queries
- Use the retrieved context to craft a prompt for final generation
