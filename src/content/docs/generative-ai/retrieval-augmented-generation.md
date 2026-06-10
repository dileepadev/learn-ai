---
title: Retrieval-Augmented Generation
description: Enhancing large language models with external knowledge retrieval — combining LLMs with information retrieval to reduce hallucinations and improve factual accuracy.
---

**Retrieval-Augmented Generation (RAG)** is a technique that enhances large language models (LLMs) by augmenting them with an external knowledge base. Rather than relying solely on weights learned during training, RAG retrieves relevant documents or facts from a knowledge store at inference time, then uses them to ground the model's response. This approach addresses two critical LLM limitations: hallucination (generating plausible but false information) and knowledge cutoff (inability to reference information beyond training data).

## The Problem: LLM Limitations

### Hallucination

LLMs generate text based on learned patterns, not from a ground truth knowledge source. They can confidently produce convincing but false statements:

- Invented citations to non-existent papers.
- Fabricated historical events or dates.
- Made-up product specifications or technical details.

### Knowledge Cutoff

An LLM's knowledge is frozen at training time. If trained in 2022, it cannot answer questions about events in 2024.

### Computational Cost

For specialized domains (law, medicine, proprietary company data), fine-tuning an LLM on all domain-specific data is expensive. Retrieval allows leveraging external knowledge without model updates.

## How RAG Works

### Architecture

RAG combines two components:

1. **Retriever**: A fast information retrieval system that, given a query, returns relevant documents or passages from a knowledge base.
2. **Generator**: The LLM that conditions on retrieved documents to generate an answer.

### Basic Pipeline

1. **Indexing phase** (offline):
   - Partition a knowledge base (Wikipedia, company documents, research papers) into chunks or documents.
   - Compute dense embeddings for each chunk using a retrieval model (e.g., BERT-based dense retrieval).
   - Store embeddings in a vector database for fast retrieval (e.g., FAISS, Pinecone, Weaviate).

2. **Retrieval phase** (at query time):
   - Encode the user query into the same embedding space.
   - Search the vector database to find the top-k most relevant documents (nearest neighbors by cosine similarity).

3. **Generation phase** (at query time):
   - Concatenate the retrieved documents with the user query: `[CONTEXT: retrieved_docs]\n[QUESTION: user_query]`.
   - Feed this concatenation to the LLM, which generates an answer grounded in the retrieved context.

## Retrieval Methods

### Dense Retrieval

Encode both queries and documents into dense vector embeddings, using cosine similarity to find relevant documents.

**Advantages**:
- Fast at scale (nearest-neighbor search with optimized libraries like FAISS).
- Can capture semantic similarity beyond keyword matching.
- Works well for natural language queries.

**Examples**: DPR (Dense Passage Retrieval), ColBERT, BM25, Hybrid methods.

### Sparse Retrieval (Keyword-Based)

Traditional methods like **TF-IDF** or **BM25** rank documents by term overlap with the query.

**Advantages**:
- Deterministic; easy to debug and understand.
- Efficient and interpretable (shows which terms matched).
- Strong baseline; often beats dense retrieval on some benchmarks.

**Disadvantages**:
- Misses semantic similarity when vocabulary differs.

### Hybrid Retrieval

Combine dense and sparse retrieval, leveraging strengths of both:
- Sparse retrieval catches keyword matches.
- Dense retrieval captures semantic similarity.
- Ensemble or learned fusion of rankings improves robustness.

### Reranking

After initial retrieval of top-k candidates, use a more sophisticated (slower) model to rerank them:

1. **Retriever** returns top-100 candidates (fast, approximate).
2. **Reranker** scores top-100 more carefully (slow, accurate), returns top-5.
3. **Generator** uses top-5 for generation.

This two-stage approach balances speed and accuracy.

## Training RAG Models

### End-to-End Training

Train the retriever and generator jointly to maximize generation quality on a task (e.g., open-domain QA).

- **Positive examples**: Ground-truth documents that support the correct answer.
- **In-batch negatives**: Other examples' documents that appear in the same batch — treated as negatives.
- **Hard negatives**: Documents that are relevant to the query but do not support the correct answer — critical for strong learning signals.

Joint training aligns the retriever and generator: the retriever learns to return documents the generator can effectively use.

### Frozen Retriever, Fine-Tuned Generator

Alternatively, use a pre-trained retriever (e.g., a publicly available dense retrieval model) and fine-tune only the generator:

- Simpler to implement.
- Reuses existing retrieval systems.
- Less flexible but often sufficient in practice.

## Practical Challenges

### Document Chunking

How to partition the knowledge base?

- **Fixed-size chunks** (e.g., 256 tokens): Simple but may break semantic units.
- **Semantic chunking**: Use language understanding to chunk at meaningful boundaries (end of sentences, paragraphs, sections).
- **Sliding windows**: Overlap between chunks to preserve context at boundaries.

Optimal chunk size depends on task and retriever; 256–1024 tokens is typical.

### Context Window Limits

The LLM has a maximum context length. With large retrieved documents:

1. **Compression**: Summarize or extract key information from retrieved documents.
2. **Selective passage inclusion**: Rank passages within a document; include only the top ones.
3. **Hierarchical retrieval**: Retrieve coarse-grained documents first, then fine-grained passages.

### Retrieval Failures

If the retriever fails to find relevant documents:
- The generator has no correct information to use, and hallucination is likely.
- Importance of retriever quality and need for fallback mechanisms (e.g., LLM can decline to answer if uncertain).

### Latency

Retrieval adds latency at inference. For real-time applications:
- Cache frequently asked queries and their retrieved documents.
- Use approximate nearest-neighbor search (e.g., HNSW) for speed.
- Parallelize retrieval across multiple systems.

## Applications

### Open-Domain Question Answering

Retrieve Wikipedia passages relevant to a question, then generate an answer grounded in those passages. Models achieve higher accuracy and reduce hallucination compared to LLMs alone.

### Knowledge-Intensive Tasks

- **Fact checking**: Retrieve evidence documents, then verify claims.
- **Citation generation**: Retrieve papers, then cite them in generated text.
- **Legal document analysis**: Retrieve relevant statutes or precedents; generate legal opinions grounded in retrieved documents.

### Proprietary Chatbots

Index company documents, code repositories, or internal wikis. Queries retrieve relevant documents, and an LLM generates responses grounded in company knowledge — enabling up-to-date, accurate chatbots without retraining.

## Evaluation

**RAG systems are evaluated on**:

- **Retrieval recall**: Did the retriever find relevant documents?
- **Answer accuracy**: Is the generated answer correct?
- **Faithfulness**: Does the answer faithfully reflect retrieved documents (not contradict them)?
- **Hallucination rate**: How often does the model generate false information?

Common benchmarks: SQuAD (QA), Natural Questions (open-domain QA), HotpotQA (multi-hop reasoning).

## Future Directions

- **Multi-hop retrieval**: Retrieve multiple relevant documents iteratively, reasoning across them (e.g., for complex questions requiring information fusion).
- **Adaptive retrieval**: Decide when to retrieve, how much to retrieve, and from which sources (e.g., local data vs. web).
- **Feedback loops**: Use user feedback to improve the retriever and generator over time.
- **Multimodal RAG**: Retrieve and reason over images, tables, and other modalities alongside text.

Retrieval-Augmented Generation bridges the gap between the flexibility of large language models and the factuality of curated knowledge bases — a practical approach to building reliable, up-to-date AI systems.
