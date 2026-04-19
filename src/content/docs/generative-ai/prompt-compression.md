---
title: Prompt Compression Techniques
description: Learn how prompt compression reduces LLM input costs and latency by condensing long contexts — covering token-level pruning, extractive compression, learned compression, and trade-offs for RAG and long-context applications.
---

**Prompt compression** refers to techniques that reduce the number of tokens in an LLM's input without significantly degrading the quality of its output. As LLM applications increasingly use long contexts — retrieved documents, conversation histories, tool outputs — prompt compression has become an important technique for controlling cost, reducing latency, and respecting context window limits.

## Why Prompt Compression Matters

LLM API pricing is proportional to tokens consumed. A typical production RAG system might inject 5–20 retrieved document chunks into every request, each 200–500 tokens long. At scale:

| Scenario | Input Tokens | GPT-4o Cost/Request | 1M Requests/Day |
|---|---|---|---|
| No compression | 5,000 | ~$0.0125 | $12,500/day |
| 50% compression | 2,500 | ~$0.00625 | $6,250/day |
| 70% compression | 1,500 | ~$0.00375 | $3,750/day |

Beyond cost, compression reduces **time-to-first-token** (TTFT) — shorter inputs require less prefill computation.

## The Compression-Quality Trade-Off

All compression techniques trade off **compression ratio** against **information preservation**:

$$\text{Compression Ratio} = 1 - \frac{\text{Compressed Tokens}}{\text{Original Tokens}}$$

A 70% compression ratio means the compressed prompt is 30% of the original size. The question is how much task performance degrades at that ratio — different tasks tolerate different compression levels.

## Approaches to Prompt Compression

### 1. Extractive Compression (Token Pruning)

**Extractive** methods select and retain the most important tokens or sentences from the original prompt, discarding the rest. No rewriting occurs.

**LLMLingua** (Microsoft, 2023) computes the **perplexity** of each token conditioned on the preceding tokens using a small proxy LLM (e.g., LLaMA-7B):

$$\text{Importance}(t_i) \propto -\log P_\text{proxy}(t_i \mid t_1, \ldots, t_{i-1})$$

High-perplexity tokens are surprising given context — likely more informative. Low-perplexity tokens are predictable and can often be dropped.

**LLMLingua-2** extends this by training a token classification model on data from GPT-4's compression decisions, improving quality and speed.

**Selective Context** (Lianmin Zheng et al.) extracts key phrases and entities, discarding function words, stop words, and redundant phrases while preserving content words.

### 2. Summarization-Based Compression (Abstractive)

Rather than selecting existing tokens, **abstractive** methods generate a condensed rewrite:

- A smaller LLM (or the same LLM) summarizes long retrieved documents.
- Conversation histories are compressed into summaries as they grow.
- Chain-of-thought traces are summarized for subsequent reasoning steps.

**Advantages:**

- Can express dense information more concisely than extraction.
- Can resolve coreferences and integrate information across passages.

**Disadvantages:**

- Introduces a second LLM call (cost and latency).
- The summarizer may drop details that are later needed.
- Harder to audit — which facts were preserved?

### 3. Learned Prompt Compression

**Learned compression** trains encoder models to represent long text as a small number of **soft tokens** — continuous embedding vectors rather than discrete words — that carry the semantic content of the original.

**GIST tokens** (Mu et al., 2023): A model is trained to compress a long instruction into $k$ special "GIST" tokens. The compressed tokens are cached and reused across requests with the same instruction.

**xRAG** (Ma et al., 2024): Retrieved document embeddings are compressed into a single continuous vector token that the LLM's attention can query, drastically reducing retrieved context length.

**LLoCO** (Context Compression via Query-Conditioned Reranking): Compresses documents by selectively retaining sentences relevant to the likely query distribution.

**Limitation:** Learned soft-token compression is model-specific — the compressed representations cannot be transferred to a different LLM.

### 4. Context Distillation via Fine-Tuning

Instead of compressing the prompt at inference time, **context distillation** trains the LLM itself to internalize frequently needed knowledge, so it doesn't need to be included in the prompt.

- Fine-tune the model on domain-specific data so it can answer domain questions without retrieval.
- "Teach" the model frequently used tool documentation so it doesn't need to be in the system prompt.

This trades inference-time prompt length for training-time investment.

### 5. Structural Compression

**Structural** compression exploits the structure of the prompt:

- **Deduplication**: Remove repeated or near-duplicate sentences across retrieved documents.
- **Template compression**: Replace boilerplate system prompt text with shorter aliases, mapping them back internally.
- **Conversation compression**: Periodically summarize and truncate conversation history, keeping only recent exchanges in full.
- **BM25 / TFIDF reranking**: Rerank retrieved chunks by relevance and discard low-scoring ones before injection.

## Comparison of Techniques

| Technique | Compression Ratio | Quality | Latency Overhead | Portability |
|---|---|---|---|---|
| **LLMLingua (token pruning)** | 60–80% | Good | Low (small proxy model) | Any LLM |
| **Summarization** | 50–90% | Variable | High (extra LLM call) | Any LLM |
| **GIST tokens** | 95%+ | Good | Near-zero at inference | Model-specific |
| **Structural (dedup + rerank)** | 20–40% | High | Very low | Any LLM |
| **Context distillation** | 90%+ | High | None at inference | Model-specific |

## Prompt Compression for RAG Systems

RAG is the primary use case for prompt compression:

```
Original: [Query] + [Doc1: 450 tokens] + [Doc2: 380 tokens] + [Doc3: 420 tokens]
Total: ~1,280 tokens

Compressed: [Query] + [CompressedDoc1: 135 tokens] + [CompressedDoc2: 110 tokens] + [CompressedDoc3: 120 tokens]
Total: ~400 tokens (69% reduction)
```

Key considerations for RAG compression:

- Compress documents **independently** of each other; compress with respect to the **query** when possible (query-conditioned compression preserves query-relevant content).
- Measure **answer faithfulness** with and without compression to verify quality is maintained.
- Different document types have different compression tolerance: legal text and code tolerate less compression than narrative prose.

## Prompt Caching vs. Prompt Compression

These are complementary techniques:

- **Prompt caching** (OpenAI, Anthropic): Reuses a cached KV representation of a prefix that is identical across requests. Reduces latency and cost for requests sharing a long system prompt.
- **Prompt compression**: Reduces the size of variable content (retrieved documents, conversation history).

Optimal production systems use both: cache a long, static system prompt and compress dynamic retrieved context.

## Further Reading

- [LLMLingua: Compressing Prompts for Accelerated Inference — Jiang et al., 2023](https://arxiv.org/abs/2310.05736)
- [LLMLingua-2 — Pan et al., 2024](https://arxiv.org/abs/2403.12968)
- [GIST: Efficient Data Transfer for Large Language Models — Mu et al., 2023](https://arxiv.org/abs/2304.08467)
- [Selective Context — Li et al., 2023](https://arxiv.org/abs/2304.01597)
