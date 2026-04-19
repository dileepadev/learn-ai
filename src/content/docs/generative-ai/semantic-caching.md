---
title: Semantic Caching for LLMs
description: Learn how semantic caching reduces LLM latency and cost by reusing responses to semantically similar queries — covering similarity-based lookup, cache key design, invalidation strategies, and production architecture.
---

**Semantic caching** is a caching technique for LLM applications that reuses previously generated responses not just for identical queries, but for **semantically similar** ones. Unlike traditional exact-match caching (which misses on any textual variation), semantic caching finds queries whose meaning is close enough that the cached response is still accurate and useful.

## The Problem with Exact-Match Caching

Traditional caching works on exact string equality:

```
Key: "What is the capital of France?"
Value: "The capital of France is Paris."
```

This cache would miss all of these equivalent queries:

- "What's France's capital city?"
- "Name the capital of France."
- "Capital of France?"
- "france capital"

In conversational AI applications, the same underlying question is asked thousands of ways. Exact-match caching typically achieves under 5% hit rates on real traffic, whereas semantic caching can achieve 30–60% hit rates on FAQ-style and knowledge retrieval workloads.

## How Semantic Caching Works

### Architecture

```
User Query
    ↓
[Embedding Model] → Query Vector
    ↓
[Vector DB Lookup] ← Similarity Search (cosine)
    ↓ (cache hit if similarity > threshold)
[Return cached response]
    ↓ (cache miss)
[LLM API Call]
    ↓
[Store (query embedding, response) in vector DB]
    ↓
[Return response to user]
```

### 1. Query Embedding

The incoming query is embedded using a fast, efficient embedding model:

$$q = E(x_\text{query}) \in \mathbb{R}^d$$

Suitable embedding models for semantic caching prioritize **speed and consistency** over deep semantic representation:

- `text-embedding-3-small` (OpenAI) — fast, low cost.
- `bge-small-en-v1.5` (BAAI) — open-source, high speed.
- `all-MiniLM-L6-v2` (Sentence Transformers) — very fast, good quality.

### 2. Similarity Search

The query vector is searched against previously cached query vectors using approximate nearest neighbor (ANN) search:

$$\text{match} = \arg\max_{q_i \in \text{cache}} \text{cos}(q, q_i)$$

A **similarity threshold** $\tau$ determines whether the match is close enough to use:

$$\text{cache hit} \iff \text{cos}(q, q_\text{best}) \geq \tau$$

Typical thresholds range from $0.92$ to $0.97$. Lower thresholds increase hit rate but risk returning an incorrect response for a different question.

### 3. Response Return or LLM Call

- **Cache hit**: Return the stored response directly. Latency: ~10ms (vector search).
- **Cache miss**: Forward to LLM API. Store the new (query embedding, response) pair. Latency: 500ms–5s.

## Threshold Selection

Choosing the right similarity threshold is the critical design decision in semantic caching:

| Threshold | Behavior |
|---|---|
| Too low (< 0.85) | High hit rate, but semantically different queries served wrong answers |
| Optimal (0.92–0.97) | Balanced hit rate and correctness for most use cases |
| Too high (> 0.99) | Effectively degenerates to near-exact-match, low hit rate |

The optimal threshold depends on the application domain:

- **Factual Q&A** (lower risk of wrong answer): Can use lower thresholds (~0.90).
- **Personalized responses** (answers must be user-specific): Should not use semantic caching, or use higher thresholds with user context segmentation.
- **Legal or medical advice**: Very high thresholds or no semantic caching — precision critical.

## Cache Key Design

What is used as the cache key (the embedded content) varies:

- **Query only**: Simple, but ignores system prompt and context.
- **Query + system prompt hash**: Different system prompts produce different cache spaces.
- **Query + user segment**: Cache per user group (e.g., geography, language).
- **Query + conversation context hash**: For multi-turn conversations, include recent context.

## Cache Invalidation

Semantic caches require invalidation strategies for:

- **Stale responses**: Cached responses become outdated when underlying knowledge changes (product updates, policy changes).
- **Model updates**: A new LLM version produces better responses; old cached responses should eventually expire.
- **TTL (Time-to-Live)**: Responses cached for more than $N$ days are automatically expired.
- **Tag-based invalidation**: Group cache entries by topic or domain; invalidate the entire group when that topic changes.

## Production Architecture

### With Redis

**GPTCache** (open-source) provides a semantic caching layer compatible with OpenAI's API:

```python
from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

onnx = Onnx()
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=get_data_manager(CacheBase("sqlite"), VectorBase("faiss")),
    similarity_evaluation=SearchDistanceEvaluation(),
)
```

### With LangChain

LangChain's `SemanticSimilarityExactMatchCache` provides a drop-in semantic cache for any `ChatModel`:

```python
from langchain.cache import RedisSemanticCache
from langchain.embeddings import OpenAIEmbeddings
import langchain

langchain.llm_cache = RedisSemanticCache(
    embedding=OpenAIEmbeddings(),
    redis_url="redis://localhost:6379",
    score_threshold=0.95,
)
```

## Cost and Latency Impact

On a high-traffic FAQ chatbot:

| Metric | Without Caching | With Semantic Cache (50% hit rate) |
|---|---|---|
| Avg. response latency | 1,200ms | 615ms |
| LLM API cost/day | $500 | $255 |
| Vector DB cost/day | $0 | $15 |
| Net savings | — | ~48% |

The economics are most favorable for:

- High-traffic applications with repetitive query patterns.
- Expensive LLM API calls (GPT-4 class models).
- FAQ, knowledge base, and customer support chatbots.

## Limitations

- **Personalized responses**: Queries like "What's my account balance?" require user-specific context that semantic caching cannot handle without user-scoped cache partitions.
- **Creative generation**: Tasks like "Write me a poem" should not return cached responses — variety is expected.
- **Long-form documents**: Caching full document generations has lower reuse value; caching sub-queries is more effective.
- **Cache poisoning**: Incorrect cached responses propagate until invalidated — monitoring for response quality is essential.

## Further Reading

- [GPTCache: A Library for Creating Semantic Cache — Zilliz, 2023](https://github.com/zilliztech/GPTCache)
- [Reducing LLM Costs with Semantic Caching — AWS, 2024](https://aws.amazon.com/blogs/machine-learning/reduce-llm-costs-with-semantic-caching/)
- [LangChain Semantic Cache Documentation](https://python.langchain.com/docs/how_to/llm_caching/)
