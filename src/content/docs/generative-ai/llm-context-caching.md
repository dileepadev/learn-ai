---
title: LLM Context Caching
description: How context caching works in large language models — reusing KV cache across requests to dramatically reduce latency and cost for repeated prefixes.
---

Context caching allows an LLM API to reuse previously computed key-value (KV) cache for a shared prompt prefix, avoiding redundant computation across many requests. It is one of the most impactful cost and latency optimizations available when working with long, repeated context.

## The Problem It Solves

Many production LLM applications share a large common prefix across all requests:
- A lengthy system prompt or persona definition.
- A full document, codebase, or knowledge base loaded every call.
- A set of few-shot examples that don't change.

Without caching, every request re-processes the entire prefix from scratch. For a 50K-token system prompt sent 1,000 times per day, that is 50 million tokens of redundant computation daily — wasted cost and added latency.

## How KV Caching Works

During the transformer forward pass, each layer computes key (K) and value (V) tensors for every token in the context. These are stored in the KV cache and reused during autoregressive generation so tokens don't need to be recomputed.

Context caching extends this idea across requests: if two requests share an identical prefix, the server can store the KV cache for that prefix and reuse it for both, computing only the new tokens appended by each unique request.

## Provider Implementations

### Anthropic (Claude)
Claude's API supports explicit cache control with `cache_control` markers. You designate parts of the prompt as cacheable:

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "<very long document or instructions here>",
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": "Summarize the key points."}],
)
```

The first request computes and stores the cache; subsequent requests with the same cached block reuse it. Cached tokens are billed at a lower rate (roughly 10% of the standard input token price) and reduce latency by up to 90% for large prefixes.

### Google Gemini
Gemini supports explicit context caching through the `CachedContent` API. You create a cached content object for a long document or large system instruction and reference its ID in subsequent calls, paying a storage fee per hour rather than per token re-read.

### OpenAI
OpenAI implements automatic prompt caching for prompts over 1,024 tokens. The longest matching prefix is cached server-side transparently — no code changes are required. Cached tokens are billed at 50% of the input token price and cache hits are shown in the response usage object.

## When to Use Context Caching

Context caching provides the most benefit when:
- **The prefix is long** (1K+ tokens). Short prefixes have minimal savings.
- **The same prefix is reused frequently** across many requests. One-off requests gain nothing.
- **The prefix is stable** — changes to the cached portion invalidate the cache.
- **Latency matters.** Cache hits bypass the prefill phase entirely, cutting time-to-first-token significantly.

Ideal scenarios: document Q&A (cache the document), coding assistants (cache the codebase), customer service bots (cache the product knowledge base), few-shot classifiers (cache the examples).

## Cost vs. Latency Trade-offs

| Scenario | Without Cache | With Cache |
|----------|--------------|------------|
| 50K-token doc, 100 queries/day | Full 50K tokens billed per query | Only first query billed at full rate; rest at cache rate |
| Time-to-first-token | Long (must process full prefix) | Short (prefix KV is preloaded) |
| Cache creation cost | N/A | Small write cost or storage fee |

The break-even point is typically reached after just 2–3 requests for long prefixes.

## Limitations

- **Cache invalidation:** Any change to the cached prefix — even a single character — invalidates the cache. Keep cached content immutable.
- **Cache lifetime:** Caches expire after a period of inactivity (typically 5 minutes for Anthropic ephemeral caches, hours/days for Gemini explicit caches).
- **Prefix must be a true prefix:** The cached content must appear at the very beginning of the prompt. You cannot cache a middle section.
- **Not all models support it:** Check provider documentation; cache support varies by model version.

## Best Practices

- Put the largest, most stable content first in your prompt (system instructions, documents) and variable content last (user message, query).
- Monitor cache hit rates in API response metadata to verify the cache is being used.
- For Anthropic, be explicit about `cache_control` markers rather than relying on implicit behavior.
- Warm the cache with a dummy request when starting a new session to avoid cold-start latency for the first real user request.
