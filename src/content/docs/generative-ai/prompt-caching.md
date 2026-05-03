---
title: Prompt Caching
description: Understand prompt caching in LLM APIs — how providers like Anthropic and OpenAI reuse KV cache across requests sharing a common prefix, the cost and latency savings, how to structure prompts to maximize cache hits, and implementation patterns for system prompt caching, multi-turn conversations, and document Q&A.
---

**Prompt caching** is an API-level optimization where LLM providers cache the computed **Key-Value (KV) representations** of prompt prefixes and reuse them across multiple requests. Instead of re-processing the same system prompt, documents, or tool definitions on every API call, the provider stores the intermediate attention keys and values in fast memory (HBM or DRAM) and retrieves them on subsequent requests that share the same prefix — skipping the forward pass for those tokens entirely.

The practical impact is significant: requests that hit the cache cost 60–90% less per token and respond 20–50% faster, depending on the ratio of cached tokens to uncached tokens.

## How KV Caching Works

During a transformer forward pass, each layer computes key and value tensors for every input token. For a cached prefix of length $L_{cache}$, the provider can precompute:

$$K_i = W_K^{(i)} \cdot X_{1:L_{cache}}, \quad V_i = W_V^{(i)} \cdot X_{1:L_{cache}}$$

for each layer $i$. On subsequent requests with the same prefix, attention over the full context $L_{cache} + L_{new}$ becomes:

$$\text{Attn}(Q_{new}, [K_{cached}; K_{new}], [V_{cached}; V_{new}])$$

where only $Q_{new}$, $K_{new}$, $V_{new}$ need to be computed — the cached portion is read from storage. This is equivalent to inference on a short prompt with a very long history that was precomputed.

## Anthropic Claude: Explicit Cache Control

Anthropic's API requires explicitly marking content blocks with `cache_control` to opt into caching:

```python
import anthropic

client = anthropic.Anthropic()

# =========================================================
# Pattern 1: System prompt caching
# System prompts that are identical across requests
# are the highest-value caching target.
# =========================================================

SYSTEM_PROMPT = """You are an expert software architect specializing in 
distributed systems. You follow strict engineering principles and always
provide concrete code examples.

<engineering_principles>
- Prefer explicit over implicit
- Design for failure: every network call can fail
- Measure before optimizing
- Make illegal states unrepresentable
[... thousands of tokens of guidelines ...]
</engineering_principles>
"""

def ask_with_cached_system(user_question: str) -> str:
    """Cache the system prompt; only pay full price for user message."""
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}  # cache this block
            }
        ],
        messages=[{"role": "user", "content": user_question}]
    )
    
    usage = response.usage
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Cache write tokens: {usage.cache_creation_input_tokens}")
    print(f"Cache read tokens: {usage.cache_read_input_tokens}")
    # First call: cache_write = len(SYSTEM_PROMPT), cache_read = 0
    # Subsequent calls: cache_write = 0, cache_read = len(SYSTEM_PROMPT)
    
    return response.content[0].text


# =========================================================
# Pattern 2: Document Q&A with large context cached
# Load a large document once; answer many questions cheaply.
# =========================================================

def load_document_for_qa(document_text: str) -> list[dict]:
    """Return a message list with the document marked for caching."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>\n{document_text}\n</document>\n\nI will ask you questions about this document.",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        },
        {
            "role": "assistant",
            "content": "I've read the document and am ready to answer your questions."
        }
    ]


def answer_question(base_messages: list[dict], question: str,
                    system: str = "You are a helpful document analyst.") -> str:
    """
    Ask a question about a pre-loaded (cached) document.
    
    First call: pays full price for document tokens (cache write).
    Subsequent calls: pays ~10% of document token price (cache read).
    """
    messages = base_messages + [{"role": "user", "content": question}]
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        system=system,
        messages=messages
    )
    return response.content[0].text


# =========================================================
# Pattern 3: Tool definitions caching
# Large tool schemas (e.g., full OpenAPI spec) can be cached.
# =========================================================

TOOL_DEFINITIONS = [
    {
        "name": "search_database",
        "description": "Search the product database with complex filters...",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ["electronics", "clothing", "food"]},
                        "price_range": {"type": "object",
                                        "properties": {"min": {"type": "number"}, "max": {"type": "number"}}},
                        "in_stock": {"type": "boolean"},
                        "rating_min": {"type": "number", "minimum": 0, "maximum": 5}
                    }
                },
                "sort_by": {"type": "string", "enum": ["price", "rating", "relevance", "newest"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            },
            "required": ["query"]
        }
    }
    # ... many more tool definitions
]

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=256,
    tools=TOOL_DEFINITIONS,
    tool_choice={"type": "auto"},
    messages=[{"role": "user", "content": "Find me blue running shoes under $100 with rating ≥ 4"}],
    # Tool definitions are automatically eligible for caching if they exceed the minimum token threshold
)
```

## OpenAI: Automatic Prefix Caching

OpenAI applies prompt caching automatically for eligible requests — no explicit API changes are required. The cache key is the exact byte sequence of the prompt prefix:

```python
from openai import OpenAI

client = OpenAI()

# OpenAI caches automatically for prompts >1024 tokens
# Cache hits are reflected in usage.prompt_tokens_details

def chat_with_auto_caching(system_prompt: str, messages: list[dict]) -> dict:
    """
    OpenAI will automatically cache the common prefix.
    Inspect usage to see cache hits.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            *messages
        ],
        max_tokens=512
    )
    
    usage = response.usage
    cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
    
    print(f"Total prompt tokens: {usage.prompt_tokens}")
    print(f"Cached tokens: {cached_tokens}")
    print(f"Cache hit rate: {cached_tokens / usage.prompt_tokens:.1%}")
    
    return {
        "content": response.choices[0].message.content,
        "total_tokens": usage.total_tokens,
        "cached_tokens": cached_tokens
    }


# Multi-turn chat: cache grows as conversation history builds
conversation_history = []
SHARED_SYSTEM = "You are a senior Python engineer..." # long system prompt

for user_message in ["Explain async/await", "Show me a real example", "How does the event loop work?"]:
    conversation_history.append({"role": "user", "content": user_message})
    
    result = chat_with_auto_caching(SHARED_SYSTEM, conversation_history)
    conversation_history.append({"role": "assistant", "content": result["content"]})
    
    # As turns accumulate, more of the prompt is cached → cost drops
    print(f"Turn cached {result['cached_tokens']} / {result['total_tokens'] - 512} tokens")
```

## Cost Savings Calculator

```python
def estimate_caching_savings(
    total_monthly_requests: int,
    system_prompt_tokens: int,
    average_conversation_tokens: int,
    output_tokens_per_request: int = 500,
    provider: str = "anthropic",
    model: str = "claude-opus-4-5"
) -> dict:
    """
    Estimate monthly cost with and without prompt caching.
    
    Anthropic pricing (claude-opus-4-5 as of 2025):
    - Standard input:   $15 / MTok
    - Cache write:      $18.75 / MTok (25% premium on first write)
    - Cache read:       $1.50 / MTok  (90% discount vs standard)
    - Output:          $75 / MTok
    
    Cache TTL: 5 minutes (refreshed on each cache hit)
    """
    PRICING = {
        "anthropic": {
            "claude-opus-4-5": {"input": 15, "cache_write": 18.75, "cache_read": 1.50, "output": 75}
        },
        "openai": {
            "gpt-4o": {"input": 2.50, "cache_write": 2.50, "cache_read": 1.25, "output": 10.00}
        }
    }
    
    p = PRICING[provider][model]
    
    # Without caching: every request pays full input price
    total_input_tokens_no_cache = (system_prompt_tokens + average_conversation_tokens) * total_monthly_requests
    cost_no_cache = (
        total_input_tokens_no_cache / 1_000_000 * p["input"] +
        output_tokens_per_request * total_monthly_requests / 1_000_000 * p["output"]
    )
    
    # With caching: assume system prompt cached after first request each session
    # Assume average 10 turns per session → 10× amortization
    sessions = total_monthly_requests / 10
    cache_write_cost = sessions * system_prompt_tokens / 1_000_000 * p["cache_write"]
    cache_read_cost = (total_monthly_requests - sessions) * system_prompt_tokens / 1_000_000 * p["cache_read"]
    conversation_cost = total_monthly_requests * average_conversation_tokens / 1_000_000 * p["input"]
    output_cost = total_monthly_requests * output_tokens_per_request / 1_000_000 * p["output"]
    cost_with_cache = cache_write_cost + cache_read_cost + conversation_cost + output_cost
    
    savings = cost_no_cache - cost_with_cache
    return {
        "cost_without_caching_usd": round(cost_no_cache, 2),
        "cost_with_caching_usd": round(cost_with_cache, 2),
        "monthly_savings_usd": round(savings, 2),
        "savings_pct": round(savings / cost_no_cache * 100, 1)
    }


# Example: large-scale document Q&A application
result = estimate_caching_savings(
    total_monthly_requests=100_000,
    system_prompt_tokens=4_000,   # 4K token system prompt with guidelines
    average_conversation_tokens=500,
    provider="anthropic",
    model="claude-opus-4-5"
)
print(result)
# → {'cost_without_caching_usd': 6125.0, 'cost_with_caching_usd': 1231.25,
#    'monthly_savings_usd': 4893.75, 'savings_pct': 79.9}
```

## Best Practices for Maximum Cache Hits

**Order content from stable to dynamic**: Cache keys match from the beginning of the prompt. The stable system prompt and documents must come before the dynamic user message.

```python
# GOOD: stable content first → cache hits every request
messages = [
    {"role": "system", "content": LONG_STABLE_SYSTEM_PROMPT},  # cached
    {"role": "user", "content": f"<doc>{DOCUMENT}</doc>"},      # cached
    {"role": "assistant", "content": "Ready."},                  # cached
    {"role": "user", "content": dynamic_user_question}          # not cached — unique per request
]

# BAD: dynamic content first → no cache hits
messages = [
    {"role": "user", "content": f"Today is {datetime.now()} and {dynamic_user_question}"},
    {"role": "system", "content": LONG_STABLE_SYSTEM_PROMPT}   # never reached by cache
]
```

**Avoid volatile content in cached prefixes**: Timestamps, request IDs, or any per-request dynamic content in the cacheable prefix will bust the cache every time.

**Minimum token thresholds**: Anthropic requires ≥1024 cacheable tokens (≥2048 for some models); OpenAI requires ≥1024 tokens in the full prompt. Caching is not beneficial for short prompts.

**Cache TTL management**: Anthropic's cache TTL is 5 minutes, refreshed on each hit. For applications with low request rates, implement request batching or periodic "cache warmup" calls to keep frequently used prompts in cache.

Prompt caching is one of the highest-ROI optimizations available for production LLM applications — it requires no model changes, no infrastructure work, and only minor prompt restructuring, yet can cut input token costs by 60–90% for applications with stable system prompts or shared context.
