---
title: "Inference Optimization: Making AI Models Faster and Cheaper"
description: "Techniques to reduce latency and cost of AI model inference through caching, batching, and architecture choices."
---

An API call takes 2 seconds. Your system needs responses in 100ms. Running 10,000 inferences per day costs $500, but your budget is $50. Inference optimization addresses these constraints.

## The Latency Bottlenecks

**Token Generation is Sequential**
```
LLM generates one token at a time.
Prompt: 500 tokens (fast)
Generation: 1 token at 50ms/token = 3+ seconds for 60 tokens
Total: 3.5+ seconds
```

You can't parallelize token generation. This is fundamental to how autoregressive models work.

## Caching Strategies

### Prompt Caching
Many requests share the same initial context. Cache it.

```
Scenario: Company knowledge base queries
- System prompt: (shared by all)
- Company policy docs: (shared by all)
- User query: (unique per request)

Without caching: Re-process system + docs for each query
With caching: Process once, reuse (10-50x faster)
```

**Implementation:**
- **Redis:** In-memory cache (milliseconds latency)
- **Vector Database Cache:** Cache embeddings and retrieved documents
- **LLM Provider Caching:** Some APIs (Claude, OpenAI) offer built-in prompt caching

### KV-Cache Management
The model maintains key-value caches during generation. Optimize this:

```
Standard: Full KV cache for each token (~100MB per request)
Optimized: Sparse KV cache (cache only important tokens)
Result: 3-4x less memory, faster generation
```

## Batching

Instead of handling one request at a time:

```
Sequential (bad for throughput):
- Request 1: 1s latency
- Request 2: 1s latency (starts after Request 1)
- Request 3: 1s latency
Total time: 3s for 3 requests

Batched (good for throughput):
- Requests 1, 2, 3: 1.2s latency (processed together)
Total time: 1.2s for 3 requests
```

**Trade-off:** Individual latency increases slightly, but throughput is higher.

**Implementation:**
- Queue requests and process in batches of 8-32
- Batch size depends on GPU memory
- Works best with similar request lengths

## Speculative Decoding

The model generates tokens faster using guidance:

```
Without: Model generates each token from scratch (50ms)
With: Fast "draft" model generates 5 likely tokens,
      slow model verifies them (70ms for 5 tokens = 14ms effective)
```

Can achieve 2-3x speedup if draft model is accurate.

## Model Selection for Speed

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **Gemini 1.5 Flash** | Very Fast | 85-90% | Speed critical, cost critical |
| **GPT-4o Mini** | Fast | 90% | Balanced, cost-effective |
| **Claude 3 Haiku** | Fast | 88% | Fast, accurate-enough |
| **GPT-4 Turbo** | Slow | 95% | Complex reasoning |
| **Claude 3 Opus** | Slow | 96% | Maximum accuracy |

## Quantization for Speed

```
FP32 Model: 70GB, 1x speed
Quantized (4-bit): 7GB, 3-4x faster
```

Local quantized models beat remote APIs for many workloads.

## Architecture Choices

### Streaming Responses
Don't wait for the full response before showing it:

```
User sees first token in 100ms
User sees complete response in 3s (but can start reading at 100ms)
```

Improves perceived latency even if actual latency is unchanged.

### Asynchronous Processing
For non-real-time tasks:

```
User submits task
System: "Processing, we'll email results"
Process in background during off-peak hours (cheaper)
Send results when ready
```

### Delegation
Complex task that needs multiple steps? Delegate to multiple faster models:

```
User query → Fast classifier (categorize) → 50ms
           → Appropriate expert model → 1s
           → Fast ranker (best response) → 50ms
Total: 1.1s (might be faster than one slow model at 3s)
```

## Cost Optimization

### 1. Input Reduction
Every token costs money.

```
Full prompt: "Analyze this customer support ticket. Consider 
all context including the customer's purchase history, previous 
interactions, refund policies, and current inventory. Then 
recommend the best response."

Optimized: "Categorize support ticket. Recommend action."
```

40% fewer tokens = 40% cheaper.

### 2. Model Selection by Complexity
```
Simple classification: Use small model ($0.001/1k tokens)
Complex reasoning: Use large model ($0.03/1k tokens)

Route 80% of requests to small model (cheap)
Route 20% to large model (expensive but handles hard cases)
Result: Average cost 80% of budget-conscious approach
```

### 3. Batch Processing
Process 1000 queries at once during off-peak hours instead of one at a time during peak. Often 30-50% cheaper.

## Practical Optimization Checklist

- [ ] Enable prompt caching (10-100x improvement for repeated context)
- [ ] Use streaming responses (better UX, same latency)
- [ ] Batch requests when possible (improves throughput)
- [ ] Right-size your model (don't use GPT-4 for classification)
- [ ] Cache embeddings (don't re-embed the same documents)
- [ ] Monitor actual latency (test with production-like data)
- [ ] Profile bottlenecks (don't optimize without data)

## Real-World Example

Starting point:
- 100 requests/second
- Average latency: 2.5s
- Cost: $50/day

After optimization:
- Use batch processing: 1.8s latency, $40/day
- Route 70% to cheaper model: 1.8s latency, $25/day
- Add prompt caching: 0.8s latency (from cache), $25/day
- Final: 0.8s-1.8s latency, $25/day (50% cost reduction)