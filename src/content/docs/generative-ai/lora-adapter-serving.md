---
title: Serving Many LoRA Adapters on One Base Model
description: Learn how multi-LoRA serving shares one base model across many fine-tuned adapters, covering batching, adapter swapping, and the memory trade-offs involved.
---

LoRA fine-tunes produce small adapter weights rather than a full model copy. Serving systems can exploit this to host hundreds or thousands of customized variants on a single GPU deployment.

## The Economics

A 7B model in 16-bit precision takes about 14 GB. A LoRA adapter at rank 16 may be only tens of megabytes. Running a separate full model per customer is wasteful; keeping one base model plus many adapters is far cheaper.

## How LoRA Inference Works

For a weight matrix W, LoRA adds a low-rank update:

```text
y = W x + (alpha / r) * B (A x)
```

The base term `W x` is shared by all requests. Only the small `B(A x)` term differs by adapter.

## Two Strategies

**Merge adapters**: compute `W' = W + BA` offline. This has no extra inference latency but needs a full model copy per adapter, so it suits a single dominant variant.

**Dynamic, unmerged adapters**: keep adapters separate and apply them per request at runtime. This enables serving many adapters at once.

## Batching Requests with Different Adapters

A naive approach would run each adapter's requests separately, wasting GPU utilization. Systems such as S-LoRA, Punica, and vLLM's multi-LoRA support use custom kernels that apply different adapters to different requests within one batch:

```text
batch: [req1 (adapter A), req2 (adapter B), req3 (adapter A)]
step 1: shared base computation for all requests
step 2: gathered low-rank computation, each request using its own adapter
```

## Memory Management

- Adapters live in CPU memory or disk and are loaded into GPU memory on demand
- An LRU cache keeps hot adapters resident
- Unified paging, as in S-LoRA, manages adapter weights and KV cache in the same memory pool

## Example Configuration

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules support-bot=./adapters/support finance-bot=./adapters/finance \
  --max-loras 4 --max-lora-rank 32
```

Clients select an adapter by passing its name as the model.

## Considerations

- Adapter cold-start loading adds latency
- High rank or many target modules increases per-token overhead
- Quantized base models work with LoRA (as in QLoRA) but require compatible kernels
- Isolate tenants carefully: adapters can encode private data
