---
title: KV Cache Optimization in LLMs
description: Learn how the Key-Value cache enables efficient autoregressive generation in Transformers, and explore advanced optimization techniques including Multi-Query Attention, Grouped-Query Attention, and sliding window caching.
---

The **Key-Value (KV) cache** is one of the most important engineering techniques enabling practical LLM inference. Without it, generating each new token in a sequence would require recomputing attention over all previous tokens — an $O(n^2)$ cost per step. The KV cache reduces this to $O(n)$ by storing and reusing intermediate computations.

## Autoregressive Generation and Redundant Computation

Transformer models generate text one token at a time. At each step, the model must compute multi-head attention over all previous tokens plus the new one:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In a sequence of length $n$, the keys $K$ and values $V$ for tokens $1 \ldots n-1$ are **identical** to those computed in the previous step. Recomputing them wastes both time and energy.

## How the KV Cache Works

During the **prefill phase** (processing the prompt), all keys and values are computed and stored in a cache:

$$K_{\text{cache}} = [k_1, k_2, \ldots, k_n], \quad V_{\text{cache}} = [v_1, v_2, \ldots, v_n]$$

During the **decode phase** (token-by-token generation), each new token only computes its own query $q_{n+1}$ and appends its new $k_{n+1}, v_{n+1}$ to the cache. Attention is then computed as:

$$\text{Attention}(q_{n+1}, K_{\text{cache}}, V_{\text{cache}})$$

This reduces the decode computation from $O(n^2)$ to $O(n)$ per step — a massive speedup for long sequences.

## Memory Cost of the KV Cache

The savings in compute come at a memory cost. For each token, each layer stores:

$$\text{KV Cache per token} = 2 \times L \times H \times d_h \times \text{dtype\_bytes}$$

where:

- $L$ = number of layers
- $H$ = number of attention heads
- $d_h$ = head dimension
- Factor of 2 for both keys and values

**Example — LLaMA-3 8B (float16):**

- $L=32$ layers, $H=32$ heads, $d_h=128$
- Per token: $2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ bytes ≈ **0.5 MB/token**
- For a 4K context: ~2 GB just for KV cache

For very long contexts or large batch sizes, KV cache memory becomes the primary GPU memory constraint.

## Multi-Query Attention (MQA)

**MQA** (Shazeer, 2019) reduces KV cache memory by sharing a single key and value head across all query heads:

- Standard: $H$ query heads, $H$ key heads, $H$ value heads.
- MQA: $H$ query heads, **1** key head, **1** value head.

Memory reduction: $H\times$ for KV cache. This dramatically reduces decode-time memory bandwidth and speeds up inference with minimal accuracy degradation.

Used in: PaLM, Falcon, Mistral.

## Grouped-Query Attention (GQA)

**GQA** (Ainslie et al., 2023) is a middle ground between standard Multi-Head Attention (MHA) and MQA:

- Query heads are divided into $G$ groups.
- Each group shares a single set of key/value heads.
- Standard MHA = GQA with $G = H$; MQA = GQA with $G = 1$.

$$\text{KV Cache reduction} = \frac{H}{G}\times$$

GQA with $G=8$ is now the standard in most frontier models:

| Model | Attention Type | Query Heads | KV Heads |
|---|---|---|---|
| GPT-4 | MHA | - | - |
| LLaMA-2 7B | MHA | 32 | 32 |
| LLaMA-3 8B | GQA | 32 | 8 |
| Mistral 7B | GQA | 32 | 8 |
| LLaMA-3 70B | GQA | 64 | 8 |
| Gemma 2 9B | GQA | 8 | 4 |

## Sliding Window Attention

For very long contexts, caching all previous tokens is both memory-intensive and slow (attention cost grows with sequence length). **Sliding window attention** limits each token's attention to a fixed local window of $w$ previous tokens:

$$\text{Attention}(q_i, K_{[i-w:i]}, V_{[i-w:i]})$$

**Mistral 7B** uses a sliding window of 4,096 tokens with a rolling buffer cache — keeping memory constant regardless of sequence length.

**Limitation:** pure sliding window attention loses information beyond the window. Solutions:

- **Dilated/strided attention** — attend to every $k$-th token beyond the window.
- **Sink tokens** — keep a small number of initial tokens (attention sinks) in cache alongside the recent window (**StreamingLLM**, Xiao et al. 2023).

## Paged Attention and vLLM

Traditional KV cache allocates a contiguous memory block per request. With variable-length sequences and concurrent batches, this causes severe **memory fragmentation** — large portions of reserved GPU memory go unused.

**PagedAttention** (Kwon et al., 2023), the core innovation of **vLLM**, applies virtual memory paging to the KV cache:

- KV cache is divided into fixed-size **blocks** (pages).
- Blocks are allocated on demand and mapped via a block table — no contiguous pre-allocation.
- Blocks can be shared between sequences (e.g., common system prompt prefix).

Results: near-zero memory waste, 2–4× throughput increase over naive implementations.

## Quantized KV Cache

KV cache values can be quantized to lower precision:

- **INT8 KV cache** — Halves memory vs. float16; minimal quality degradation.
- **INT4 KV cache** — 4× reduction; requires careful calibration.
- **FP8 KV cache** — Hardware-native on H100 GPUs; best quality/compression trade-off.

LLM serving frameworks (vLLM, TensorRT-LLM, SGLang) all support quantized KV caches for production deployments.

## KV Cache Offloading

For very long contexts exceeding GPU VRAM, KV cache can be **offloaded** to CPU RAM or NVMe storage:

- **KVShift** — Offloads less recently used cache entries to CPU.
- **InfiniGen** — Pre-fetches relevant KV entries from CPU based on predicted attention patterns.

Trade-off: PCIe bandwidth becomes the bottleneck; latency increases.

## Prefix Caching / Prompt Caching

If many requests share a common prefix (system prompt, few-shot examples, large document), their KV cache can be computed once and reused:

- **vLLM prefix caching** — Automatic hash-based detection of matching prefixes.
- **Anthropic / OpenAI prompt caching** — API-level feature that caches and reuses KV states for repeated prefixes, significantly reducing costs for document Q&A and chatbot use cases.

## Summary of Techniques

| Technique | What it reduces | Trade-off |
|---|---|---|
| KV Cache (baseline) | Recomputation | Memory usage |
| MQA | KV memory ($H\times$) | Slight quality degradation |
| GQA | KV memory ($H/G\times$) | Minimal quality degradation |
| Sliding Window | Memory growth with length | Loses long-range context |
| PagedAttention | Memory fragmentation | Implementation complexity |
| INT8/INT4 Quantization | KV memory (2–4×) | Quantization error |
| Prefix Caching | Repeated prompt compute | Cache storage overhead |

KV cache optimization is central to the economics of LLM serving — the techniques above collectively enable serving much longer contexts, larger batches, and higher request throughput on the same hardware.
