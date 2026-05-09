---
title: Retentive Networks and RWKV
description: Explore the Retentive Network (RetNet) and RWKV architectures — two linear-complexity alternatives to the Transformer that achieve training parallelism and efficient recurrent inference simultaneously, without attention's quadratic cost.
---

The Transformer's self-attention mechanism is its greatest strength and most significant bottleneck: it captures long-range dependencies with $O(L^2)$ time and memory cost in sequence length $L$. For training long-context language models and deploying them for real-time inference, this quadratic scaling is increasingly prohibitive. **Retentive Networks (RetNet)** and **RWKV** are two distinct architectures that independently arrived at a similar insight: design a sequence model that behaves like a recurrent network at inference time (constant per-step cost) but can be parallelized like a Transformer during training.

## The "Impossible Triangle" of Sequence Modeling

Sun et al. (2023) framed the design space as an "impossible triangle":

- **Training parallelism:** The ability to compute all sequence positions simultaneously during training (like Transformers), enabling GPU utilization.
- **Efficient inference:** Constant-time, constant-memory per token generation (like RNNs), enabling low-latency deployment.
- **Strong performance:** Competitive accuracy on downstream tasks.

Classic RNNs and LSTMs satisfy inference efficiency but fail training parallelism. Transformers satisfy training parallelism and performance but fail inference efficiency. RetNet and RWKV claim to satisfy all three simultaneously.

## Retentive Networks (RetNet)

### The Retention Mechanism

RetNet replaces attention with **multi-scale retention (MSR)**, derived from a recurrent formulation that has an equivalent parallel training form.

**Recurrent form** (inference, $O(1)$ per token):

$$s_n = \gamma s_{n-1} + k_n^\top v_n$$
$$o_n = q_n s_n$$

where $q_n, k_n \in \mathbb{R}^d$ and $v_n \in \mathbb{R}^{d_v}$ are projections of the input, $s_n \in \mathbb{R}^{d \times d_v}$ is the recurrent state, and $\gamma \in (0, 1)$ is a fixed decay scalar.

**Parallel form** (training, $O(L^2)$ computation but fully parallelizable):

$$\text{Retention}(Q, K, V) = (Q K^\top \odot D)\, V$$

where $D_{mn} = \gamma^{m-n}$ for $m \geq n$ (causal mask with exponential decay) and $D_{mn} = 0$ for $m < n$.

This is equivalent to scaled dot-product attention, but the decay mask $D$ removes the softmax normalization — replacing content-based weighting with **positional decay**. Later tokens influence future outputs less, controlled by $\gamma$.

**Chunkwise parallel form** (training efficiency): Process sequences in chunks of length $B$, using the parallel form within each chunk and the recurrent form to carry state between chunks. This gives $O(L \cdot B)$ memory and is the primary training mode in practice.

### Multi-Scale Retention

Different retention heads use different decay rates $\gamma_h$, enabling multi-scale temporal integration:

$$\gamma_h = 1 - 2^{-5 - h}, \quad h = 1, \ldots, H$$

Heads with $\gamma_h \approx 1$ integrate information over long distances; heads with smaller $\gamma_h$ focus on local context. This is analogous to multi-head attention but with fixed positional rather than content-based mixing.

### Architecture

A RetNet layer stacks:

1. **Multi-Scale Retention** (replaces multi-head attention)
2. **Gated Feed-Forward Network** with SiLU activation

The full model alternates retention layers and FFN layers, identical in structure to a Transformer.

### Performance

On language modeling benchmarks, RetNet-3B matches GPT-3-equivalent Transformer models in perplexity while offering:

- **8× lower memory** during inference (fixed recurrent state vs. growing KV cache)
- **5× faster** throughput at long sequences
- Fully parallel training with no modification to the training infrastructure

## RWKV: Receptance Weighted Key Value

**RWKV** (Peng et al., 2023) is an independent architecture developed by the community that achieves the same RNN/Transformer duality through a different mechanism — and has been scaled to full open-source language models (RWKV-4, RWKV-6) up to 14B parameters.

### The RWKV Attention-Free Mechanism

RWKV's core is the **time-mixing** block. For each token position $t$ and channel $i$:

**Key, Value, Receptance** projections:

$$k_t = W_k \cdot x_t, \quad v_t = W_v \cdot x_t, \quad r_t = \sigma(W_r \cdot x_t)$$

**Exponential moving average with learnable decay:**

$$\text{wkv}_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t} v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}}$$

where $w \geq 0$ is a learnable per-channel **time decay** and $u$ is a learnable **in-context bonus** that boosts the current token's value.

**Output:**

$$o_t = W_o \cdot (r_t \odot \text{wkv}_t)$$

The receptance gate $r_t$ (sigmoid) acts as a learned forget gate, analogous to an LSTM's forget gate. The $\text{wkv}$ computation is an exponentially weighted sum — equivalent to a linear RNN — but the key insight is that $e^{k_i}$ acts as a **content-dependent weight**, giving RWKV data-dependent recurrence without the full $O(L^2)$ cost of softmax attention.

### Recurrent Inference Form

At inference time, $\text{wkv}_t$ is computed from two running scalars (numerator and denominator) updated recurrently in $O(1)$ per step — the RWKV hidden state is a fixed-size vector regardless of context length.

### Channel Mixing Block

Complementing time-mixing, RWKV uses a **channel-mixing** block that is a gated variant of the FFN:

$$r'_t = \sigma(W'_r \cdot x'_t), \quad k'_t = \text{ReLU}(W'_k \cdot x'_t)^2$$
$$o'_t = r'_t \odot (W'_v \cdot k'_t)$$

This replaces the standard FFN while maintaining the RNN-compatible structure.

### RWKV-6 Improvements

RWKV-6 introduced **data-dependent time mixing**: the decay $w$ and bonus $u$ become functions of the input rather than fixed channel-wise parameters, closing some of the expressiveness gap with softmax attention and matching Transformer performance on a wider range of tasks.

## RetNet vs. RWKV vs. Transformers vs. Mamba

| Property | Transformer | RetNet | RWKV | Mamba (SSM) |
| --- | --- | --- | --- | --- |
| Training complexity | $O(L^2)$ | $O(L^2)$ / chunk | $O(L)$ | $O(L)$ |
| Inference per token | $O(L)$ KV cache | $O(1)$ | $O(1)$ | $O(1)$ |
| Content-aware mixing | Yes (softmax) | No (fixed decay) | Partial (key weight) | Yes (selective) |
| Open-source models | GPT-J, Llama | RetNet-3B | RWKV-14B | Mamba-2.8B |
| Long-range capability | Excellent | Good | Good | Excellent |
| Hardware efficiency | Moderate | Good | High | High |

## Practical Considerations

### When to Use RetNet/RWKV

These architectures are most valuable when:

- **Inference latency is critical:** Constant-memory recurrent inference avoids the KV-cache memory growth that limits Transformer context at deployment.
- **On-device / edge deployment:** Fixed-size hidden state is highly amenable to embedded and mobile hardware with limited DRAM.
- **Continuous/streaming processing:** Recurrent form naturally processes tokens one at a time without buffering the full context.
- **Very long sequences:** Linear-time RWKV training outperforms Transformers at sequence lengths exceeding 4K–8K tokens.

### Limitations

- **Positional decay fixed-structure:** RetNet's fixed exponential decay is less flexible than learned attention patterns for tasks requiring sharp, content-dependent long-range retrieval.
- **In-context learning gap:** RWKV models show somewhat weaker few-shot in-context learning compared to same-size Transformers, attributed to the loss of full content-based mixing.
- **Ecosystem maturity:** The tooling, benchmark coverage, and community support are smaller than the Transformer ecosystem.

## Current State and Trajectory

RWKV is backed by a large open-source community (RWKV.com), with models trained and fine-tuned across dozens of downstream tasks. RetNet has inspired subsequent work including **GateLoop** and **Griffin** (DeepMind, 2024), which combine linear recurrences with selective attention layers, achieving Transformer parity on benchmarks with linear inference cost.

The broader lesson from RetNet and RWKV is that the softmax is not necessary for competitive language modeling — structured linear recurrences with the right inductive biases capture enough of the temporal structure that matters for next-token prediction, at a fraction of the inference cost.

## Summary

Retentive Networks and RWKV are two of the most developed linear-complexity alternatives to the Transformer, each achieving the long-sought combination of parallelizable training and efficient recurrent inference. RetNet achieves this through a mathematically elegant equivalence between a decayed recurrence and a masked parallel form. RWKV achieves it through exponential key weighting in an EMA-style recurrence with learned gating. Both represent mature, production-ready architectures with open-source models and growing adoption in latency-sensitive and on-device AI applications.
