---
title: "Speculative Streaming: Low-Latency LLM Inference Without a Draft Model"
description: Learn how Speculative Streaming accelerates LLM inference by generating and verifying multiple tokens in a single forward pass using self-speculation — no separate draft model required.
---

Latency is the dominant constraint in interactive LLM deployments. Users experience token-by-token streaming: the first token appears after the prefill pass, and subsequent tokens appear as each autoregressive step completes. For large models, each step takes tens to hundreds of milliseconds — too slow for fluid interaction.

**Speculative decoding** (introduced in 2023 by Leviathan et al. and Chen et al.) accelerates this by using a small, fast **draft model** to generate multiple candidate tokens, then verifying them with the large target model in a single forward pass. Accepted tokens cost nearly nothing; rejected tokens fall back to the target model's distribution.

The limitation: speculative decoding requires maintaining a separate draft model aligned with the target model's distribution. For self-hosted deployments, this doubles memory requirements and operational complexity.

**Speculative Streaming** (2024) eliminates the draft model entirely, using the target model itself to speculate via a Medusa-style multi-token head approach — but integrated tightly with streaming generation for ultra-low first-token latency.

## Background: Standard Speculative Decoding

In classic speculative decoding:

1. **Draft phase:** A small model $q$ generates $k$ candidate tokens $\tilde{x}_1, \dots, \tilde{x}_k$ autoregressively
2. **Verify phase:** The target model $p$ evaluates all $k$ candidates in a single batched forward pass
3. **Accept/reject:** Each candidate $\tilde{x}_i$ is accepted with probability $\min(1, p(\tilde{x}_i) / q(\tilde{x}_i))$; the first rejection terminates acceptance and samples a corrected token from the residual distribution

The speedup comes from **verify being much cheaper than sequential autoregressive generation** for the large model: the forward pass for $k$ tokens takes roughly the same time as for 1 token (when memory-bandwidth-bound, which is typical).

## Self-Speculation: Draft with the Target Model Itself

Self-speculation uses early layers of the target model as the draft. The key insight: for many inputs, the top-1 predictions of the first few transformer layers are identical to the final predictions of the full model.

**Approaches to self-speculation:**

1. **Layer skipping:** Generate draft tokens by running only the first $L_{draft}$ layers, then verify with the full $L_{full}$ layers. Requires early exit heads trained on the partial residual stream.

2. **Medusa heads:** Attach $k$ additional lightweight prediction heads to the final hidden state — each head predicts one token ahead of the standard head. Used in Medusa and EAGLE architectures.

3. **Speculative Streaming's approach:** Generate draft tokens during the **prefill phase** using a combination of attention pattern reuse and early-exit heads, then stream verified tokens without a separate drafting step during decoding.

## Speculative Streaming Architecture

Speculative Streaming introduces two key components:

### 1. Prefill-Time Speculation

During the prefill pass (processing the input prompt), the model computes KV caches for all prompt tokens and also generates speculative draft continuations for the first $k$ output tokens **for free** by inspecting attention patterns and early-layer logits.

The rationale: the final token of the prompt attends to the full context during prefill. Its hidden state already encodes what the model "wants" to say next — running a lightweight head on this hidden state generates a high-quality draft without an extra forward pass.

### 2. Streaming Verification Loop

After prefill, the standard autoregressive loop is replaced with a streaming verification loop:

```
For each generation step:
  1. Verify the k speculative tokens generated at the previous step
  2. Accept all tokens up to the first mismatch
  3. Generate k new speculative tokens for the next step
  4. Stream all accepted tokens to the client immediately
```

This overlaps speculation with streaming: clients receive a burst of accepted tokens, then a brief pause while the next speculative batch is verified, rather than a steady drip of single tokens.

## Implementation Sketch

A simplified implementation using a Medusa-style multi-head setup:

```python
class SpeculativeStreamingModel(nn.Module):
    def __init__(self, base_model, num_speculative_heads=4, vocab_size=32000):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        # Additional heads for speculating k tokens ahead
        self.spec_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_speculative_heads)
        ])

    def speculate(self, hidden_states):
        """
        Generate k draft tokens from the current hidden state.
        hidden_states: (batch, seq_len, hidden_size)
        Returns: list of k draft token tensors, each (batch, vocab_size)
        """
        last_hidden = hidden_states[:, -1, :]  # (batch, hidden_size)
        drafts = [head(last_hidden) for head in self.spec_heads]
        return drafts  # logits for positions +1, +2, ..., +k

    def verify_and_accept(
        self, base_logits, draft_logits_list, draft_token_ids
    ):
        """
        Accept draft tokens that match the base model distribution.
        Returns accepted token ids and the index of first rejection.
        """
        accepted = []
        for i, (draft_logit, draft_id) in enumerate(
            zip(draft_logits_list, draft_token_ids)
        ):
            # Sample from base model at this position
            base_token = base_logits[i].argmax(dim=-1)
            if base_token == draft_id:
                accepted.append(draft_id)
            else:
                # Reject: use base model's token instead
                accepted.append(base_token)
                break
        return accepted
```

Note: production implementations use tree-structured speculation (multiple candidate sequences in parallel) and probabilistic acceptance (not just greedy matching) for correctness with non-greedy sampling.

## Comparison: Speculative Streaming vs. Other Approaches

| Approach | Draft Model Required | Memory Overhead | Typical Speedup | TTFT Impact |
|---|---|---|---|---|
| Standard speculative decoding | Yes (separate model) | 1.5–2× | 2–3× | Slight increase |
| Medusa | No (extra heads only) | ~5% | 1.5–2.5× | Negligible |
| EAGLE | No (single-layer draft) | ~15% | 2–3× | Negligible |
| Speculative Streaming | No | ~5–10% | 1.8–2.8× | Reduced |
| Prompt Lookup Decoding | No | None | 1.5–2× (copy-heavy) | None |

TTFT = Time To First Token

Speculative Streaming's unique advantage is reducing TTFT by piggybacking speculative drafting onto the prefill computation rather than adding a separate step.

## Tree-Structured Speculation

Rather than speculating a single sequence of $k$ tokens, advanced implementations speculate a **tree of candidate continuations**. The tree allows the verifier to accept longer sequences by exploring multiple possibilities:

```
              [token A]
             /         \
        [token B]    [token C]
        /      \
  [token D] [token E]
```

The verifier runs a batched forward pass over the entire tree's token set (with appropriate causal masking) and accepts the longest valid path.

This is how **Medusa**, **EAGLE**, and **Hydra** achieve their highest speedups — by encoding the tree in the attention mask matrix and processing all nodes in one forward pass.

## Practical Deployment Considerations

**When speculative streaming helps most:**
- Long outputs (more tokens = more opportunities for speculation to pay off)
- Repetitive or formulaic content (code boilerplate, structured data, templated text)
- Greedy or near-greedy decoding (temperature near 0, where draft acceptance is highest)
- Memory-bandwidth-bound inference (small batch sizes on large models)

**When it helps least:**
- Creative, high-temperature generation (frequent rejections make speculation inefficient)
- Very short outputs (overhead of speculation setup exceeds savings)
- Compute-bound inference (large batch sizes where arithmetic intensity is already high)

**Integration with other optimizations:** Speculative streaming composes well with:
- Quantization (the speculative heads can run in 4-bit while the verifier runs in 8-bit)
- Flash Attention (the verification forward pass benefits from efficient attention)
- Continuous batching (draft tokens can be verified across multiple requests)

## Open-Source Implementations

- **[EAGLE](https://github.com/SafeAILab/EAGLE)** — state-of-the-art self-speculation with a single-layer draft model
- **[Medusa](https://github.com/FasterDecoding/Medusa)** — multi-head self-speculation
- **[vLLM](https://github.com/vllm-project/vllm)** — native speculative decoding support including draft-model and Medusa modes
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — speculative decoding with draft model support via `--model-draft`
- **[text-generation-inference (TGI)](https://github.com/huggingface/text-generation-inference)** — Medusa and speculative decoding support

Speculative streaming and related self-speculation methods represent the practical frontier of LLM inference optimization for interactive applications — delivering near-human-speed response latency for conversational, coding, and document generation use cases.
