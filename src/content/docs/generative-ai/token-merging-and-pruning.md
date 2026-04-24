---
title: Token Merging and Token Pruning
description: Learn how token merging (ToMe) and token pruning techniques reduce the computational cost of transformer inference by dynamically eliminating redundant tokens — enabling faster vision transformers and LLMs without significant accuracy loss.
---

**Token merging** and **token pruning** are inference efficiency techniques that reduce the number of tokens a transformer processes by identifying and eliminating redundant or uninformative tokens at runtime. Since transformer attention complexity scales as $O(n^2)$ in sequence length, reducing $n$ (the number of active tokens) produces large computational savings — often achieving 30–50% speedup with minimal accuracy degradation.

These techniques complement other efficiency methods: while quantization reduces the cost per operation and speculative decoding reduces the number of model passes, token reduction directly reduces the volume of computation in every attention layer.

## Why Tokens Are Redundant

Transformers process sequences as collections of discrete tokens. In many contexts, significant redundancy exists:

**In vision transformers (ViTs)**: Image patches corresponding to a homogeneous background, sky, or solid color contain very similar information — attending to all of them independently wastes computation. Studies show that up to 75% of ViT image patches can be discarded without significant accuracy loss on ImageNet classification.

**In language models**: Long documents often contain repeated context, padding, or filler tokens. At deeper layers, many token representations converge to carry similar information — particularly common words and syntactic tokens that have already been processed.

**In multimodal models**: Vision-language models process hundreds of image tokens alongside text — image tokens are often far more redundant than text tokens.

## Token Merging (ToMe)

**Token Merging** (Bolya et al., 2022, "Token Merging: Your ViT But Faster") is an elegant, training-free technique that reduces token count by combining similar tokens rather than discarding them — preserving information that would be lost by pruning.

### Core Algorithm

At each transformer layer, ToMe:

1. **Computes similarity** between all pairs of tokens using the keys $K$ from the self-attention computation (which are already computed and carry semantic meaning).

2. **Bipartite matching**: Partitions tokens into two sets $A$ and $B$ of equal size, then finds a matching between sets that pairs the most similar tokens across sets (using a greedy bipartite matching algorithm that runs in $O(n)$ time).

3. **Merges matched pairs**: Each matched pair $(a, b)$ is replaced by their weighted average $(a + b) / 2$ (or a weighted average based on how many tokens have previously been merged into each). The merged token count decreases by $r$ per layer, where $r$ is a configurable merge ratio.

4. **Unmerging at output**: At the final layer, merged tokens are unmerged back to their original positions using the stored merging information — enabling per-token output predictions (e.g., for segmentation or detection) when needed.

```python
# Pseudocode: ToMe token reduction at one attention layer
def apply_tome(keys, values, r):
    """Merge r token pairs per layer based on key similarity."""
    n = keys.shape[0]
    a, b = keys[:n//2], keys[n//2:]  # Split into two sets
    
    # Compute cosine similarity between sets A and B
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    scores = a_norm @ b_norm.T  # [n//2, n//2]
    
    # Greedy bipartite matching: for each a-token, find most similar b-token
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(descending=True)[:r]
    
    # Merge top-r pairs by averaging
    merged_a = a[edge_idx]
    merged_b = b[node_idx[edge_idx]]
    merged = (merged_a + merged_b) / 2
    
    # Unmatched tokens pass through unchanged
    unmatched_a = a[~edge_idx]
    return torch.cat([unmatched_a, merged], dim=0)
```

### ToMe Results

On ViT-H with a merge ratio of $r = 8$ per layer (merging 8 token pairs per layer across all 32 layers — eliminating a large fraction of tokens over the depth of the network):

- **ImageNet top-1 accuracy**: Drops from 88.5% to 87.5% (−1%).
- **Throughput improvement**: 2× faster than unmodified ViT-H.
- **No retraining required**: ToMe is applied at inference time to any pretrained ViT.

With a small amount of fine-tuning on the merged model, the accuracy drop can be fully recovered while maintaining the throughput gain.

### ToMe for Text Generation

Applied to LLMs, ToMe operates on the hidden states between attention layers, merging similar token representations. The challenge in language models is that token order carries grammatical meaning — unlike image patches, nearby text tokens are not necessarily similar in content. ToMe for LLMs requires careful analysis of which tokens to merge across layers and at which positions.

## Token Pruning

**Token pruning** discards tokens rather than merging them — a more aggressive reduction that loses information but incurs lower computational overhead (no merging computation, and fewer tokens enter future layers without the overhead of tracking merged token metadata).

### Attention-Score-Based Pruning

The self-attention weights reveal which tokens attend strongly to others — tokens that are rarely attended to (low total attention received from other tokens) contribute little to downstream computation and are candidates for pruning:

$$\text{importance}(i) = \sum_j \alpha_{ji}$$

where $\alpha_{ji}$ is the attention weight from token $j$ to token $i$. Tokens with low aggregate importance are pruned.

**SpAtten** (Wang et al., 2021) implements cascade token pruning based on cumulative attention weight thresholds — progressively pruning more tokens in deeper layers where many tokens have already been determined to be uninformative.

### Learned Token Importance Predictors

Rather than using heuristic importance scores, **learned pruning** trains a lightweight predictor (often a small linear classifier on top of hidden states) to predict whether each token should be retained at each layer:

- **LTP** (Learned Token Pruning, Kim & Cho, 2021): Trains a small gate network that predicts per-token keep/discard decisions during training, with a budget constraint regularizing the total number of kept tokens.
- **TR-BERT** (Token Reduction BERT): Adds a policy network that makes layer-by-layer pruning decisions — tokens that are confidently classified early exit deeper layers.

### Lazy Propagation

A variant that doesn't discard tokens but skips computation for less important tokens:

- Tokens flagged as low-importance bypass certain attention and FFN computations in later layers.
- Their representations are propagated from their last active layer — a cheap approximation that maintains positional integrity for remaining tokens.

## Dynamic Adaptive Computation

Both merging and pruning can be made **input-adaptive** — applying more reduction on easy inputs and less on hard ones:

**Threshold-based adaptive pruning**: Instead of a fixed number of tokens to prune per layer, prune all tokens below an importance threshold. This produces variable-length sequences with more tokens for complex inputs and fewer for simple ones.

**Early exit with token selection**: Different tokens exit at different layers — simple, high-confidence tokens stop at shallow layers; complex or ambiguous tokens continue through all layers. This implements a form of **adaptive depth** alongside token count reduction.

## Token Reduction in Multimodal Models

Multimodal large language models (MLLMs) that process images alongside text typically encode images as hundreds of tokens (e.g., LLaVA encodes 256 image tokens; some models use 2048+). These image tokens often contain significant redundancy:

### Visual Token Compression

**LLaVA-PruMerge** selectively merges and prunes image tokens before feeding them to the language model:

1. Compute attention scores between the CLS token and image patch tokens — patches strongly attended to by CLS are kept as important.
2. Prune unattended image tokens; merge similar attended tokens.
3. Pass the compressed set of image tokens to the LLM.

This reduces image tokens from 256 to ~50 with minimal VQA performance degradation — enabling substantially lower latency for visual understanding tasks.

**Q-Former** (BLIP-2) and **Perceiver Resampler** (Flamingo) use cross-attention to compress image tokens into a fixed-size bottleneck (32 or 64 tokens) before passing to the LLM — a learnable form of token compression trained end-to-end.

## Comparison of Reduction Strategies

| Strategy | Information loss | Compute savings | Requires training |
| --- | --- | --- | --- |
| Token pruning | High (discards tokens) | High | Optional |
| Token merging (ToMe) | Low (averages tokens) | Medium-high | No |
| Lazy propagation | Low (reuses states) | Medium | No |
| Learned pruning | Variable | High | Yes |
| Q-Former / Resampler | Variable | Very high | Yes |

## Combining with Other Efficiency Techniques

Token merging and pruning compose naturally with other inference efficiency methods:

- **With quantization**: Reduce token count (fewer operations) and operation cost (cheaper arithmetic) simultaneously.
- **With speculative decoding**: Token merging in the draft model reduces the cost of generating draft tokens.
- **With FlashAttention**: FlashAttention's memory efficiency scales with sequence length — shorter sequences from token reduction make FlashAttention even more beneficial.
- **With KV cache optimization**: Fewer tokens means smaller KV caches — reducing memory pressure and enabling longer context windows within a given memory budget.

Token reduction is particularly impactful for real-time and on-device inference, where compute and memory are severely constrained — enabling capable vision and language models to run on hardware that would otherwise be insufficient.
