---
title: "Attention Mechanisms Explained: From Bahdanau to Multi-Head Attention"
description: "Understand how attention revolutionized sequence modeling — from the original attention paper to self-attention, multi-head attention, and the transformer architecture that powers modern AI."
---

Attention is one of the most important ideas in deep learning. It enabled the transformer architecture, which in turn enabled GPT, BERT, and virtually all modern language models. Understanding attention from first principles gives you the foundation to understand why LLMs work the way they do.

## The Problem Attention Solves

Before attention, sequence models had a fundamental limitation: they processed the entire input into a fixed-size hidden state, then generated the output from that single representation.

For a long document being summarized, this is problematic. The summary needs to focus on different parts of the document for different aspects. A single vector can't capture everything.

## Bahdanau Attention (2014)

The original attention paper (Bahdanau et al., 2014) introduced a mechanism for neural machine translation. The key insight: at each step of generating the output, the model should "attend" to the most relevant parts of the input.

### How It Works

Given encoder hidden states (h₁, h₂, ..., hₙ) and a decoder state (sᵢ), attention computes:

1. **Attention scores**: How relevant is each encoder state to the current decoder state?
   ```
   eᵢⱼ = vᵀ tanh(W₁ hᵢ + W₂ sⱼ)
   ```

2. **Attention weights**: Normalize scores to probabilities (softmax):
   ```
   αᵢⱼ = exp(eᵢⱼ) / Σₖ exp(eᵢₖ)
   ```

3. **Context vector**: Weighted sum of encoder states:
   ```
   cᵢ = Σⱼ αᵢⱼ hⱼ
   ```

4. **Attend and predict**: Combine context with decoder state to generate output.

This was revolutionary because it allowed the model to dynamically focus on different parts of the input for each output token.

## Self-Attention

**Self-attention** applies attention within a single sequence — each position attends to all other positions in the sequence.

For a sequence of n tokens, self-attention computes a new representation for each token by combining information from all other tokens:

```python
# Simplified self-attention
def self_attention(Q, K, V):
    # Q, K, V are (n, d_k) matrices
    scores = Q @ K.T  # (n, n) attention scores
    weights = softmax(scores / sqrt(d_k))  # (n, n) attention weights
    output = weights @ V  # (n, d_v) new representations
    return output
```

Key insight: each token's new representation depends on all other tokens, regardless of their distance in the original sequence.

## Scaled Dot-Product Attention

The specific form of attention used in transformers:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

Why scale by √d_k? Without scaling, dot products grow with dimension size, pushing softmax into regions of extreme gradients.

## Multi-Head Attention

Running attention multiple times in parallel captures different relationship types:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Multiple attention heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V):
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, heads, seq, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention per head
        head_outputs = []
        for i in range(self.num_heads):
            head_out = attention(Q[:, i], K[:, i], V[:, i])
            head_outputs.append(head_out)
        
        # Concatenate heads and project
        concat = torch.cat(head_outputs, dim=-1)
        return self.W_O(concat)
```

Each head might learn to attend to:
- One head: syntactic relationships (verb → subject).
- Another head: semantic relationships (cat → animal).
- Another: long-range dependencies.

## Positional Encoding

Attention is permutation-invariant — it has no notion of order. Positional encodings inject position information:

```python
# Sinusoidal positional encoding
def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

Alternative approaches: learned position embeddings, relative position representations, rotary position embeddings (RoPE).

## Attention Patterns in Modern LLMs

### Full Self-Attention
Every token attends to every other token. Quadratic complexity O(n²) in sequence length. Used in most LLMs for short contexts.

### Sliding Window Attention
Each token attends only to a fixed window (e.g., 4096 tokens) of neighbors. Linear complexity. Used in Mistral and Mamba.

### Sparse Attention
Various patterns that reduce complexity:
- **Strided attention**: Attend to every k-th token.
- **Random attention**: Attend to random positions.
- **Block-sparse**: Combines window and global attention.

### GQA (Grouped Query Attention)
Shares key-value projections across query heads. Reduces memory for KV cache. Used in LLaMA-2 and many efficient models.

## Efficient Attention Variants

| Method | Complexity | Key Idea |
|--------|------------|----------|
| FlashAttention | O(nd) | IO-aware tiling in SRAM |
| Linear Attention | O(n) | Kernelized attention approximation |
| Performer | O(n) | Random feature approximation |
| Ring Attention | Distributed | Process in blocks across devices |

## Why Attention Works So Well

1. **Direct access**: Each token can directly incorporate information from any other token.
2. **Parallelizable**: All positions attend in parallel (unlike RNNs).
3. **Expressive**: The attention weight matrix can represent complex relationships.
4. **Interpretable**: Attention weights provide some interpretability.

## The Limits of Attention

- **Quadratic cost**: Full attention doesn't scale to very long sequences.
- **Fixed context**: Models can only attend to what's in the context window.
- **No inherent structure**: Attention treats all positions equally, missing inductive biases that CNNs capture.

Despite these limitations, attention remains the backbone of modern deep learning. The transformer architecture built on attention has enabled the rapid progress in AI that defines our current era.