---
title: Perceiver IO
description: Understand the Perceiver IO architecture — a general-purpose transformer that handles arbitrary input and output modalities through cross-attention to a compact latent space, enabling a single model to process images, audio, point clouds, and multimodal data without task-specific inductive biases.
---

Standard deep learning architectures are modality-specific: CNNs process grids, Transformers process sequences, GNNs process graphs. Each bakes in assumptions about input structure that limit generality. **Perceiver IO** (Jaegle et al., 2021) is a general-purpose architecture that handles arbitrary input and output modalities by mapping everything through cross-attention into a compact latent space — and then decoding from that latent space to outputs of any desired shape.

## The Core Problem

Applying a standard Transformer directly to large inputs is prohibitive. For a $H \times W$ image with $C$ channels, full self-attention costs $\mathcal{O}((HWC)^2)$ in memory and compute. A 224×224 RGB image has 150,528 input elements — squaring that is completely impractical.

Earlier architectures solve this by restricting the receptive field (CNNs) or downsampling aggressively (patch-based ViT). Perceiver IO takes a different approach: don't run self-attention over inputs at all — cross-attend from a small, fixed-size latent array to the inputs instead.

## Architecture Overview

Perceiver IO has three stages:

```text
Input Array (arbitrary size M × C_in)
         │
         │  Cross-Attention (Input → Latent)
         ▼
Latent Array (fixed size N × D, N << M)
         │
         │  L × Transformer Blocks (self-attention in latent space)
         ▼
Latent Array (N × D)
         │
         │  Cross-Attention (Latent → Output Queries)
         ▼
Output Array (arbitrary size O × C_out)
```

The key insight: expensive attention is only performed between the small latent array and external inputs/outputs. Self-attention stays inside the latent — cheap because $N \ll M$.

## Stage 1: Input Encoding via Cross-Attention

The latent array $\mathbf{Z} \in \mathbb{R}^{N \times D}$ is a learned parameter (initialized randomly, trained end-to-end). It queries the entire input array $\mathbf{X} \in \mathbb{R}^{M \times C}$ via cross-attention:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

where:

- $\mathbf{Q} = \mathbf{Z}\mathbf{W}_Q$ — queries from the latent array (shape $N \times d_k$)
- $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, $\mathbf{V} = \mathbf{X}\mathbf{W}_V$ — keys and values from the input array (shape $M \times d_k$)

This cross-attention is $\mathcal{O}(NM)$ — linear in $M$ since $N$ is fixed — compared to $\mathcal{O}(M^2)$ for full self-attention. For $N = 512$ and $M = 150{,}528$, this is a ~300× saving.

## Stage 2: Latent Processing via Self-Attention

The encoded latent $\mathbf{Z}' \in \mathbb{R}^{N \times D}$ is processed by $L$ standard Transformer blocks (multi-head self-attention + FFN):

$$\mathbf{Z}^{(l+1)} = \text{TransformerBlock}(\mathbf{Z}^{(l)})$$

Self-attention within the latent is $\mathcal{O}(N^2)$ — cheap because $N$ is small (typically 512–2048). This is where the model does most of its reasoning, unconstrained by input size.

## Stage 3: Output Decoding via Cross-Attention

To produce outputs of arbitrary shape, Perceiver IO introduces **output queries** $\mathbf{Y} \in \mathbb{R}^{O \times D}$:

- For classification: $O = 1$ (single class output)
- For pixel-level output: $O = H \times W$ (one query per pixel)
- For optical flow: $O = H \times W$ (one query per spatial location)
- For language modeling: $O = \text{sequence length}$

The output queries attend to the final latent:

$$\hat{\mathbf{Y}} = \text{CrossAttention}(\mathbf{Y}_{\text{query}}, \mathbf{Z}^{(L)}, \mathbf{Z}^{(L)})$$

Output queries can be learned parameters, position embeddings, or task-specific embeddings — making the architecture fully general.

## Position Encodings

Without explicit structure, Perceiver IO uses **Fourier feature positional encodings**. For a position $p$ in a $d$-dimensional space:

$$\gamma(p) = \left[\sin(2\pi f_1 p_1), \cos(2\pi f_1 p_1), \ldots, \sin(2\pi f_k p_d), \cos(2\pi f_k p_d)\right]$$

These are concatenated to the input features and output queries, providing spatial context without baking in grid structure. The same encoding scheme works for 1D audio, 2D images, 3D point clouds, and 4D video.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, latent_dim, input_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(input_dim, latent_dim)
        self.v_proj = nn.Linear(input_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, queries, context):
        B, N, D = queries.shape
        _, M, _ = context.shape

        Q = self.q_proj(queries).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class PerceiverIO(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=512,
        num_latents=512,
        num_heads=8,
        num_layers=6,
        output_dim=None,
    ):
        super().__init__()
        output_dim = output_dim or latent_dim

        # Learned latent array
        self.latent = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Input cross-attention
        self.input_cross_attn = MultiHeadCrossAttention(latent_dim, input_dim, num_heads)
        self.input_norm = nn.LayerNorm(latent_dim)

        # Latent self-attention stack
        self.latent_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Output cross-attention
        self.output_cross_attn = MultiHeadCrossAttention(output_dim, latent_dim, num_heads)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, inputs, output_queries):
        # inputs: (B, M, input_dim)
        # output_queries: (B, O, output_dim)
        B = inputs.shape[0]

        # Expand latent for batch
        z = self.latent.unsqueeze(0).expand(B, -1, -1)

        # Encode: cross-attend latent → input
        z = self.input_norm(z + self.input_cross_attn(z, inputs))

        # Process: self-attend in latent space
        for layer in self.latent_layers:
            z = layer(z)

        # Decode: cross-attend output queries → latent
        out = self.output_norm(output_queries + self.output_cross_attn(output_queries, z))
        return out
```

## Handling Multiple Modalities

Perceiver IO handles multimodal inputs by concatenating all modalities along the sequence dimension, with modality-specific positional/type embeddings:

```python
def prepare_multimodal_input(image, audio, text_tokens):
    # image: (B, H*W, C_img)  — flattened spatial positions
    # audio: (B, T, C_audio)  — temporal positions
    # text_tokens: (B, L, C_text)  — token positions

    # Project each modality to common input_dim
    image_emb = image_proj(image) + image_pos_emb    # (B, H*W, D)
    audio_emb = audio_proj(audio) + audio_pos_emb    # (B, T, D)
    text_emb  = text_proj(text_tokens) + text_pos_emb  # (B, L, D)

    # Concatenate along sequence axis — Perceiver handles any length
    combined = torch.cat([image_emb, audio_emb, text_emb], dim=1)  # (B, H*W+T+L, D)
    return combined
```

No architectural changes are needed — the cross-attention encoder handles any concatenated sequence.

## Applications

### Image Classification

For ImageNet classification, $O = 1$ classification query attends to the final latent:

- Input: 50,176 flattened pixel features (224×224) + 2D Fourier position encodings
- Latent: 512 elements × 1024-dimensional
- Output query: single learned class token
- Result: competitive with ViT on ImageNet without spatial inductive biases

### Optical Flow

Perceiver IO achieves strong results on Sintel and KITTI optical flow benchmarks:

- Input: two stacked video frames + position encodings
- Output queries: one per pixel (position-encoded)
- Output: 2D flow vectors per pixel
- Same architecture, different output queries — zero architectural changes needed

### StarCraft II Agent (GATO)

DeepMind's GATO (a generalist agent) uses Perceiver IO-style architectures to process mixed observations (images, discrete tokens, continuous scalars) and produce actions for multiple games and tasks from a single model.

## Comparison with Alternatives

| Aspect | ViT (Patch) | CNN | Perceiver IO |
| --- | --- | --- | --- |
| Input type | Fixed grid (images) | Fixed grid | Arbitrary |
| Output type | Fixed | Fixed | Arbitrary |
| Input complexity | $\mathcal{O}(P^2)$ patches | $\mathcal{O}(HW)$ | $\mathcal{O}(NM)$ |
| Multimodal | Requires adapters | Requires adapters | Native |
| Spatial inductive bias | Patch partitioning | Strong (conv) | None |
| Latent size | Tied to input | Tied to input | Fixed, decoupled |

## Limitations

Perceiver IO has several trade-offs to consider:

- **Slower convergence**: without spatial inductive biases, Perceiver requires more data or training steps to match specialized architectures on vision tasks
- **Positional encoding sensitivity**: the Fourier feature scheme must be carefully calibrated per modality; irregular or unordered inputs (e.g., graphs) need custom encodings
- **Latent bottleneck**: the fixed latent size $N$ bounds model capacity; for tasks requiring fine-grained spatial correspondence, $N$ must be large (increasing cost)
- **Attention weight interpretability**: cross-attention from a small latent to large inputs produces attention patterns that are harder to interpret than self-attention over patches

## Summary

Perceiver IO unifies sequence-to-sequence processing across arbitrary modalities through three cross-attention operations:

- **Input encoding**: a fixed latent array cross-attends to inputs of any size and modality
- **Latent processing**: cheap self-attention in the small latent space performs reasoning
- **Output decoding**: output queries of arbitrary shape cross-attend to the latent

This design decouples model capacity from input/output size, making it possible to apply a single architecture to images, audio, video, point clouds, and multimodal combinations without task-specific structural assumptions — a step toward genuinely general-purpose neural networks.
