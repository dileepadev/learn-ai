---
title: Slot Attention
description: A guide to Slot Attention, an object-centric learning mechanism that decomposes visual scenes into a variable number of slots through iterative competitive attention.
---

# Slot Attention

Slot Attention is an object-centric learning mechanism proposed by Locatello et al. (2020) at Google Research. It decomposes unstructured visual inputs into a **fixed set of object slots** through an iterative competitive attention process, enabling compositional scene understanding without object-level supervision.

## Motivation: Object-Centric Representations

Most deep learning models treat images as holistic arrays, learning distributed representations that entangle multiple objects. Object-centric learning aims to produce **modular representations** where each slot independently encodes a distinct entity, enabling:

- Systematic generalization to novel object combinations
- Sample-efficient learning by reusing object-level knowledge
- Interpretable latent spaces with semantic structure
- Downstream reasoning, planning, and causal inference over objects

## The Slot Attention Mechanism

### Inputs and Outputs

- **Input**: $N$ feature vectors $X \in \mathbb{R}^{N \times D}$ from a CNN or ViT encoder
- **Slots**: $K$ randomly initialized vectors $S \in \mathbb{R}^{K \times D}$, where $K \ll N$
- **Output**: $K$ updated slot vectors, each capturing one scene entity

### Iterative Competitive Attention

Slot Attention runs for $T$ iterations (typically 3–7). Each iteration consists of:

**Step 1 — Attention logits** (slots compete for input features):

$$A_{ij} = \frac{\exp\!\left(\frac{1}{\sqrt{D}} k(X_i) \cdot q(S_j)\right)}{\sum_{j'} \exp\!\left(\frac{1}{\sqrt{D}} k(X_i) \cdot q(S_{j'})\right)}$$

where $k(\cdot)$ and $q(\cdot)$ are linear projections of inputs and slots. The softmax is over **slots** (not inputs), forcing competitive binding.

**Step 2 — Normalized weighted mean** (aggregation):

$$W_j = \frac{\sum_i A_{ij} \cdot v(X_i)}{\sum_{i'} A_{i'j} + \epsilon}$$

**Step 3 — Slot update via GRU:**

$$S_j^{t+1} = \text{GRU}(W_j^t,\; S_j^t)$$

followed by a residual MLP for further refinement. After $T$ iterations each slot converges to represent a distinct image region or object.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # Learnable Gaussian for slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, N, D = inputs.shape
        slots = self.slots_mu + self.slots_sigma * torch.randn(
            B, self.num_slots, D, device=inputs.device
        )

        inputs_n = self.norm_inputs(inputs)
        k = self.to_k(inputs_n)   # (B, N, D)
        v = self.to_v(inputs_n)   # (B, N, D)

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))  # (B, K, D)

            # Softmax over slots — competitive binding
            dots = torch.einsum("bnd,bkd->bnk", k, q) * self.scale  # (B, N, K)
            attn = dots.softmax(dim=2)  # (B, N, K)

            # Normalize over inputs for weighted mean
            attn_w = attn / (attn.sum(dim=1, keepdim=True) + self.eps)
            updates = torch.einsum("bnk,bnd->bkd", attn_w, v)  # (B, K, D)

            slots = self.gru(
                updates.flatten(0, 1),
                slots_prev.flatten(0, 1),
            ).reshape(B, self.num_slots, D)

            slots = slots + self.mlp(slots)

        return slots   # (B, K, D)


# Example
encoder_out = torch.randn(4, 64 * 64, 64)  # 4 images, 64x64 spatial, 64-d features
model = SlotAttention(num_slots=7, dim=64, iters=3)
slots = model(encoder_out)
print(slots.shape)   # (4, 7, 64)
```

## Object Discovery Pipeline

The full unsupervised object discovery model wraps slot attention in an encoder-decoder:

```
Image → CNN Encoder → Slot Attention → K Slot Vectors
                                              ↓
                              Spatial Broadcast Decoder × K
                                              ↓
                         K (RGB + alpha) Reconstructions → Composite
                                              ↓
                               Pixel Reconstruction Loss
```

The **spatial broadcast decoder** tiles each slot vector across a spatial grid and decodes it to an RGBA image. The composite reconstruction is:

$$\hat{x} = \sum_{k=1}^{K} m_k \odot \hat{x}_k, \qquad \sum_k m_k = 1$$

Training minimizes pixel reconstruction loss with **zero object-level annotations**.

## Spatial Broadcast Decoder

```python
class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim: int, img_size: int = 64):
        super().__init__()
        self.img_size = img_size
        # Learnable positional grid
        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid", torch.stack([grid_x, grid_y], dim=-1))

        self.conv = nn.Sequential(
            nn.Conv2d(slot_dim + 2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 4, 1),   # RGB + alpha
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        B, K, D = slots.shape
        H = W = self.img_size
        # Tile each slot across spatial grid
        s = slots.reshape(B * K, D, 1, 1).expand(-1, -1, H, W)
        grid = self.grid.permute(2, 0, 1).unsqueeze(0).expand(B * K, -1, -1, -1)
        x = torch.cat([s, grid], dim=1)
        return self.conv(x).reshape(B, K, 4, H, W)
```

## Extensions and Variants

### SLATE — Slot Attention with Transformer Decoder

Replaces the spatial broadcast decoder with an autoregressive dVAE-based transformer, enabling discrete token-based reconstruction and stronger generalization to complex scenes.

### DINOSAUR

Combines slot attention with DINO self-supervised ViT features, achieving strong zero-shot segmentation on natural images without any pixel-level annotations.

### SlotFormer

Adds a transformer over slot sequences across video frames — slots persist over time and interact via attention, enabling video prediction and planning.

### Object-Centric DALL·E (OCVAE)

Integrates slot representations with generative image models to enable compositional image generation conditioned on individual object descriptions.

## Why Competitive Softmax?

The key algorithmic insight is softmax **over slots** rather than over spatial positions:

- Standard attention (over positions): each query attends to all positions — representations entangle multiple objects
- Slot attention (over slots): each position is claimed by the most appropriate slot — forces **winner-take-all** binding

This competition is what drives each slot to specialize on a distinct entity.

## Comparison with Related Methods

| Method | Supervision | Variable K | Video | Complex Scenes |
|---|---|---|---|---|
| Slot Attention | None | ❌ | ❌ | Limited |
| SLATE | None | ❌ | ❌ | Moderate |
| SlotFormer | None | ❌ | ✅ | Moderate |
| DETR | Box labels | ✅ | ❌ | Strong |
| Mask R-CNN | Mask labels | ✅ | ❌ | Excellent |

## Challenges

- **Fixed $K$**: slots must be pre-specified; excess slots bind to background, too few causes object merging
- **Real-world scenes**: works best on synthetic benchmarks (CLEVR, Multi-dSprites); complex natural scenes remain difficult
- **Slot collapse**: competitive softmax can cause all slots to converge to the same object during training
- **Scalability**: iterative attention over high-resolution feature maps is computationally costly

## Applications

- **Robotics**: object-centric state representations for manipulation planning and model-based RL
- **Visual QA**: compositional scene graphs inferred from slot bindings
- **World models**: slot-based dynamics models for environment simulation
- **Video decomposition**: tracking and segmenting objects in video without annotation

## Summary

Slot Attention provides a differentiable, unsupervised mechanism for decomposing visual scenes into object-centric representations. Its competitive softmax over slots — rather than positions — enables $K$ distinct entities to bind to disjoint image regions. The resulting slot vectors support compositional generalization and form the basis of increasingly capable object-centric world models, connecting to broader research in systematic generalization and neuro-symbolic AI.
