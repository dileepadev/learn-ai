---
title: Vision Mamba
description: Explore Vision Mamba — state space model architectures adapted for computer vision. Covers selective state spaces, cross-scan mechanisms, VMamba and Vim architectures, linear complexity scaling, and comparisons with Vision Transformers for high-resolution image understanding tasks.
---

**Vision Mamba** refers to a family of vision backbone architectures that replace the quadratic self-attention of Vision Transformers (ViT) with **Selective State Space Models (SSMs)** — enabling linear-complexity sequence modeling while retaining competitive representational power. The core motivation is that ViT's $O(N^2)$ attention over image patches becomes prohibitively expensive at high resolution, while SSMs can process sequences in $O(N)$ time and memory.

## Background: State Space Models and Mamba

A continuous-time SSM maps an input signal $x(t)$ to output $y(t)$ through a hidden state $h(t)$:

$$\dot{h}(t) = \mathbf{A} h(t) + \mathbf{B} x(t)$$
$$y(t) = \mathbf{C} h(t)$$

where $\mathbf{A} \in \mathbb{R}^{N \times N}$, $\mathbf{B} \in \mathbb{R}^{N \times 1}$, $\mathbf{C} \in \mathbb{R}^{1 \times N}$ are learnable matrices. Discretized for sequences with step size $\Delta$:

$$h_k = \bar{\mathbf{A}} h_{k-1} + \bar{\mathbf{B}} x_k, \quad y_k = \mathbf{C} h_k$$

**Mamba** (Gu & Dao, 2023) introduces the key innovation of **input-dependent (selective) state space** matrices — $\mathbf{B}$, $\mathbf{C}$, and $\Delta$ become functions of the input token rather than fixed parameters. This selectivity allows the model to filter irrelevant information and focus on context-dependent signals, closing the gap with attention-based models:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SelectiveSSM(nn.Module):
    """
    Simplified selective state space layer (Mamba-style).
    
    Key idea: B, C, delta are input-dependent (not fixed), enabling
    the model to selectively retain or forget information per token.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Input projections
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        
        # Causal depthwise conv before SSM
        self.conv1d = nn.Conv1d(
            d_model, d_model, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_model
        )
        
        # SSM parameters
        # A is fixed (diagonal log-space), B/C/delta are input-dependent
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))   # log for numerical stability
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection
        
        # Input-dependent projections for B, C, delta
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        
        # Gate and input branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        
        # Depthwise conv (causal context)
        x_conv = rearrange(x_branch, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[..., :L]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)
        
        # Input-dependent SSM parameters
        x_dbl = self.x_proj(x_conv)
        dt, B_mat, C_mat = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_model), positive
        
        # Fixed A matrix (negative to ensure stability)
        A = -torch.exp(self.A_log.float())  # (d_model, d_state)
        
        # Selective scan (simplified recurrence for illustration)
        # In practice this uses a hardware-efficient parallel scan kernel
        y = self._selective_scan(x_conv, dt, A, B_mat, C_mat)
        
        # Gated output
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, u, dt, A, B, C):
        """Recurrent form of selective scan (training illustration only)."""
        B_batch, L, D = u.shape
        N = self.d_state
        
        # Discretize A and B
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)                           # (B, L, D, N)
        
        h = torch.zeros(B_batch, D, N, device=u.device)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * u[:, i].unsqueeze(-1)
            y = (h * C[:, i].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y)
        
        return torch.stack(ys, dim=1) + u * self.D
```

## The 2D Challenge: Adapting SSMs to Images

SSMs are inherently sequential — they process tokens in a fixed order. Images have no natural 1D ordering; patches in a 2D grid have spatial relationships in all directions. Naive row-major flattening loses spatial locality and makes the SSM directionally biased.

**Vision Mamba (Vim)** addresses this with **bidirectional scanning**: each patch is processed in both forward and backward directions, and the two hidden states are merged before projection. This ensures each patch has context from all positions in the sequence.

**VMamba** addresses it more comprehensively with a **Cross-Scan Module (CSM)**: every patch is processed in four independent scans — left-to-right, right-to-left, top-to-bottom, and bottom-to-top. The four output sequences are summed to produce the final feature map:

```python
class CrossScanModule(nn.Module):
    """
    VMamba Cross-Scan Module.
    
    Scans the 2D feature map in 4 directions so every position 
    receives context from all spatial directions.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        # 4 independent SSMs, one per scan direction
        self.ssms = nn.ModuleList([
            SelectiveSSM(d_model, d_state) for _ in range(4)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, H, W, d_model) — 2D feature map
        Returns: (batch, H, W, d_model)
        """
        B, H, W, D = x.shape
        
        # Flatten to sequences in 4 directions
        # Direction 0: left-to-right, top-to-bottom (row-major)
        seq_lr = x.reshape(B, H * W, D)
        # Direction 1: right-to-left, bottom-to-top (reverse)
        seq_rl = seq_lr.flip(1)
        # Direction 2: top-to-bottom, left-to-right (column-major)
        seq_tb = x.permute(0, 2, 1, 3).reshape(B, H * W, D)
        # Direction 3: bottom-to-top, right-to-left (reverse column-major)
        seq_bt = seq_tb.flip(1)
        
        sequences = [seq_lr, seq_rl, seq_tb, seq_bt]
        outputs = []
        
        for i, (seq, ssm) in enumerate(zip(sequences, self.ssms)):
            out = ssm(seq)
            if i == 1:
                out = out.flip(1)   # reverse back to original order
            elif i == 2:
                out = out.reshape(B, W, H, D).permute(0, 2, 1, 3).reshape(B, H * W, D)
            elif i == 3:
                out = out.flip(1)
                out = out.reshape(B, W, H, D).permute(0, 2, 1, 3).reshape(B, H * W, D)
            outputs.append(out)
        
        # Merge all 4 directions
        merged = torch.cat(outputs, dim=-1)   # (B, H*W, 4*D)
        out = self.out_proj(merged)            # (B, H*W, D)
        return out.reshape(B, H, W, D)


class VMambaBlock(nn.Module):
    """One Vision Mamba (VMamba) encoder block."""
    def __init__(self, d_model: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.csm = CrossScanModule(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, D)
        x = x + self.csm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

## VMamba Hierarchical Architecture

Like Swin Transformer, VMamba uses a hierarchical (pyramid) design with patch merging between stages, producing multi-scale feature maps suitable for dense prediction tasks:

```python
class VMamba(nn.Module):
    """
    VMamba backbone — hierarchical SSM vision encoder.
    4 stages with 2× spatial downsampling between stages.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dims: list[int] = [96, 192, 384, 768],
        depths: list[int] = [2, 2, 9, 2]
    ):
        super().__init__()
        
        # Patch embedding: non-overlapping 4×4 patches → tokens
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0], kernel_size=patch_size,
                      stride=patch_size),
            nn.LayerNorm(embed_dims[0])  # Applied after rearranging
        )
        
        # Build 4 stages
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage = nn.Sequential(*[
                VMambaBlock(d_model=dim) for _ in range(depth)
            ])
            self.stages.append(stage)
            
            # Patch merging (2×2 → 1 token, 4D → 2D)
            if i < len(embed_dims) - 1:
                next_dim = embed_dims[i + 1]
                self.downsamplers.append(
                    nn.Sequential(
                        nn.LayerNorm(4 * dim),
                        nn.Linear(4 * dim, next_dim, bias=False)
                    )
                )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns multi-scale feature maps for dense prediction."""
        features = []
        
        # Initial patch embedding
        x = self.patch_embed(x)  # (B, C, H/4, W/4)
        x = x.permute(0, 2, 3, 1)  # (B, H/4, W/4, C)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x.permute(0, 3, 1, 2))   # (B, C, H, W)
            
            if i < len(self.downsamplers):
                B, H, W, C = x.shape
                # 2×2 patch merging
                x = x.reshape(B, H // 2, 2, W // 2, 2, C)
                x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H // 2, W // 2, 4 * C)
                x = self.downsamplers[i](x)
        
        return features  # 4 feature maps at 1/4, 1/8, 1/16, 1/32 resolution
```

## Vision Mamba vs. Vision Transformer

| Property | ViT / Swin | VMamba / Vim |
| --- | --- | --- |
| Attention complexity | $O(N^2)$ (ViT), $O(N \cdot W^2)$ (Swin) | $O(N)$ |
| Memory scaling | Quadratic with resolution | Linear |
| Long-range modeling | Global (ViT) / local window (Swin) | Global via recurrence |
| Inductive bias | None (ViT) / locality (Swin) | Sequential (mitigated by multi-direction scan) |
| Hardware efficiency | Mature CUDA kernels | Custom CUDA scan kernels (Mamba) |
| Typical ImageNet-1K accuracy | Swin-T: 81.3%, ViT-B: 81.8% | VMamba-T: 82.2%, Vim-S: 80.5% |

## Applications and Strengths

SSM-based vision models excel particularly at:

- **High-resolution tasks**: Semantic segmentation, instance segmentation, object detection — where quadratic attention becomes a bottleneck
- **Medical imaging**: Whole slide pathology images, 3D volumetric CT/MRI scans where resolution far exceeds typical benchmarks
- **Video understanding**: Extending the temporal dimension with linear cost
- **Remote sensing**: Satellite and aerial imagery at very high resolution

The linear scaling of Vision Mamba models makes them a compelling alternative to window-based attention (Swin) for resolution-sensitive applications, with the VMamba cross-scan approach effectively recovering 2D spatial context that naive SSM sequential processing would lose.
