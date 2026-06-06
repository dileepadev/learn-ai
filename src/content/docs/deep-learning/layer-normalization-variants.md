---
title: "Layer Normalization Variants"
description: "Understanding batch normalization, layer normalization, RMS normalization, and group normalization — when to use each and how they differ."
date: "2026-06-06"
tags: ["deep-learning", "normalization", "training"]
---

Normalization layers stabilize and accelerate neural network training by controlling the distribution of activations. Different normalization strategies make different assumptions about what should be normalized and over which dimensions.

## The Problem Normalization Solves

During training, the distribution of activations in each layer shifts as network weights update — a phenomenon called **internal covariate shift**. This forces each layer to continuously adapt to changing input distributions, slowing learning and requiring careful initialization and lower learning rates.

Normalization addresses this by fixing the mean and variance of activations, either:

- **Pre-activation**: Normalize before the nonlinearity (more common)
- **Post-activation**: Normalize after the nonlinearity (less common, can hurt representation power)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def demonstrate_covariate_shift():
    """Show why normalization helps training stability."""
    torch.manual_seed(42)
    
    class BadNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(128, 256) for _ in range(10)
            ])
        
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x))
                # Without normalization, activations can grow unbounded
            return x
    
    model = BadNetwork()
    x = torch.randn(32, 128)  # Batch of 32
    
    with torch.no_grad():
        for i in range(10):
            x = F.relu(model.layers[i](x))
            mean, std = x.mean().item(), x.std().item()
            print(f"Layer {i+1}: mean={mean:.4f}, std={std:.4f}")
```

## Batch Normalization

Batch normalization (Ioffe & Szegedy, 2015) normalizes across the **batch dimension** for each channel independently:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

Where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance.

Key properties:
- Learns shift ($\beta$) and scale ($\gamma$) parameters per channel
- Acts as a regularizer (noise from batch statistics acts as dropout)
- Requires sufficient batch size to estimate stable statistics
- Not suitable for RNNs (statistics differ per timestep)

```python
class BatchNorm2d(nn.Module):
    """Simplified batch normalization."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (for inference)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        if self.training:
            # Compute batch statistics
            mu = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            
            # Update running statistics with exponential moving average
            self.running_mean = self.momentum * self.running_mean + \
                                (1 - self.momentum) * mu.squeeze()
            self.running_var = self.momentum * self.running_var + \
                               (1 - self.momentum) * var.squeeze()
        else:
            # Use running statistics at test time
            mu = self.running_mean.view(1, self.num_features, 1, 1)
            var = self.running_var.view(1, self.num_features, 1, 1)
        
        # Normalize and apply learned parameters
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma.view(1, self.num_features, 1, 1) * x_norm + \
               self.beta.view(1, self.num_features, 1, 1)


# PyTorch built-in
bn = nn.BatchNorm2d(256)
```

## Layer Normalization

Layer normalization (Ba et al., 2016) normalizes across the **feature dimension** for each sample independently:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

Where $\mu$ and $\sigma^2$ are computed across all features for a single sample.

Key properties:
- Computed independently for each sample — no batch dependence
- Works for any batch size (including batch size 1)
- The default for transformer models (in encoder layers)
- For RNNs, normalizes across both time and features

```python
class LayerNorm(nn.Module):
    """Simplified layer normalization."""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable affine parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        # x shape: (batch, ..., features)
        # Compute mean and variance across feature dimension only
        dims = list(range(-1, -len(self.normalized_shape) - 1, -1))
        mu = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# Layer norm in transformers normalizes the last dimension
# For BERT: layer_norm(dim=768) applied to (batch, seq, 768)
# For ViT: layer_norm(dim=768) applied to (batch, patch+1, 768)

class TransformerLN(nn.Module):
    """Layer normalization as used in transformers."""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps)
    
    def forward(self, x):
        # x: (batch, seq, d_model) or (batch, channels, H, W) for ViT
        return self.norm(x)
```

## RMS Normalization

Root Mean Square normalization (Zhang et al., 2019) is a simpler alternative to layer norm:

$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)}$$

Where $\text{RMS}(x) = \sqrt{\frac{1}{n} \sum_i x_i^2}$ and there's a learned scalar $g$.

Key properties:
- Removes mean centering (focuses on scale only)
- Faster than layer norm (no mean computation)
- Used in LLaMA and other modern LLMs
- Better convergence in some cases

```python
class RMSNorm(nn.Module):
    """Root Mean Square normalization (used in LLaMA, etc.)."""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Only scale parameter (no shift) — simpler than layer norm
        self.g = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x):
        # Compute RMS: sqrt(mean(x^2))
        dims = list(range(-1, -len(self.normalized_shape) - 1, -1))
        rms = torch.sqrt((x.pow(2).mean(dim=dims, keepdim=True) + self.eps))
        
        x_norm = x / rms
        return self.g * x_norm


# RMS norm is faster than layer norm because it only computes
# the squared mean, avoiding the extra mean subtraction step

class LlamaRMSNorm(nn.Module):
    """RMSNorm as used in LLaMA models."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        # Compute RMS for last dimension
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * output
```

## Group Normalization

Group normalization (Wu & He, 2018) normalizes across channel **groups** rather than the whole batch or all features:

$$\hat{x}_{ijk} = \frac{x_{ijk} - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}$$

Where statistics are computed over groups of channels.

Key properties:
- Independent of batch size (works for any batch size)
- Groups are typically 8-32 channels
- Default choice for object detection and segmentation (small batch sizes)
- Often combined with weight standardization

```python
class GroupNorm(nn.Module):
    """Simplified group normalization."""
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        # Scale and shift for each channel
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        # Reshape to (batch, groups, channels_per_group, height, width)
        assert self.num_channels % self.num_groups == 0
        channels_per_group = self.num_channels // self.num_groups
        
        x = x.reshape(x.shape[0], self.num_groups, channels_per_group, *x.shape[2:])
        
        # Compute mean and variance per group
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        
        # Reshape back
        return self.gamma.view(1, self.num_channels, 1, 1) * \
               x_norm.reshape_as(x) + self.beta.view(1, self.num_channels, 1, 1)


# PyTorch built-in
gn = nn.GroupNorm(num_groups=32, num_channels=256)  # 8 channels per group

# Group norm in ResNet blocks for detection/segmentation
class ConvBlockGN(nn.Module):
    """Conv block with group norm (common in detection models)."""
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        return self.relu(x)
```

## Comparison Summary

| Normalization | Dimensions | Batch Dependence | Best For |
| --- | --- | --- | --- |
| Batch Norm | (batch, features) | Yes | Image classification, CNNs with large batches |
| Layer Norm | (features per sample) | No | Transformers, RNNs, any batch size |
| RMS Norm | (features per sample) | No | LLMs (LLaMA, etc.), faster than layer norm |
| Group Norm | (groups, spatial) | No | Small batch sizes, object detection, segmentation |

```python
def choose_normalization(architecture: str, batch_size: int) -> str:
    """Guidelines for choosing normalization."""
    if architecture == "cnn" and batch_size >= 32:
        return "BatchNorm2d"
    elif architecture == "transformer":
        return "LayerNorm (pre-norm) or RMSNorm"
    elif batch_size < 8 or architecture in ["detection", "segmentation"]:
        return "GroupNorm"
    elif "llama" in architecture.lower() or "llm" in architecture.lower():
        return "RMSNorm"
    else:
        return "LayerNorm"
```

## Practical Recommendations

- **Image classification (CNNs)**: BatchNorm2d works well with batch size >= 32
- **Object detection/segmentation**: GroupNorm (8-32 groups) since batch sizes are typically 1-8
- **Transformers**: LayerNorm or RMSNorm; pre-norm is more stable than post-norm
- **RNNs**: LayerNorm (applied to hidden state dimension)
- **Small language models**: RMSNorm for speed; LayerNorm for simplicity
- **Very deep networks**: Consider combining normalization with skip connections

Batch normalization's regularizing effect diminishes at small batch sizes. Group norm and layer norm avoid this trade-off entirely.