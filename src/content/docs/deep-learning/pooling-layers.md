---
title: "Pooling Layers in Deep Learning"
description: "Understanding max pooling, average pooling, adaptive pooling, and learnable pooling methods."
date: "2026-06-06"
tags: ["deep-learning", "convolutional-neural-networks", "pooling"]
---

Pooling layers reduce spatial dimensions, provide translation invariance, and control computational complexity.

## Max Pooling

Returns the maximum value in each window:

```python
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)
    
    def forward(self, x):
        return self.pool(x)


# Output size
output_size = (input_size - kernel_size) // stride + 1
```

## Average Pooling

Returns the mean of each window:

```python
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Global average pooling
global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output: (batch, channels, 1, 1)
```

## Adaptive Pooling

Pool to any output size, learning parameters if needed:

```python
# Adaptive pooling to fixed size
adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

# Adaptive to 1D output (global pooling)
global_pool = nn.AdaptiveAvgPool1d(1)


class LearnablePooling(nn.Module):
    """Learnable weighted pooling."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x shape: (batch, dim, seq_len) or (batch, dim, H, W)
        weight = self.weight / (self.weight.sum() + self.eps)
        if x.dim() == 3:
            return torch.einsum('bdn,b n->bd', x, weight)
        else:
            return torch.einsum('bdhw,b->bd', x, weight)
```

## Power Average Pooling

Generalized pooling with learnable exponent:

```python
class PowerAvgPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, p=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = p
    
    def forward(self, x):
        batch, ch, h, w = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        
        # Unfold
        x_unfold = x.unfold(2, kh, self.stride).unfold(3, kw, self.stride)
        
        # Power pooling
        pooled = (x_unfold ** self.p).mean(dim=(-1, -2))
        
        return pooled ** (1.0 / self.p)


# p=1: average pooling
# p=2: RMS-like pooling
# p→∞: approaches max pooling
```

## Mixed Pooling

Combines max and average pooling:

```python
class MixedPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, mix_ratio=0.5):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride)
        self.mix_ratio = nn.Parameter(torch.tensor(mix_ratio))
    
    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        
        # Learnable combination
        ratio = torch.sigmoid(self.mix_ratio)
        return ratio * max_out + (1 - ratio) * avg_out
```

## Strided Convolution as Pooling

Some architectures use strided convolution instead of pooling:

```python
# Downsampling with strided convolution
downsample = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

# Advantages:
# - Learnable downsampling
# - Can increase channels if needed
```

## Pooling Strategies Comparison

| Type | Invariance | Information Preserved | Use Case |
| --- | --- | --- | --- |
| Max | Position | Activations | Feature detection |
| Average | Smooth | Global statistics | Global features |
| Adaptive | Any size | Flexible | Variable input sizes |
| Power | Parameterized | Tunable | Custom behavior |

For classification, global average pooling before the classifier is standard.