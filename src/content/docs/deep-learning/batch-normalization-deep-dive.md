---
title: "Batch Normalization: A Deep Dive"
description: "Understanding batch normalization — how it works, why it helps, and practical considerations for training deep networks."
date: "2026-06-06"
tags: ["deep-learning", "normalization", "training"]
---

Batch normalization stabilizes and accelerates training by normalizing layer inputs. This seemingly simple technique enabled training of networks with 10+ times more layers.

## How Batch Normalization Works

For a mini-batch $\mathcal{B} = \{x_1, ..., x_m\}$:

$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

The learned parameters $\gamma$ and $\beta$ allow the network to undo normalization if beneficial.

```python
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.running_mean.fill_(0)
        self.running_var.fill_(1)
    
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = self.momentum * self.running_mean + \
                                    (1 - self.momentum) * mean.squeeze()
                self.running_var = self.momentum * self.running_var + \
                                   (1 - self.momentum) * var.squeeze()
        else:
            mean = self.running_mean.view(1, self.num_features, 1, 1)
            var = self.running_var.view(1, self.num_features, 1, 1)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma.view(1, self.num_features, 1, 1) * x_norm + \
               self.beta.view(1, self.num_features, 1, 1)
```

## Why Batch Normalization Helps

### Reduces Internal Covariate Shift

By normalizing inputs to each layer, batch norm reduces the distribution shift between consecutive layers, making optimization easier.

### Smooths the Loss Landscape

Batch norm creates a smoother optimization landscape with fewer sharp local minima.

### Provides Regularization

The noise in batch statistics acts as a regularizer, reducing the need for dropout.

### Allows Higher Learning Rates

Stable gradients enable faster training with larger learning rates.

## Practical Considerations

### Batch Size

Smaller batches provide stronger regularization but less accurate statistics. A batch size of 32 is common for ImageNet models.

### Momentum

The momentum parameter controls how fast running statistics update:

```python
# Higher momentum = more stable statistics
bn = nn.BatchNorm2d(256, momentum=0.9)  # Default
bn = nn.BatchNorm2d(256, momentum=0.99)  # More stable
bn = nn.BatchNorm2d(256, momentum=0.1)  # More adaptive
```

### Inference vs Training

At inference, use running statistics; at training, use batch statistics.

### Spatial Dimensions

BatchNorm2d normalizes across (N, H, W) for each channel. For other dimensions, use:

```python
nn.BatchNorm1d(256)  # For fully-connected or temporal data
nn.BatchNorm3d(256)  # For 3D data (video, medical imaging)
```

## Synchronized Batch Norm

In distributed training, synchronize batch norm statistics across workers:

```python
# PyTorch DDP with synchronized batch norm
from torch.nn import SyncBatchNorm
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    SyncBatchNorm(64),
    nn.ReLU()
)
model = torch.nn.parallel.DistributedDataParallel(model)
```

## Instance Normalization

For style transfer, instance normalization is preferred:

```python
class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # Normalize each instance and channel independently
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma.view(1, self.num_features, 1, 1) * x_norm + \
               self.beta.view(1, self.num_features, 1, 1)
```

## When to Use Batch Norm

| Architecture | Recommendation |
| --- | --- |
| CNNs | BatchNorm2d (standard) |
| RNNs | LayerNorm or no norm |
| Transformers | LayerNorm (pre-norm) |
| GANs | Often omitted or use spectral norm |
| Very deep networks | Essential with skip connections |