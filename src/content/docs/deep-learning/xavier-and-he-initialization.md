---
title: "Weight Initialization: Xavier and He Initialization"
description: "Understanding how to properly initialize neural network weights for effective training of deep networks."
date: "2026-06-06"
tags: ["deep-learning", "initialization", "training"]
---

Proper weight initialization is crucial for training deep networks. Poor initialization can lead to vanishing or exploding gradients, making training impossible.

## The Problem

Without proper initialization, activations and gradients can either:

- **Vanish**: Become too small to propagate useful signals
- **Explode**: Become numerically unstable

## Xavier / Glorot Initialization

For linear or tanh activations:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

```python
# Normal variant
nn.init.xavier_normal_(layer.weight)

# Uniform variant (more common)
nn.init.xavier_uniform_(layer.weight)


# Manual calculation
def xavier_init(fan_in, fan_out):
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return torch.rand(fan_out, fan_in) * 2 * bound - bound
```

## He / Kaiming Initialization

For ReLU activations (accounts for the ~50% dead ReLUs):

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

```python
# Normal variant for ReLU
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Uniform variant
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')


# For leaky ReLU
nn.init.kaiming_normal_(layer.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
```

## Mode Selection

- **fan_in**: Preserves variance in forward pass
- **fan_out**: Preserves variance in backward pass

```python
# For deep networks, fan_in is usually better
conv = nn.Conv2d(64, 128, 3)
nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
```

## Bias Initialization

```python
# Standard: zero bias
nn.init.zeros_(layer.bias)

# For ReLU: sometimes small positive bias helps
nn.init.constant_(layer.bias, 0.01)

# For output layers: depends on task
# Classification: zero
# Regression: may need adjustment
```

## Default PyTorch Behavior

```python
# nn.Linear and nn.Conv2d use kaiming_uniform_ by default
layer = nn.Linear(512, 256)
print(layer.weight)  # Already initialized
```

## Summary Table

| Activation | Initialization | Variance |
| --- | --- | --- |
| Linear/tanh | Xavier | $2 / (n_{in} + n_{out})$ |
| ReLU | He (fan_in) | $2 / n_{in}$ |
| Leaky ReLU | He (with a) | $2 / ((1 + a^2) n_{in})$ |

For modern transformers with GELU, He initialization works well.