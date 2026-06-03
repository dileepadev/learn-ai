---
title: "Activation Functions in Deep Learning"
description: "Understanding ReLU, GELU, Swish, Mish, and other activation functions — their properties and when to use each."
date: "2026-06-06"
tags: ["deep-learning", "activation-functions", "neural-networks"]
---

Activation functions introduce non-linearity into neural networks. The choice of activation affects training stability, convergence speed, and final performance.

## Rectified Linear Unit (ReLU)

$$f(x) = \max(0, x)$$

```python
# Simple implementation
class ReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0)


# PyTorch built-in
relu = nn.ReLU()

# Pros: Simple, fast, sparse activation (50% zeros expected)
# Cons: Dying ReLU problem (neurons can get stuck at 0)
```

## Leaky ReLU

$$f(x) = \max(\alpha x, x), \quad \alpha \approx 0.01$$

```python
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        return torch.where(x >= 0, x, self.negative_slope * x)


leaky_relu = nn.LeakyReLU(negative_slope=0.01)
```

## Parametric ReLU (PReLU)

Learn the negative slope during training:

```python
class PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.full([num_parameters], init))
    
    def forward(self, x):
        return torch.where(x >= 0, x, self.weight * x)


prelu = PReLU(num_parameters=1)
```

## GELU (Gaussian Error Linear Unit)

$$f(x) = x \cdot \Phi(x)$$

Where $\Phi$ is the standard normal CDF. Used in BERT, GPT, and most modern transformers.

```python
class GELU(nn.Module):
    def forward(self, x):
        return x * torch.special.ncdf(x)  # Approximation


# Fast approximation
class FastGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
        ))


gelu = nn.GELU()  # PyTorch built-in uses the fast approximation
```

## Swish

$$f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

```python
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


swish = nn.SiLU()  # Same as Swish in PyTorch
```

## Mish

$$f(x) = x \cdot \tanh(\ln(1 + e^x))$$

```python
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


mish = nn.Mish()
```

## Softplus

$$f(x) = \ln(1 + e^x)$$

```python
softplus = nn.Softplus()
```

## Sigmoid and Tanh (Legacy Use)

```python
# Sigmoid: often for output layers only
sigmoid = nn.Sigmoid()

# Tanh: rarely used in hidden layers now
tanh = nn.Tanh()
```

## Comparison Table

| Activation | Formula | Monotonic | Smooth | Sparsity |
| --- | --- | --- | --- | --- |
| ReLU | max(0, x) | Yes | No | 50% |
| Leaky ReLU | max(0.01x, x) | Yes | No | <50% |
| GELU | x·Φ(x) | Yes | Yes | Low |
| Swish | x·σ(x) | No | Yes | Low |
| Mish | x·tanh(softplus(x)) | No | Yes | Low |

## Choosing an Activation

| Layer Type | Recommended Activation |
| --- | --- |
| CNN hidden layers | ReLU, GELU |
| Transformer hidden layers | GELU |
| RNN hidden states | Tanh, ReLU |
| Output (binary) | Sigmoid |
| Output (multi-class) | Softmax (logits before) |
| Output (regression) | None (linear) |

GELU is the default for modern transformers and often works well across tasks.