---
title: Weight Initialization Strategies
description: Understand why neural network weight initialization matters and how different strategies — Xavier/Glorot, He/Kaiming, orthogonal initialization, and Maximal Update Parameterization (μP) — prevent vanishing and exploding gradients, enabling stable training of deep networks.
---

Weight initialization is one of the most underappreciated yet fundamental aspects of deep learning. The initial values of network weights determine whether gradients flow usefully through the network during the first few steps of training — or whether they vanish to zero (leaving layers unlearnable) or explode to infinity (causing divergence). Proper initialization ensures that activations and gradients maintain reasonable variance throughout the network depth.

## Why Initialization Matters: The Variance Propagation Problem

Consider a single linear layer with $n_{in}$ inputs and no activation function. If weights $w_{ij} \sim \mathcal{N}(0, \sigma^2)$ and inputs $x_i$ have variance 1, the output variance is:

$$\text{Var}(y_j) = n_{in} \cdot \sigma^2 \cdot \text{Var}(x)$$

For $L$ stacked layers, signal variance scales as $(n_{in} \cdot \sigma^2)^L$. If $n_{in} \cdot \sigma^2 > 1$, activations explode; if $< 1$, they vanish. The same analysis applies to gradients in backpropagation. The goal of principled initialization is to choose $\sigma$ so that variance is preserved across layers.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def measure_activation_variance(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_layers: int = 10
) -> dict[str, list[float]]:
    """
    Track activation variance at each layer to detect vanishing/exploding signals.
    Uses forward hooks to capture intermediate activations.
    """
    variances = []
    hooks = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            variances.append(output.detach().var().item())
    
    # Register hooks on all linear/conv layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    return variances


def compare_initializations(depth: int = 20, width: int = 512):
    """Compare activation variance across initialization strategies."""
    x = torch.randn(32, width)   # batch of 32, width-dim inputs
    results = {}
    
    for init_name, init_fn in [
        ("Random Normal σ=0.01", lambda m: nn.init.normal_(m.weight, 0, 0.01)),
        ("Xavier Uniform", lambda m: nn.init.xavier_uniform_(m.weight)),
        ("He Normal", lambda m: nn.init.kaiming_normal_(m.weight, nonlinearity='relu')),
        ("Orthogonal", lambda m: nn.init.orthogonal_(m.weight)),
    ]:
        model = nn.Sequential(
            *[nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(depth)]
        )
        
        # Apply initialization
        for module in model.modules():
            if isinstance(module, nn.Linear):
                init_fn(module)
                nn.init.zeros_(module.bias)
        
        variances = measure_activation_variance(model, x)
        results[init_name] = variances
        print(f"{init_name}: final layer variance = {variances[-1]:.6f}")
    
    return results
```

## Xavier / Glorot Initialization

Xavier initialization (Glorot & Bengio, 2010) was derived for **linear activations** (or sigmoid/tanh, which are approximately linear near zero). The goal is to preserve variance through both the forward and backward pass. The solution balances fan-in and fan-out:

$$\sigma^2 = \frac{2}{n_{in} + n_{out}}$$

$$W \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{in} + n_{out}}},\; \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

The uniform variant is the default in most frameworks. The normal variant uses $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$.

```python
import torch
import torch.nn as nn

def xavier_init_manual(fan_in: int, fan_out: int) -> torch.Tensor:
    """Manual Xavier uniform initialization."""
    bound = (6.0 / (fan_in + fan_out)) ** 0.5
    return torch.empty(fan_out, fan_in).uniform_(-bound, bound)


# PyTorch built-in
layer = nn.Linear(512, 256)
nn.init.xavier_uniform_(layer.weight)   # Uniform variant (default for Linear)
nn.init.xavier_normal_(layer.weight)    # Normal variant

# Xavier is the default for nn.Linear in PyTorch:
# torch.nn.Linear.__init__ calls kaiming_uniform_ with a=math.sqrt(5)
# which approximates xavier for typical layer sizes
```

## He / Kaiming Initialization

Xavier assumes linear activations. **ReLU** clips negative activations to zero, effectively halving the variance of activations after each layer. He initialization (He et al., 2015) corrects for this:

$$\sigma^2 = \frac{2}{n_{in}}$$

The factor of 2 compensates for the half of activations zeroed by ReLU. For leaky ReLU with negative slope $a$:

$$\sigma^2 = \frac{2}{(1 + a^2) \cdot n_{in}}$$

```python
# He initialization (PyTorch)
conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')

# mode='fan_in': forward pass variance preservation (recommended for deep nets)
# mode='fan_out': backward pass preservation (useful for very deep resnets)

# For leaky relu with slope 0.2 (common in GANs):
nn.init.kaiming_normal_(conv.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

# Demonstration: why He init is critical for ReLU networks
def test_relu_variance_preservation():
    torch.manual_seed(42)
    depth, width = 50, 256
    x = torch.randn(1, width)
    
    # Bad: Xavier (designed for tanh, underestimates variance loss from ReLU)
    model_xavier = nn.Sequential(*[
        nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(depth)
    ])
    for m in model_xavier.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    with torch.no_grad():
        out_xavier = model_xavier(x)
    
    # Good: He (accounts for ReLU variance halving)
    model_he = nn.Sequential(*[
        nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(depth)
    ])
    for m in model_he.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    with torch.no_grad():
        out_he = model_he(x)
    
    print(f"Xavier output norm: {out_xavier.norm().item():.4f}")
    print(f"He     output norm: {out_he.norm().item():.4f}")
```

## Orthogonal Initialization

Orthogonal initialization sets the weight matrix to a random orthogonal matrix (for square matrices) or a random matrix with orthonormal rows/columns (for rectangular). Orthogonal transformations preserve vector norms exactly, so activations neither grow nor shrink:

$$\|W x\|_2 = \|x\|_2 \quad \text{(if } W \text{ is orthogonal)}$$

Orthogonal initialization is particularly valuable for:

- **RNNs**: Prevents vanishing/exploding gradients over long sequences
- **Very deep networks**: Exact norm preservation without batch normalization
- **ResNets at initialization**: Delta-orthogonal init ensures identity-like behavior at depth

```python
def orthogonal_init_example():
    """Orthogonal initialization for RNN hidden-to-hidden weights."""
    # Standard RNN cell
    hidden_size = 256
    rnn = nn.RNNCell(input_size=128, hidden_size=hidden_size)
    
    # Initialize hidden-to-hidden weight as orthogonal matrix
    nn.init.orthogonal_(rnn.weight_hh)
    
    # Verify: W^T W ≈ I
    W = rnn.weight_hh.detach()
    product = W @ W.T
    identity_error = (product - torch.eye(hidden_size)).abs().max().item()
    print(f"Max deviation from identity: {identity_error:.6f}")   # ≈ 1e-6
    
    # Delta-orthogonal init for conv layers (Xiao et al. 2018)
    # Used to train 10,000-layer networks without batch norm
    def delta_orthogonal_init(conv: nn.Conv2d):
        """Initialize convolution as near-identity at center kernel position."""
        k = conv.kernel_size[0]
        nn.init.zeros_(conv.weight)
        center = k // 2
        # Set center slice to orthogonal matrix
        nn.init.orthogonal_(conv.weight[:, :, center, center])
```

## Maximal Update Parameterization (μP)

A fundamental problem with standard parameterizations: **optimal hyperparameters (especially learning rate) change as model width increases**. This makes it impossible to tune a small proxy model and transfer those hyperparameters to a large model.

**μP** (Yang et al., 2022) addresses this with a parameterization where the optimal learning rate (and other hyperparameters) is provably width-independent — a property called **hyperparameter transfer (HPT)**:

$$W^L \sim \mathcal{N}\!\left(0,\; \frac{\sigma_\text{init}^2}{\text{fan\_in}}\right) \quad \text{(μP input layers)}$$
$$W^h \sim \mathcal{N}\!\left(0,\; \frac{\sigma_\text{init}^2}{\text{fan\_in}}\right) \quad \text{(μP hidden layers)}$$

The key difference from standard parameterization: forward pass activations are divided by $\sqrt{n}$ at hidden layers, and the learning rate for hidden layers scales as $1/n$ (where $n$ is width). This ensures that the magnitude of weight updates is the same relative to weight magnitude regardless of width.

```python
import torch
import torch.nn as nn

class MuPLinear(nn.Linear):
    """
    Linear layer with Maximal Update Parameterization (μP).
    
    Ensures that:
    1. Initialization variance is independent of width
    2. Output scaling is 1/fan_in (not 1/sqrt(fan_in))
    3. Learning rate can be set once and transferred to any width
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init_std: float = 1.0, is_output: bool = False):
        super().__init__(in_features, out_features, bias)
        self.init_std = init_std
        self.is_output = is_output
        self._mup_init()

    def _mup_init(self):
        # μP: hidden layer init ~ N(0, sigma^2 / fan_in)
        # Standard PyTorch uses 1/sqrt(fan_in)
        std = self.init_std / self.in_features if not self.is_output else \
              self.init_std / self.in_features ** 0.5
        nn.init.normal_(self.weight, mean=0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # μP output scaling: divide by fan_in for hidden layers
        out = super().forward(x)
        if not self.is_output:
            out = out / self.in_features ** 0.5   # additional 1/sqrt(n) scaling
        return out


def build_mup_mlp(width: int, depth: int, input_dim: int, output_dim: int) -> nn.Sequential:
    """Build MLP with μP — same optimal LR regardless of width."""
    layers = [MuPLinear(input_dim, width)]  # input layer: standard scaling
    for _ in range(depth - 2):
        layers.extend([nn.GELU(), MuPLinear(width, width)])
    layers.extend([nn.GELU(), MuPLinear(width, output_dim, is_output=True)])
    return nn.Sequential(*layers)


# Key benefit: tune LR on width=256, transfer to width=4096
# Without μP: LR needs to be re-tuned at each width
# With μP: optimal LR(width=256) ≈ optimal LR(width=4096)
```

## Initialization Summary

| Strategy | Formula | Best for | Key property |
| --- | --- | --- | --- |
| Random Normal | $\mathcal{N}(0, 0.01^2)$ | Prototype/debugging | Simple, usually suboptimal |
| Xavier Uniform | $\mathcal{U}(-\sqrt{6/(n_{in}+n_{out})}, \cdot)$ | sigmoid / tanh / linear | Balances forward + backward variance |
| He / Kaiming | $\mathcal{N}(0, 2/n_{in})$ | ReLU networks | Accounts for ReLU's 50% zero rate |
| Orthogonal | Random orthogonal matrix | RNNs, very deep nets | Exact norm preservation |
| μP | $\mathcal{N}(0, \sigma^2/n_{in})$ + output scaling | Large-scale pretraining | Hyperparameter transfer across widths |

## Practical Recommendations

In practice, PyTorch applies `kaiming_uniform_` (a variant of He) by default for `nn.Linear` and `nn.Conv2d`. For most standard architectures this is fine. Switch to:

- **Xavier** when using sigmoid/tanh activations
- **Orthogonal** for RNN hidden-to-hidden weights or signal propagation experiments without batch norm
- **μP** when training large models and wanting to transfer hyperparameters from small proxy models
- **Zero + small random** for output projection layers in some transformer implementations (helps with early training stability)

Initialization interacts strongly with learning rate, batch normalization/layer normalization, and residual connections. With proper normalization (LayerNorm, BatchNorm) and residual connections, the exact initialization choice matters less — but a principled choice still gives faster early training and better final performance.
