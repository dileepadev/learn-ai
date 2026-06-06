---
title: "Residual Networks and Skip Connections"
description: "Understanding residual connections, their theory, variants (ResNet, DenseNet, Highway Networks), and how they enable training of very deep networks."
date: "2026-06-06"
tags: ["deep-learning", "computer-vision", "residual-networks", "skip-connections"]
---

Residual networks (ResNet) introduced the concept of skip connections, which allow gradients to flow directly through layers without attenuation. This simple insight enabled training of networks with hundreds or thousands of layers, fundamentally changing deep learning.

## The Degradation Problem

Before residual networks, simply stacking more layers made training harder — not because of overfitting, but because deeper networks had higher training error. This was called **degradation**: a 56-layer network had higher training error than a 20-layer network.

The hypothesis: deep networks should be able to at least achieve the performance of shallow networks (by learning identity mappings). The problem was that standard networks couldn't learn identity mappings efficiently.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def demonstrate_degradation():
    """Show why degradation occurs in deep networks."""
    class PlainConvNet(nn.Module):
        def __init__(self, depth: int = 20):
            super().__init__()
            layers = []
            for i in range(depth):
                layers.append(nn.Conv2d(64, 64, 3, padding=1))
                layers.append(nn.BatchNorm2d(64))
                layers.append(nn.ReLU())
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.features(x)
            return self.classifier(x.mean(dim=(2, 3)))
    
    # 20-layer network trains fine
    # 56-layer network trains worse (even with batch norm)
    # The problem: each layer must learn both identity AND transformation
    # With residual connections, each layer only learns the residual
    print("Plain networks struggle to learn identity mappings.")
    print("Residual networks make identity the easiest solution.")
```

## The Residual Connection

A residual connection adds the input to the output of a block:

$$y = F(x) + x$$

Where $F(x)$ is the learned transformation. If the optimal solution is identity, the network just sets $F(x) = 0$ (which is easy to learn). This makes the optimization landscape much smoother.

```python
class ResidualBlock(nn.Module):
    """Basic residual block (Conv -> BN -> ReLU -> Conv -> BN)."""
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Shortcut connection (identity or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or channels != channels:  # Fixed typo: was channels != channels
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=stride),
                nn.BatchNorm2d(channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """Bottleneck residual block (1x1 -> 3x3 -> 1x1)."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # Bottleneck: reduce -> transform -> expand
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Projection shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)  # Residual connection
        return F.relu(out)
```

## Full ResNet Architecture

```python
class ResNet(nn.Module):
    """ResNet for image classification."""
    def __init__(self, block_type: str, num_blocks: list, num_classes: int = 10):
        super().__init__()
        
        # Choose block type
        if block_type == 'basic':
            block = ResidualBlock
        elif block_type == 'bottleneck':
            block = BottleneckBlock
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)
        
        # Classification head
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int):
        """Create a stack of residual blocks."""
        layers = []
        
        # First block may downsample
        layers.append(ResidualBlock(out_channels, stride))
        
        # Remaining blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Standard ResNet configurations
resnet18 = ResNet('basic', [2, 2, 2, 2])
resnet34 = ResNet('basic', [3, 4, 6, 3])
resnet50 = ResNet('bottleneck', [3, 4, 6, 3])  # More efficient
resnet101 = ResNet('bottleneck', [3, 4, 23, 3])
resnet152 = ResNet('bottleneck', [3, 8, 36, 3])
```

## Why Residual Connections Work

### 1. Gradient Flow

During backpropagation, the gradient can flow directly through the skip connection:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} + \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial (x + F(x))}{\partial x}$$

The term $\frac{\partial \mathcal{L}}{\partial y}$ is added at each layer, enabling gradients to propagate to shallower layers without attenuation.

```python
class GradientFlowAnalysis:
    """Analyze gradient flow in ResNet vs plain network."""
    def __init__(self, model):
        self.model = model
        self.gradients = []
    
    def compute_gradient_norm(self, x, target_layer_idx: int = 5):
        """Compute gradient norm at different depths."""
        x.requires_grad_(True)
        
        for i, layer in enumerate(self.model.features):
            x = layer(x)
            
            if i == target_layer_idx:
                # Compute gradient of loss w.r.t. this layer's output
                grad = torch.autograd.grad(
                    torch.ones(x.size(0)).sum(),
                    x,
                    create_graph=True
                )[0]
                self.gradients.append(grad.norm().item())
        
        return self.gradients
```

### 2. Ensemble Interpretation

ResNet with multiple parallel skip connections can be interpreted as an ensemble of networks of different depths.

### 3. Adaptive Depth

The network can use different paths for different inputs, effectively choosing its depth adaptively.

```python
class AdaptiveDepthResNet(nn.Module):
    """ResNet that uses variable depth during inference."""
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.layers = nn.ModuleList()
        
        channels = [64, 128, 256, 512]
        for i in range(4):
            self.layers.append(self._make_layer(block, channels[i], num_blocks[i], 2))
    
    def _make_layer(self, block, channels, num_blocks, stride):
        layers = [block(64 if channels == 128 else channels, channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(block(channels, channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x, max_blocks: int = None):
        x = self.stem(x)
        
        num_layers = len(self.layers)
        blocks_to_use = max_blocks or num_layers
        
        for i, layer in enumerate(self.layers):
            if i >= blocks_to_use:
                break
            x = layer(x)
        
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.fc(x)
```

## Variants of Skip Connections

### Dense Connections (DenseNet)

DenseNet connects each layer to all subsequent layers:

$$x_l = [x_0, x_1, ..., x_{l-1}, F_l(x_l)]$$

```python
class DenseLayer(nn.Module):
    """Single DenseNet layer."""
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1)
    
    def forward(self, x):
        out = torch.cat([x, F.relu(self.bn1(self.conv1(x)))], dim=1)
        out = F.relu(self.bn2(self.conv2(out)))
        return torch.cat([out, out], dim=1)  # Concatenate
```

### Highway Networks

Highway networks use a gating mechanism:

$$y = H(x, W_H) \cdot T(x, W_T) + x \cdot C(x, W_C)$$

Where $T$ is the transform gate and $C$ is the carry gate.

```python
class HighwayConv2d(nn.Module):
    """2D convolution with highway connection."""
    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.transform_gate = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        transform = torch.sigmoid(self.transform_gate(x))
        carry = 1 - transform
        return self.conv(x) * transform + x * carry
```

### Pre-activation ResNet

Pre-activation (He et al., 2016) places normalization before convolution:

$$y = x + F(\text{BN}(\text{ReLU}(x)))$$

This improves gradient flow and enables cleaner identity mappings.

```python
class PreActivationResBlock(nn.Module):
    """Pre-activation residual block (better gradient flow)."""
    def __init__(self, channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=stride),
                nn.BatchNorm2d(channels)
            )
    
    def forward(self, x):
        # Pre-activation: BN -> ReLU -> Conv
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        # Residual connection
        out = out + self.shortcut(x)
        return out


# Pre-activation ResNet was used to train 1001-layer networks
```

## Skip Connections in Transformers

Skip connections are equally important in transformers:

```python
class TransformerLayerWithResidual(nn.Module):
    """Transformer layer with residual connections."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm: norm -> attention -> residual
        x_norm = self.attention_norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm: norm -> FFN -> residual
        x_norm = self.ffn_norm(x)
        ff_out = self.ffn(x_norm)
        x = x + self.dropout(ff_out)
        
        return x
```

## Practical Recommendations

- **Basic block**: 2 conv layers, good for smaller networks
- **Bottleneck block**: 1x1 -> 3x3 -> 1x1, parameter-efficient for deep networks
- **Stride 1 blocks**: Maintain spatial dimensions
- **Stride 2 blocks**: Downsample, increase channels
- **Pre-activation**: Use for networks deeper than 100 layers
- **Number of blocks**: [3, 4, 6, 3] is a good starting point (ResNet-34)

Residual connections are one of the most important architectural innovations in deep learning, enabling the training of networks that would otherwise be impossible to optimize.