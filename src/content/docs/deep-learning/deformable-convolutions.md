---
title: Deformable Convolutions
description: Explore deformable convolutions — a powerful extension of standard CNNs where the sampling grid adapts to the input geometry, enabling networks to learn object-specific receptive fields, handle geometric transformations, and improve accuracy on detection and segmentation tasks.
---

**Deformable convolutions** (Dai et al., Microsoft Research, 2017) augment standard convolutional neural networks with the ability to learn *where* to look, not just *what* to look for. In a standard convolution, the sampling grid is a fixed rectangular pattern — the same $3 \times 3$ kernel is applied at the same relative offsets to every spatial location. Deformable convolutions add learnable 2D offsets to each sampling location, allowing the receptive field to adapt to the shape, scale, and pose of objects in the image.

This seemingly small change has significant practical impact: it allows networks to model complex geometric transformations without requiring data augmentation or specialized architectures for each transformation type.

## Standard vs. Deformable Convolution

**Standard convolution** at output location $\mathbf{p}_0$:

$$y(\mathbf{p}_0) = \sum_{\mathbf{p}_n \in \mathcal{R}} w(\mathbf{p}_n) \cdot x(\mathbf{p}_0 + \mathbf{p}_n)$$

where $\mathcal{R} = \{(-1,-1), (-1,0), \ldots, (1,1)\}$ is the fixed $3 \times 3$ sampling grid.

**Deformable convolution** adds a learned offset $\Delta\mathbf{p}_n$ to each sampling location:

$$y(\mathbf{p}_0) = \sum_{\mathbf{p}_n \in \mathcal{R}} w(\mathbf{p}_n) \cdot x(\mathbf{p}_0 + \mathbf{p}_n + \Delta\mathbf{p}_n)$$

The offset $\Delta\mathbf{p}_n \in \mathbb{R}^2$ is fractional — the displaced position $\mathbf{p}_0 + \mathbf{p}_n + \Delta\mathbf{p}_n$ is not generally on an integer pixel location. **Bilinear interpolation** resolves this:

$$x(\mathbf{p}) = \sum_{\mathbf{q}} G(\mathbf{q}, \mathbf{p}) \cdot x(\mathbf{q})$$

where $G(\mathbf{q}, \mathbf{p}) = \max(0, 1 - |q_x - p_x|) \cdot \max(0, 1 - |q_y - p_y|)$ is the bilinear kernel and $\mathbf{q}$ sums over all integer locations.

The offsets are predicted by a separate lightweight convolutional branch applied to the same input feature map — making the entire operation end-to-end trainable via backpropagation through bilinear interpolation.

## Deformable ConvNets v2: Modulated Deformable Convolutions

**DCNv2** (Zhu et al., 2019) extends DCNv1 with **modulation scalars** $\Delta m_k \in [0, 1]$ per sampling point — allowing the network to not only shift where it samples but also to suppress or amplify the contribution of each location:

$$y(\mathbf{p}_0) = \sum_{k=1}^{K} w_k \cdot x(\mathbf{p}_0 + \mathbf{p}_k + \Delta\mathbf{p}_k) \cdot \Delta m_k$$

With both offsets and modulation, the network can effectively ignore irrelevant sampling locations ($\Delta m_k \approx 0$) and focus on the most informative ones. This is analogous to spatial attention but tightly integrated into the convolution operation.

## Implementation in PyTorch

PyTorch's `torchvision.ops.deform_conv2d` provides an efficient CUDA implementation. Here is a module wrapping it:

```python
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

class DeformableConv2d(nn.Module):
    """
    Modulated Deformable Convolution (DCNv2).
    Learns per-sample-location offsets AND modulation scalars.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Main convolution weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # Offset + mask prediction: 2*k*k offsets (x,y per location) + k*k masks
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,  # 2 for offsets (x,y), 1 for mask per location
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.offset_mask_conv.weight)
        nn.init.zeros_(self.offset_mask_conv.bias)  # Initialize offsets to zero = standard conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k2 = self.kernel_size * self.kernel_size
        
        # Predict offsets and modulation masks from input
        offset_mask = self.offset_mask_conv(x)
        
        # Split: first 2*k^2 channels are offsets, last k^2 are masks
        offset = offset_mask[:, :2 * k2, :, :]          # (B, 2*k^2, H, W)
        mask = offset_mask[:, 2 * k2:, :, :].sigmoid()  # (B, k^2, H, W), range [0,1]
        
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask
        )


class DeformableResBlock(nn.Module):
    """
    ResNet bottleneck block with deformable convolution replacing the 3x3 conv.
    Commonly used to upgrade ResNet-50/101 backbones in detectors.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(channels // 4)
        
        # Replace standard 3x3 with deformable 3x3
        self.deform_conv = DeformableConv2d(channels // 4, channels // 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels // 4)
        
        self.conv3 = nn.Conv2d(channels // 4, channels, 1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.deform_conv(out)))
        out = self.bn3(self.conv3(out))
        
        return self.relu(out + residual)
```

## Visualizing Learned Offsets

The sampling patterns learned by deformable convolutions are interpretable — and reveal that the network learns semantically meaningful receptive fields:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_offsets(offset: torch.Tensor, img: torch.Tensor,
                      location: tuple = (50, 50), kernel_size: int = 3):
    """
    Visualize the deformed sampling grid at a specific image location.
    
    offset: (B, 2*k^2, H, W) — learned offsets
    location: (row, col) — the output location to inspect
    """
    row, col = location
    k = kernel_size
    k2 = k * k
    
    # Standard grid (no deformation)
    offsets_y = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    offsets_x = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    
    # Extract learned offsets at this location
    off = offset[0, :, row, col].detach().cpu().numpy()
    delta_y = off[:k2]
    delta_x = off[k2:2*k2]
    
    standard_y = row + offsets_y
    standard_x = col + offsets_x
    deformed_y = row + offsets_y + delta_y
    deformed_x = col + offsets_x + delta_x
    
    img_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (pts_y, pts_x, title) in zip(axes, [
        (standard_y, standard_x, "Standard Grid"),
        (deformed_y, deformed_x, "Deformed Grid")
    ]):
        ax.imshow(img_np)
        ax.scatter(pts_x, pts_y, c="red", s=80, zorder=5)
        ax.scatter([col], [row], c="blue", s=120, marker="*", zorder=6, label="center")
        ax.set_title(title)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("deformable_offsets.png", dpi=100)
```

When visualized on object detection tasks, the sampling points of deformable convolutions group around object parts — for example, centering on the joints of a person, the wheels of a car, or the edges of a text bounding box.

## Applications and Impact

**Object detection**: Deformable convolutions are integrated into most top-performing detection architectures. Faster R-CNN and FCOS with DCNv2 backbones consistently outperform their standard CNN counterparts on COCO by 2–4 AP points.

**Semantic segmentation**: DeepLab and similar architectures use deformable convolutions to adaptively capture multi-scale context — complementing or replacing dilated (atrous) convolutions.

**Deformable DETR**: Extends the DETR (Detection Transformer) architecture with multi-scale deformable attention, attending to a small set of learned key sampling points around each query rather than all spatial locations — reducing the $O(H^2W^2)$ attention complexity that made vanilla DETR slow to converge.

**Pose estimation**: Adaptively sampling around body joints and limbs improves robustness to unusual poses.

## Deformable Attention vs. Deformable Convolution

Deformable attention (used in Deformable DETR) is conceptually related but distinct:

- **Deformable convolution**: Applies fixed learned weights at shifted locations — local, translation-equivariant.
- **Deformable attention**: Each query predicts reference points and attends to them with content-dependent attention weights — global, not equivariant, each query can look anywhere.

Both represent the same insight — adaptive, input-conditioned sampling patterns outperform rigid grids — but operate in fundamentally different architectural contexts.

Deformable convolutions remain a practical, low-overhead upgrade for any CNN backbone when geometric invariance is important. Replacing the $3 \times 3$ convolutions in the last few stages of a ResNet backbone with DCNv2 adds fewer than 1% parameters while consistently improving accuracy on detection and segmentation benchmarks.
