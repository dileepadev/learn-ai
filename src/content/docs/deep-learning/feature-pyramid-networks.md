---
title: Feature Pyramid Networks
description: Learn Feature Pyramid Networks (FPN) — the multi-scale feature fusion architecture that revolutionized object detection and segmentation by combining deep, semantically rich features with high-resolution spatial features through a top-down pathway and lateral connections, enabling accurate detection of objects at vastly different scales.
---

**Feature Pyramid Networks** (Lin et al., Facebook AI Research, 2017) solved one of the most persistent challenges in computer vision: detecting and segmenting objects that appear at wildly different scales within the same image. A car occupying half the frame and a distant pedestrian just 20 pixels tall require fundamentally different detection strategies — and FPN provides a principled architecture for handling both simultaneously.

Before FPN, detectors either used image pyramids (slow, requires multiple forward passes) or processed features at a single scale (fast but poor at small objects). FPN introduced a **feature pyramid constructed from a single-scale input using a top-down pathway with lateral connections**, achieving the accuracy of image pyramids at roughly the cost of a single-scale detector.

## The Multi-Scale Problem

Convolutional neural networks naturally produce a spatial hierarchy of features. As an image passes through successive convolution and pooling layers, spatial resolution decreases while semantic richness (the "what") increases:

- **Early layers** (e.g., after conv1): high resolution (224×224), low-level features (edges, textures)
- **Middle layers**: medium resolution (56×56), mid-level features (parts, patterns)
- **Deep layers** (e.g., after conv5): low resolution (7×7), high-level semantic features (objects, categories)

The dilemma: small objects need high-resolution features (to not be "missed" by low resolution), but accurate recognition needs deep, semantic features. FPN resolves this by **combining the two via a top-down pathway**.

## FPN Architecture

FPN wraps any backbone CNN (ResNet, MobileNet, etc.) with two pathways:

```
Bottom-up pathway (backbone CNN forward pass):
  C2 (1/4)  → C3 (1/8)  → C4 (1/16) → C5 (1/32)
  56×56       28×28        14×14         7×7

Top-down pathway + lateral connections:
  P5 = 1×1 conv(C5)
         ↓ upsample 2×
  P4 = P5_upsampled + 1×1 conv(C4)
         ↓ upsample 2×
  P3 = P4_upsampled + 1×1 conv(C3)
         ↓ upsample 2×
  P2 = P3_upsampled + 1×1 conv(C2)

Final: 3×3 conv applied to each Pn to reduce aliasing
```

Each level of the output pyramid $\{P2, P3, P4, P5\}$ has the **same channel dimension** (256 by default) but different spatial resolutions. Downstream heads (detection, segmentation) attach to all pyramid levels simultaneously.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class FPN(nn.Module):
    """
    Feature Pyramid Network wrapping a ResNet-50 backbone.
    Outputs feature maps {P2, P3, P4, P5} with 256 channels each.
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        # Load pretrained backbone; capture intermediate feature maps
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Bottom-up stages (C2..C5 in FPN notation)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool,
                                    backbone.layer1)   # → 256 ch, stride 4
        self.layer2 = backbone.layer2                  # → 512 ch, stride 8
        self.layer3 = backbone.layer3                  # → 1024 ch, stride 16
        self.layer4 = backbone.layer4                  # → 2048 ch, stride 32

        # Lateral 1×1 convolutions: project backbone channels → out_channels
        self.lat2 = nn.Conv2d(256,  out_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(512,  out_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lat5 = nn.Conv2d(2048, out_channels, kernel_size=1)

        # Output 3×3 convolutions: smooth merged features, reduce aliasing
        self.out2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.lat2, self.lat3, self.lat4, self.lat5,
                  self.out2, self.out3, self.out4, self.out5]:
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Bottom-up: extract multi-scale feature maps
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down pathway with lateral connections
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        # 3×3 conv to finalize each level
        return {
            "P2": self.out2(p2),   # stride 4  — best for small objects
            "P3": self.out3(p3),   # stride 8
            "P4": self.out4(p4),   # stride 16
            "P5": self.out5(p5),   # stride 32 — best for large objects
        }


# Usage
model = FPN(out_channels=256)
img = torch.randn(2, 3, 800, 800)
features = model(img)
for name, feat in features.items():
    print(f"{name}: {feat.shape}")
# P2: torch.Size([2, 256, 200, 200])
# P3: torch.Size([2, 256, 100, 100])
# P4: torch.Size([2, 256, 50, 50])
# P5: torch.Size([2, 256, 25, 25])
```

## Assigning Anchors to Pyramid Levels

A key design decision is which pyramid level handles which object sizes. The standard assignment rule maps anchor scale $w \times h$ to level $k$:

$$k = \lfloor k_0 + \log_2(\sqrt{wh} / 224) \rfloor$$

where $k_0 = 4$ is the reference level for a 224-pixel object. Small anchors (e.g., 32×32) go to P2; large anchors (e.g., 512×512) go to P5.

```python
import math

def assign_level(anchor_w: float, anchor_h: float,
                 k0: int = 4, min_level: int = 2, max_level: int = 5) -> int:
    """
    Assign an anchor box to the appropriate FPN level.
    anchor_w, anchor_h: anchor dimensions in pixels
    """
    scale = math.sqrt(anchor_w * anchor_h)
    level = k0 + math.log2(scale / 224.0)
    return max(min_level, min(max_level, round(level)))

# Examples
print(assign_level(32, 32))    # → 2 (small objects → P2)
print(assign_level(128, 128))  # → 3
print(assign_level(512, 512))  # → 5 (large objects → P5)
```

## FPN in Downstream Architectures

**Mask R-CNN**: Uses FPN as its backbone feature extractor. Each region proposal from the RPN is assigned to the appropriate pyramid level using the anchor assignment rule, then RoIAlign extracts fixed-size features for classification and mask prediction.

**RetinaNet**: Attaches a shared classification head and box regression head to every FPN level. Adds **P6** (subsampled from P5) and **P7** (subsampled from P6) for detecting very large objects.

**FCOS / CenterPoint**: Anchor-free detectors that assign each FPN level a specific range of object scales, eliminating the need for anchor hyperparameter tuning entirely.

## Variants and Improvements

### Path Aggregation Network (PANet)

PANet (Liu et al., 2018) adds a **bottom-up path augmentation** on top of FPN, creating a second information flow from low-level features back up to deep levels:

```
FPN top-down: C5 → P5 → P4 → P3 → P2
PANet bottom-up: P2 → N3 → N4 → N5
```

The intuition: low-level features (edges, textures) contain localization information useful for instance segmentation. The extra bottom-up path brings this information to all levels in fewer hops (4 vs. 100+ for FPN alone).

### BiFPN (Bi-directional FPN)

BiFPN (EfficientDet, Tan et al., 2020) adds **weighted feature fusion** — learning the importance of each input feature map rather than summing them equally:

$$P_{\text{out}}^j = \text{BN}\!\left(\text{Conv}\!\left(\sum_i \frac{w_i}{\epsilon + \sum_k w_k} \cdot P_i\right)\right)$$

BiFPN also removes nodes with only one input edge (which don't contribute multi-scale fusion) and stacks the BiFPN block multiple times to increase receptive field.

### FPN for Semantic Segmentation

In segmentation networks like **Panoptic FPN** and **Semantic FPN**, the pyramid levels are upsampled and merged into a single high-resolution feature map before the segmentation head:

```python
class SemanticFPNHead(nn.Module):
    """
    Merge all FPN levels into a single 1/4-resolution prediction map.
    """
    def __init__(self, in_channels=256, num_classes=80):
        super().__init__()
        # Upsample P3, P4, P5 to match P2 resolution (stride 4)
        self.upsamples = nn.ModuleDict({
            "P3": nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True)),
            "P4": nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True)),
            "P5": nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True)),
        })
        self.final = nn.Sequential(
            nn.Conv2d(256 + 128 * 3, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, features: dict) -> torch.Tensor:
        p2_shape = features["P2"].shape[-2:]
        p3_up = F.interpolate(self.upsamples["P3"](features["P3"]), p2_shape, mode="bilinear", align_corners=False)
        p4_up = F.interpolate(self.upsamples["P4"](features["P4"]), p2_shape, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(self.upsamples["P5"](features["P5"]), p2_shape, mode="bilinear", align_corners=False)
        merged = torch.cat([features["P2"], p3_up, p4_up, p5_up], dim=1)
        return self.final(merged)
```

FPN's top-down pathway with lateral connections has become a near-universal building block in dense prediction architectures. It appears in virtually every state-of-the-art object detector, instance segmentation model, and panoptic segmentation system — making it one of the most practically impactful architectural contributions in the history of computer vision.
