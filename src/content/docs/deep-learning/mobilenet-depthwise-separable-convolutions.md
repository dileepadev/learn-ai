---
title: MobileNet and Depthwise Separable Convolutions
description: Learn how depthwise separable convolutions factorize standard convolutions to cut computation, and how MobileNet architectures use them for on-device vision.
---

MobileNet is a family of lightweight CNNs designed for mobile and embedded devices. Its key technique is the depthwise separable convolution, which reduces computation by roughly an order of magnitude compared with standard convolution.

## Standard Convolution Cost

For a kernel size K, input channels M, output channels N, and feature map size HxW:

```text
standard cost = K * K * M * N * H * W
```

Every output channel mixes spatial and cross-channel information at once.

## Depthwise Separable Convolution

It splits the operation into two steps:

1. **Depthwise convolution**: one KxK filter per input channel (spatial filtering only)
2. **Pointwise convolution**: a 1x1 convolution that mixes channels

```text
separable cost = K*K*M*H*W  +  M*N*H*W
reduction      = 1/N + 1/K^2     (about 8-9x for 3x3 kernels)
```

```python
import torch.nn as nn

def separable_conv(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch), nn.ReLU6(),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU6(),
    )
```

## Evolution of the Family

- **MobileNetV1**: stacks of depthwise separable convolutions, plus width and resolution multipliers
- **MobileNetV2**: inverted residuals with linear bottlenecks (expand, depthwise, project, with skip connections between narrow layers)
- **MobileNetV3**: architecture search, squeeze-and-excitation, and the h-swish activation

## Trade-offs

Fewer FLOPs trade some representational capacity, and depthwise layers have low arithmetic intensity, so real speedups depend on hardware. Quantization-aware training is often used together with MobileNets for int8 deployment.
