---
title: "Convolutional Layers in Detail"
description: "Understanding convolutions — operations, strides, padding, dilation, and transposed convolutions."
date: "2026-06-06"
tags: ["deep-learning", "convolutional-neural-networks", "computer-vision"]
---

Convolutional layers apply learned filters to extract features from images. Understanding the various parameters is essential for designing effective CNNs.

## Standard Convolution

```python
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
    
    def forward(self, x):
        # Manual convolution implementation
        batch, in_ch, in_h, in_w = x.shape
        k = self.kernel_size
        stride = self.stride
        pad = self.padding
        dilation = self.dilation
        
        # Output dimensions
        out_h = (in_h + 2*pad - dilation*(k-1) - 1) // stride + 1
        out_w = (in_w + 2*pad - dilation*(k-1) - 1) // stride + 1
        
        # Unfold input
        x_padded = F.pad(x, (pad, pad, pad, pad))
        unfolded = x_padded.unfold(2, k + (k-1)*(dilation-1), stride * dilation)
        unfolded = unfolded.unfold(3, k + (k-1)*(dilation-1), stride * dilation)
        
        # Convolution
        output = torch.einsum('bchw, ocpq -> boqw', unfolded, self.weight)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output


# PyTorch built-in
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
```

## Padding

Controls output size and influences border handling:

```python
# Same padding (output size equals input size)
conv_same = nn.Conv2d(64, 64, 3, padding=1)  # For stride=1

# Valid padding (no padding)
conv_valid = nn.Conv2d(64, 64, 3, padding=0)

# Calculate output size
output_size = (input_size + 2*padding - kernel_size) // stride + 1
```

## Dilation

Dilation introduces gaps between kernel elements:

```python
# Dilation 1: standard convolution
conv_d1 = nn.Conv2d(64, 64, 3, dilation=1)

# Dilation 2: receptive field doubles
conv_d2 = nn.Conv2d(64, 64, 3, dilation=2)

# Dilation 4
conv_d4 = nn.Conv2d(64, 64, 3, dilation=4)

# Output size with dilation
output_size = (input_size + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
```

## Grouped Convolutions

Channels are split into groups that convolve independently:

```python
# Depthwise convolution
conv_dw = nn.Conv2d(64, 64, 3, groups=64, padding=1)

# Group convolution (used in ResNeXt)
conv_group = nn.Conv2d(64, 128, 3, groups=32, padding=1)

# Calculate parameters saved
# Depthwise: 3*3*64 = 576 params vs 3*3*64*64 = 36864 for full conv
```

## Pointwise Convolution (1x1)

Changes channel dimension efficiently:

```python
# 1x1 convolution
conv_1x1 = nn.Conv2d(64, 128, 1)

# Use cases:
# - Channel reduction/expansion
# - Cross-channel mixing
# - Efficient alternative to fully connected
```

## Transposed Convolution

Upsamples spatial dimensions (learnable upsampling):

```python
class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
    
    def forward(self, x):
        # Simplified transposed convolution
        batch, in_ch, in_h, in_w = x.shape
        k = self.weight.shape[2]
        
        # Output size
        out_h = (in_h - 1) * stride + k
        out_w = (in_w - 1) * stride + k
        
        # Full convolution with input zeros inserted
        x_expanded = torch.zeros(batch, in_ch, in_h * stride, in_w * stride, device=x.device)
        x_expanded[:, :, ::stride, ::stride] = x
        
        return F.conv2d(x_expanded, self.weight, self.bias)


# PyTorch built-in
deconv = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
```

## Depthwise Separable Convolution

Combines depthwise and pointwise convolutions:

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # Depthwise: one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            padding=1, groups=in_channels
        )
        # Pointwise: combine channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


# Used in MobileNet, EfficientNet
# Reduces parameters: k*k*in_ch*1 + in_ch*out_ch*1*1
# vs k*k*in_ch*out_ch
```

## Practical Recommendations

| Scenario | Kernel Size | Padding | Recommendation |
| --- | --- | --- | --- |
| Early layers | 7x7 or 3x3 | Same | Capture low-level features |
| Deep layers | 3x3 | Same | Balance receptive field and params |
| Downsampling | 3x3 with stride 2 | 1 | Combine conv + pooling |
| Upsampling | Transposed or resize + conv | - | Prefer resize + conv for artifacts |
| Mobile nets | 3x3 separable | - | Depthwise separable |