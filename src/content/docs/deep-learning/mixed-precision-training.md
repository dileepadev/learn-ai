---
title: "Mixed Precision Training"
description: "Using FP16/BF16 mixed precision to reduce memory usage and speed up training by 2-3x."
date: "2026-06-06"
tags: ["deep-learning", "training", "performance", "mixed-precision"]
---

Mixed precision training uses lower precision (FP16/BF16) for most operations while keeping critical computations in FP32. This reduces memory by ~50% and can speed up training 2-3x.

## Why Mixed Precision

- FP16 uses half the memory of FP32
- Modern GPUs have Tensor Cores for FP16 matrix multiplication
- Most networks train well with FP16 without loss of accuracy

## PyTorch AMP (Automatic Mixed Precision)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class AMPTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()  # For gradient scaling
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward with autocast
            with autocast():
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
            
            # Backward with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
```

## Common Patterns

```python
# Simple training loop with AMP
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()

for epoch in range(epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Gradient Scaling Explained

The scaler prevents gradients from underflowing in FP16:

```python
# Without scaler:
loss.backward()  # Gradients in FP16 can underflow to 0

# With scaler:
scaler.scale(loss).backward()  # Scale up gradients
scaler.step(optimizer)          # Unscaled parameters, scaled gradients
scaler.update()                 # Adjust scaler for next iteration
```

## When AMP Works Well

| Architecture | Compatibility |
| --- | --- |
| CNNs | Excellent |
| Transformers | Excellent |
| RNNs | Good (may need tuning) |
| GANs | Good (may need care) |

## BF16 vs FP16

```python
# Use BF16 for better numerical stability
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)

# BF16 is available on A100 and newer GPUs
# FP16 has larger dynamic range issues but more precision in [0,1]
```

## Common Issues and Fixes

```python
# Issue: Loss becomes NaN
# Fix: Increase initial scale
scaler = GradScaler(init_scale=65536.0)

# Issue: Poor convergence
# Fix: Use GradScaler consistently, don't skip backward()

# Issue: Some operations fail in FP16
# Fix: Cast specific operations to FP32
with autocast():
    logits = model(inputs)
logits = logits.float()  # Ensure FP32 for softmax
```

## Memory Usage Comparison

| Precision | Memory | Speedup |
| --- | --- | --- |
| FP32 | 100% | 1x |
| FP16 | ~50% | 1.5-2x |
| FP16 + cuDNN | ~50% | 2-3x |

Mixed precision training is recommended for almost all modern deep learning tasks.