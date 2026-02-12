---
title: "Memory-Efficient Training Techniques"
description: "Gradient checkpointing, gradient accumulation, and other techniques for training large models with limited GPU memory."
date: "2026-06-06"
tags: ["deep-learning", "memory", "training", "optimization"]
---

Training large models requires careful memory management. These techniques help fit bigger models or larger batches into limited GPU memory.

## Gradient Checkpointing

Instead of storing all intermediate activations, recompute them during the backward pass:

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# For sequential models
class CheckpointedResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], checkpoint=True)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], checkpoint=True)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], checkpoint=True)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], checkpoint=True)
        
        self.fc = nn.Linear(512, 10)
    
    def _make_layer(self, block, channels, num_blocks, checkpoint=False):
        layers = []
        for i in range(num_blocks):
            layers.append(block(channels, i == 0 and 64 or channels))
        if checkpoint:
            return nn.ModuleList([checkpoint SequentialLayer(layers)])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        return self.fc(x)


# For custom modules
class CheckpointedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        ])
    
    def forward(self, x):
        # Checkpoint the middle
        def create_gradient_checkpoint_function():
            def gradient_checkpoint_function(x):
                return checkpoint(self._forward_impl, x)
            return gradient_checkpoint_function
        
        x = self.layers[0](x)
        x = checkpoint(self._forward_block, x)
        x = self.layers[-1](x)
        return x
    
    def _forward_block(self, x):
        for layer in self.layers[1:-1]:
            x = layer(x)
        return x
```

## Gradient Accumulation

Process small batches sequentially and accumulate gradients:

```python
class GradientAccumulator:
    def __init__(self, model, accumulation_steps=4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step = 0
    
    def train_step(self, inputs, targets):
        # Forward
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss = loss / self.accumulation_steps
        
        # Backward
        loss.backward()
        self.step += 1
        
        # Update weights after accumulating
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps


# Usage - simulates batch_size=128 with 4x accumulation of batch_size=32
effective_batch_size = 32 * 4
```

## Optimizer State Offloading

```python
class OffloadedOptimizer:
    def __init__(self, model, optimizer_class=torch.optim.AdamW):
        self.model = model
        self.optimizer = optimizer_class(model.parameters(), lr=0.001)
        self.device = 'cuda'
    
    def step(self):
        # Move optimizer state to CPU to save GPU memory
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Store states on CPU
                    p.data = p.data.to('cpu')
                    p.grad = p.grad.to('cpu')
        
        # Update on CPU
        self.optimizer.step()
        
        # Move back to GPU
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.data.device.type == 'cpu':
                    p.data = p.data.to(self.device)
                    p.grad = p.grad.to(self.device)
```

## Efficient Data Loading

```python
# Use persistent workers
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    persistent_workers=True,  # Keep workers alive between epochs
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2  # Prefetch 2 batches per worker
)
```

## Mixed Precision + Checkpointing

```python
def train_with_all_optimizations(model, train_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            with autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
```

## Memory Saving Summary

| Technique | Memory Saved | Speed Impact |
| --- | --- | --- |
| Gradient checkpointing | ~30-50% | Slower (recompute) |
| Gradient accumulation | ~50% (virtual batch) | Same |
| Mixed precision | ~50% | Faster |
| Optimizer offloading | Variable | Slower (CPU-GPU transfer) |

Combine techniques for maximum memory efficiency.