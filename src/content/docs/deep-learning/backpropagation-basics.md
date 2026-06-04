---
title: "Backpropagation: How Neural Networks Learn"
description: "Understand backpropagation — the core algorithm that enables neural networks to learn from data, from forward passes to gradient computation and weight updates."
---

Backpropagation is the algorithm that makes neural networks learn. It computes how each weight in the network contributes to the output error, enabling efficient optimization via gradient descent.

## The Learning Problem

Neural networks learn by adjusting weights to minimize a loss function. For a network with millions of weights, computing gradients analytically is impossible. Backpropagation provides an efficient algorithm using the chain rule.

## Forward Pass: Computing Outputs

```python
import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Example: Forward pass
model = SimpleNetwork()
x = torch.tensor([[0.5, 1.0]])  # Input: 2 features
output = model(x)

print(f"Input: {x}")
print(f"Output: {output}")
# Output depends on learned weights
```

## The Chain Rule: Foundation of Backpropagation

For a function composition f(g(x)), the chain rule states:

```
df/dx = df/dg * dg/dx
```

For neural networks with many layers, this extends to:

```
∂L/∂W¹ = ∂L/∂y * ∂y/∂a * ∂a/∂W¹
```

Where L is loss, y is output, a is pre-activation, and W are weights.

## Backward Pass: Computing Gradients

```python
# Set up
model = SimpleNetwork()
x = torch.tensor([[0.5, 1.0]])
y_true = torch.tensor([[1.0]])  # Target

# Forward pass
y_pred = model(x)

# Compute loss
criterion = nn.BCELoss()
loss = criterion(y_pred, y_true)

print(f"Loss: {loss.item()}")

# Backward pass (automatically computed by PyTorch)
loss.backward()

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad shape {param.grad.shape}, mean {param.grad.mean():.4f}")
```

## Step-by-Step Manual Backpropagation

```python
def manual_backprop():
    """Manual computation of gradients for a simple network."""
    import numpy as np
    
    # Simple network: 2 inputs -> 2 hidden -> 1 output
    np.random.seed(42)
    
    # Initialize weights
    W1 = np.random.randn(2, 2)  # Input to hidden
    W2 = np.random.randn(2, 1)  # Hidden to output
    
    # Forward pass
    x = np.array([[0.5, 1.0]])
    z1 = x @ W1  # Pre-activation
    a1 = np.tanh(z1)  # Activation
    z2 = a1 @ W2
    y_pred = 1 / (1 + np.exp(-z2))  # Sigmoid
    
    # Loss (MSE)
    y_true = np.array([[1.0]])
    loss = 0.5 * np.sum((y_pred - y_true) ** 2)
    
    # Backward pass
    dL_dy_pred = y_pred - y_true
    
    # Output layer gradients
    dy_pred_dz2 = y_pred * (1 - y_pred)  # Sigmoid derivative
    dz2_dW2 = a1
    
    dL_dW2 = dz2_dW2.T @ (dL_dy_pred * dy_pred_dz2)
    dL_da1 = (dL_dy_pred * dy_pred_dz2) @ W2.T
    
    # Hidden layer gradients
    da1_dz1 = 1 - a1 ** 2  # Tanh derivative
    dz1_dW1 = x
    
    dL_dW1 = dz1_dW1.T @ (dL_da1 * da1_dz1)
    
    print(f"Loss: {loss}")
    print(f"dL/dW1 shape: {dL_dW1.shape}")
    print(f"dL/dW2 shape: {dL_dW2.shape}")
    
    return dL_dW1, dL_dW2
```

## Gradient Descent Update

After computing gradients, update weights:

```python
# Learning rate
lr = 0.1

# Update weights (manually)
with torch.no_grad():
    for param in model.parameters():
        param -= lr * param.grad

# Or use optimizer (preferred)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.step()
```

## Computational Graph and Autograd

PyTorch's autograd system builds a computational graph automatically:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

# PyTorch builds: x -> x^2 -> + -> 3x -> + -> y
# Each operation records its gradient function

# Backward pass computes all gradients
y.backward()

print(x.grad)  # dy/dx = 2x + 3 = 7
```

### How Autograd Works

```python
# Each operation creates a Function node
# Function nodes store:
# 1. Reference to input tensors
# 2. Gradient function (backward)
# 3. Accumulate gradients during backward

# Example: x ** 2
class SquareBackward:
    @staticmethod
    def apply(ctx, input):
        # Forward: stores input for backward
        ctx.save_for_backward(input)
        return input * input
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward: d(x^2)/dx = 2x
        input, = ctx.saved_tensors
        return grad_output * (2 * input)
```

## Common Pitfalls and Solutions

### Vanishing Gradients

```python
# Problem: Gradients shrink exponentially in deep networks
# Sigmoid derivative max is 0.25, so gradients diminish by 75%+ per layer

# Solution 1: Use ReLU instead of sigmoid
model = nn.Sequential(
    nn.Linear(100, 100),
    nn.ReLU(),  # Derivative is 1 for x > 0
    nn.Linear(100, 100),
    nn.ReLU(),
)

# Solution 2: Batch normalization
model = nn.Sequential(
    nn.Linear(100, 100),
    nn.BatchNorm1d(100),
    nn.ReLU(),
)
```

### Exploding Gradients

```python
# Problem: Gradients grow exponentially

# Solution: Gradient clipping
max_norm = 1.0

# Before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

### Dead Neurons

```python
# Problem: ReLU neurons can "die" (always output 0)
# If pre-activation is always negative, gradient is 0

# Solution: Leaky ReLU
model = nn.Sequential(
    nn.Linear(100, 100),
    nn.LeakyReLU(0.01),  # Allows small gradient when x < 0
)

# Solution: ELU (exponential linear unit)
model = nn.Sequential(
    nn.Linear(100, 100),
    nn.ELU(),
)
```

## Advanced Gradient Computation

### Gradient Checkpointing (Memory Optimization)

```python
# Trade computation for memory in backward pass
from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return checkpoint(self.layer, x)  # Recompute this layer in backward
```

### Custom Autograd Functions

```python
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Forward computation
        result = x ** 2 + torch.sin(x)
        ctx.save_for_backward(x)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward: d/dx(x^2 + sin(x)) = 2x + cos(x)
        x, = ctx.saved_tensors
        grad_input = grad_output * (2 * x + torch.cos(x))
        return grad_input

# Usage
x = torch.tensor([1.0], requires_grad=True)
y = CustomFunction.apply(x)
y.backward()
print(x.grad)  # 2 + cos(1)
```

## Monitoring Training

```python
def train_epoch(model, dataloader, optimizer, criterion):
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        x, y = batch
        
        # Forward
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

# Track gradients during training
def log_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: norm = {grad_norm:.4f}")
```

## Efficient Implementations

Backpropagation efficiency depends on:

1. **Memory layout**: Contiguous tensors, proper strides.
2. **Kernel fusion**: Combine operations to reduce memory access.
3. **Mixed precision**: Use fp16 for most computations, fp32 for master weights.

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    x, y = batch
    
    with autocast():
        y_pred = model(x)
        loss = criterion(y_pred, y)
    
    scaler.scale(loss).backward()
    
    scaler.step(optimizer)
    scaler.update()
```

Backpropagation is elegant in its simplicity (just chain rule) yet powerful enough to train networks with billions of parameters. Understanding it deeply helps you debug training issues and design better architectures.