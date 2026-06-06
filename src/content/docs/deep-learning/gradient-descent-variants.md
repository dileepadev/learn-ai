---
title: "Gradient Descent Variants: SGD, Adam, and Beyond"
description: "Understanding stochastic gradient descent, momentum, adaptive methods, and modern optimizers for training deep networks."
date: "2026-06-06"
tags: ["deep-learning", "optimization", "training"]
---

Gradient descent is the workhorse of deep learning optimization. Understanding the mathematical foundations and practical trade-offs of different optimization algorithms is essential for training models effectively. This guide covers the progression from basic SGD to modern optimizers like AdamW and Lion.

## The Optimization Problem

Training a neural network means minimizing a loss function:

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i; \theta), y_i)$$

Where $\theta$ are the model parameters. For deep networks, this loss landscape is:

- **High-dimensional**: Millions to billions of parameters
- **Non-convex**: Many local minima and saddle points
- **Ill-conditioned**: Different directions have different curvature

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def visualize_loss_landscape():
    """Visualize a 2D slice of a loss landscape."""
    # Use a simple model for visualization
    torch.manual_seed(42)
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)
        
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    model = SimpleNet()
    
    # Loss landscape slice: vary two directions
    directions = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            directions.append(param.flatten())
    
    # This is a simplified visualization
    print("Loss landscapes are high-dimensional and complex.")
    print("Optimization algorithms navigate these landscapes efficiently.")
```

## Stochastic Gradient Descent

SGD computes gradients on mini-batches rather than the full dataset:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\mathcal{B}_t; \theta_t)$$

This provides:
- **Noise**: Helps escape sharp local minima
- **Scalability**: Works with infinite datasets
- **Speed**: Much faster than full-batch gradient descent

```python
class SGD:
    """Simple stochastic gradient descent optimizer."""
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.sub_(param.grad * self.lr)
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None


# PyTorch built-in
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## Momentum

Momentum accumulates past gradients to accelerate optimization:

$$v_{t+1} = \mu \cdot v_t + (1 - \mu) \cdot \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}$$

This creates a "velocity" that smooths out oscillations and accelerates in consistent directions.

```python
class SGDMomentum:
    """SGD with momentum."""
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        with torch.no_grad():
            for param, velocity in zip(self.params, self.velocities):
                if param.grad is not None:
                    velocity.mul_(self.momentum).add_(param.grad)
                    param.sub_(self.lr * velocity)
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None


# The momentum term accelerates along directions with consistent gradients
# and dampens oscillations in high-curvature directions
```

## Nesterov Accelerated Gradient

NAG looks ahead before taking a step:

$$v_{t+1} = \mu \cdot v_t + \eta \cdot \nabla_\theta \mathcal{L}(\theta_t - \mu \cdot v_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

This "Nesterov" momentum often provides faster convergence in practice.

```python
class NesterovMomentum:
    """Nesterov Accelerated Gradient."""
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        with torch.no_grad():
            for param, velocity in zip(self.params, self.velocities):
                if param.grad is not None:
                    # Lookahead gradient
                    lookahead = param - self.momentum * velocity
                    grad = torch.autograd.grad(
                        torch.sum(forward_pass(lookahead, self.model)),
                        self.params
                    )[0]
                    
                    velocity.mul_(self.momentum).add_(grad)
                    param.sub_(self.lr * velocity)
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None


# PyTorch: set nesterov=True
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

## Adaptive Methods: RMSprop

RMSprop adapts the learning rate per parameter based on recent gradient magnitudes:

$$E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1 - \rho) \cdot g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$$

This divides updates by the root-mean-square of recent gradients.

```python
class RMSprop:
    """RMSprop optimizer."""
    def __init__(self, params, lr=0.01, rho=0.9, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        
        self.square_avg = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        with torch.no_grad():
            for param, square_avg in zip(self.params, self.square_avg):
                if param.grad is not None:
                    # Update exponential moving average of squared gradients
                    square_avg.mul_(self.rho).addcmul_(param.grad, param.grad, value=1 - self.rho)
                    
                    # Compute update
                    param.div_(torch.sqrt(square_avg + self.eps))
                    param.sub_(param.grad * self.lr)
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None


optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
```

## Adam: Adaptive Moment Estimation

Adam combines momentum and adaptive learning rates:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

Bias correction:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Update:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

```python
class Adam:
    """Adam optimizer with bias correction."""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        self.m = [torch.zeros_like(p) for p in self.params]  # 1st moment
        self.v = [torch.zeros_like(p) for p in self.params]  # 2nd moment
        self.t = 0  # timestep
    
    def step(self):
        self.t += 1
        with torch.no_grad():
            for param, m, v in zip(self.params, self.m, self.v):
                if param.grad is not None:
                    # Update biased first moment estimate
                    m.mul_(self.beta1).add_(param.grad, alpha=1 - self.beta1)
                    
                    # Update biased second raw moment estimate
                    v.mul_(self.beta2).addcmul_(param.grad, param.grad, alpha=1 - self.beta2)
                    
                    # Compute bias-corrected estimates
                    m_hat = m / (1 - self.beta1 ** self.t)
                    v_hat = v / (1 - self.beta2 ** self.t)
                    
                    # Update parameters
                    param.sub_(m_hat * self.lr / (torch.sqrt(v_hat) + self.eps))
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None


# PyTorch built-in (use PyTorch version in practice)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

## AdamW: Decoupled Weight Decay

AdamW (Loshchilov & Hutter, 2019) decouples weight decay from adaptive learning rates:

$$\theta_{t+1} = \theta_t - \eta \cdot \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_t \right)$$

This differs from L2 regularization which is absorbed into the adaptive update.

```python
class AdamW:
    """AdamW with decoupled weight decay."""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    def step(self):
        self.t += 1
        with torch.no_grad():
            for param, m, v in zip(self.params, self.m, self.v):
                if param.grad is not None:
                    # Update biased moments
                    m.mul_(self.beta1).add_(param.grad, alpha=1 - self.beta1)
                    v.mul_(self.beta2).addcmul_(param.grad, param.grad, alpha=1 - self.beta2)
                    
                    # Bias correction
                    m_hat = m / (1 - self.beta1 ** self.t)
                    v_hat = v / (1 - self.beta2 ** self.t)
                    
                    # Decoupled weight decay
                    param.sub_(param * self.lr * self.weight_decay)
                    
                    # Parameter update
                    param.sub_(m_hat * self.lr / (torch.sqrt(v_hat) + self.eps))
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None


# PyTorch: use the weight_decay parameter correctly
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## Lion: Learning with Intelligent Optimum

Lion (Beta) is a newer optimizer that uses sign-based updates:

$$\theta_{t+1} = \theta_t - \eta \cdot \text{sign}(g_t)$$

It tracks the sign of gradients rather than their magnitude, which implicitly applies a form of clipping.

```python
class Lion:
    """Lion optimizer (sign-based)."""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    def step(self):
        self.t += 1
        with torch.no_grad():
            for param, m in zip(self.params, self.m):
                if param.grad is not None:
                    # Update momentum
                    m.mul_(self.beta1).add_(param.grad, alpha=1 - self.beta1)
                    
                    # Sign-based update
                    update = torch.sign(m)
                    
                    # Weight decay
                    if self.weight_decay > 0:
                        update = update + self.weight_decay * torch.sign(param)
                    
                    param.sub_(update * self.lr)
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None


# Lion often requires smaller learning rates than Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # AdamW
optimizer = Lion(model.parameters(), lr=1e-4)  # Lion: 3x smaller LR
```

## Optimizer Comparison

| Optimizer | Update Formula | Best For |
| --- | --- | --- |
| SGD + Momentum | $v_{t+1} = \mu v_t + g_t$ | Vision models, well-tuned hyperparameters |
| RMSprop | $g_t / \sqrt{E[g^2]_t}$ | RNNs, online learning |
| Adam | $m_t / \sqrt{v_t}$ | General purpose, default choice |
| AdamW | Adam + decoupled decay | Transformers, LLMs |
| Lion | $\text{sign}(g_t)$ | Emerging, often needs tuning |

```python
def compare_optimizers():
    """Compare different optimizers on a simple task."""
    torch.manual_seed(42)
    
    def train_with_optimizer(optimizer_class, name, epochs=100, lr=0.01):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 1))
        optimizer = optimizer_class(model.parameters(), lr=lr)
        
        losses = []
        for _ in range(epochs):
            x = torch.randn(32, 100)
            y = torch.randn(32, 1)
            
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        return losses
    
    # Comparison code (actual results vary by task)
    print("Different optimizers converge at different rates.")
```

## Learning Rate Scheduling

Learning rate scheduling is crucial for good performance:

```python
class CosineAnnealing:
    """Cosine annealing with warm restarts."""
    def __init__(self, optimizer, T_max, eta_min=0, warmup=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup = warmup
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup:
            lr = self.optimizer.param_groups[0]['lr'] * self.current_epoch / self.warmup
        else:
            progress = (self.current_epoch - self.warmup) / (self.T_max - self.warmup)
            lr = self.eta_min + 0.5 * (self.optimizer.param_groups[0]['lr'] - self.eta_min) * \
                 (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# PyTorch schedulers
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=epochs
)
```

## Practical Recommendations

- **Start with AdamW**: Good default for most tasks
- **SGD + momentum**: Often better for CNNs with extensive hyperparameter tuning
- **Lion**: Emerging option, can outperform but requires careful tuning
- **Learning rate**: Start with 1e-3 for AdamW, 0.01 for SGD, 1e-4 for Lion
- **Weight decay**: 0.01-0.1 for AdamW, 0.1-1.0 for Lion
- **Use scheduler**: Cosine annealing or one-cycle policy
- **Gradient clipping**: Essential for RNNs and very deep networks