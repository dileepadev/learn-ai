---
title: "Learning Rate Scheduling Strategies"
description: "Understanding cosine annealing, step decay, warmup, and other learning rate schedules for training neural networks effectively."
date: "2026-06-06"
tags: ["deep-learning", "optimization", "training"]
---

Learning rate scheduling is one of the most impactful hyperparameters in training neural networks. The right schedule can mean the difference between quick convergence and getting stuck in poor local minima.

## Why Schedule the Learning Rate

During training, the loss landscape changes as parameters move. Early in training, large steps help escape poor initializations. Later, smaller steps fine-tune the solution. A well-designed schedule accounts for this.

## Common Scheduling Strategies

### Step Decay

Reduce the learning rate by a factor at specific epochs:

```python
class StepDecay:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma


# Simple step decay
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)
```

### Cosine Annealing

Smooth decay following a cosine curve:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

```python
class CosineAnnealing:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min + 0.5 * (
                param_group['initial_lr'] - self.eta_min
            ) * (1 + math.cos(math.pi * self.epoch / self.T_max))


# PyTorch built-in
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)
```

### Cosine Annealing with Warm Restarts

Restart the schedule periodically with different period lengths:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

### One-Cycle Policy

Rapid warmup, followed by a cosine anneal, and a final fine-tuning phase:

```python
class OneCycleLR:
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.epoch = 0
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        epoch_pct = self.step_num / self.total_steps
        
        if epoch_pct < self.pct_start:
            # Warmup phase: linear increase
            lr = self.max_lr * epoch_pct / self.pct_start
        else:
            # Annealing phase: cosine decay
            progress = (epoch_pct - self.pct_start) / (1 - self.pct_start)
            lr = self.eta_min + 0.5 * (self.max_lr - self.eta_min) * (
                1 + math.cos(math.pi * progress)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=10000
)
```

## Learning Rate Warmup

Warmup gradually increases the learning rate from a small value to the target rate:

```python
class GradualWarmup:
    def __init__(self, optimizer, warmup_epochs, target_lr, after_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.target_lr * self.epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.after_scheduler.step()


# Combine warmup with cosine annealing
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100
)
scheduler = GradualWarmup(optimizer, warmup_epochs=5, 
                          target_lr=0.01, after_scheduler=base_scheduler)
```

## Practical Recommendations

- **One-cycle**: Excellent default for most vision tasks
- **Cosine with restarts**: Good for transformers and deep networks
- **Step decay**: Simple, works well when you know approximately when to decay
- **Warmup**: Essential for transformers (use 500-1000 steps)
- **Monitor**: Use learning rate range test to find good bounds