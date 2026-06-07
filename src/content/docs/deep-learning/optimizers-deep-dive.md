---
title: "Deep Dive into Deep Learning Optimizers"
description: "A comprehensive exploration of optimization algorithms beyond Adam — LAMB, LARS, and second-order methods."
date: "2026-06-06"
tags: ["deep-learning", "optimization", "training"]
---

While Adam is the default choice for many tasks, specialized optimizers offer advantages for specific scenarios.

## LAMB: Layer-wise Adaptive Moments for Big Batch Training

LAMB adjusts learning rates per layer and per parameter, enabling stable training with very large batch sizes:

```python
class LAMB:
    """Layer-wise Adaptive Moments for Big batch training."""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
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
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Update biased first moment estimate
                self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                
                # Update biased second raw moment estimate
                self.v[i].mul_(self.beta2).addcmul_(grad, grad, alpha=1 - self.beta2)
                
                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Decoupled weight decay
                param.sub_(param * self.lr * self.weight_decay)
                
                # LAMB: update with layer-wise normalization
                param_norm = param.norm()
                grad_norm = grad.norm()
                
                if param_norm != 0 and grad_norm != 0:
                    ratio = (param_norm / grad_norm).clamp(0, 10)
                else:
                    ratio = 1
                
                param.sub_(m_hat / (torch.sqrt(v_hat) + self.eps) * self.lr * ratio)
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None


# LAMB enables BERT training with batch size 4096+
# Used in training large transformers with large batches
```

## LARS: Layer-wise Adaptive Rate Scaling

Adds layer-wise learning rate scaling to SGD:

```python
class LARS:
    """Layer-wise Adaptive Rate Scaling."""
    def __init__(self, params, lr=0.001, momentum=0.9, 
                 weight_decay=0.01, eps=1e-5):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eps = eps
        
        self.velocities = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # LARS coefficient
                param_norm = param.norm()
                grad_norm = grad.norm()
                
                if param_norm > 0 and grad_norm > 0:
                    lr = self.lr * param_norm / (grad_norm + self.weight_decay * param_norm + self.eps)
                else:
                    lr = self.lr
                
                # Momentum update
                self.velocities[i].mul_(self.momentum).add_(grad)
                param.sub_(self.velocities[i] * lr)


# LARS is primarily for training with very large batch sizes
# Often combined with LAMB
```

## AdaGrad

Accumulating learning rates for sparse features:

```python
class AdaGrad:
    def __init__(self, params, lr=0.01, eps=1e-10):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        
        self.sum_squared = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Accumulate squared gradients
                self.sum_squared[i].addcmul_(grad, grad)
                
                # Update
                param.sub_(grad * self.lr / (torch.sqrt(self.sum_squared[i]) + self.eps))
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None


# AdaGrad can work well for sparse features
# But learning rate often becomes too small over time
```

## AdaDelta

Self-tuning learning rate without setting absolute LR:

```python
class AdaDelta:
    def __init__(self, params, rho=0.9, eps=1e-6):
        self.params = list(params)
        self.rho = rho
        self.eps = eps
        
        self.eg2 = [torch.zeros_like(p) for p in self.params]
        self.edelta2 = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Update running average of squared gradients
                self.eg2[i].mul_(self.rho).addcmul_(grad, grad, alpha=1 - self.rho)
                
                # Compute update
                delta = torch.sqrt(self.edelta2[i] + self.eps) / \
                        torch.sqrt(self.eg2[i] + self.eps) * grad
                
                # Update parameters
                param.sub_(delta)
                
                # Update running average of squared deltas
                self.edelta2[i].mul_(self.rho).addcmul_(delta, delta, alpha=1 - self.rho)
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None


# AdaDelta doesn't need a learning rate setting
# Used in some RNN applications
```

## Comparison and Selection

| Optimizer | Learning Rate | Best For |
| --- | --- | --- |
| SGD + Momentum | 0.01 - 0.1 | Well-tuned CNNs, vision |
| Adam | 1e-4 - 1e-3 | General purpose, transformers |
| AdamW | 1e-4 - 1e-3 | Transformers, modern architectures |
| LAMB | 1e-3 - 1e-2 | Large batch transformer training |
| AdaGrad | 0.01 - 1.0 | Sparse features |
| AdaDelta | None needed | When LR tuning is difficult |

For most modern deep learning, AdamW with a cosine scheduler is a safe starting point.