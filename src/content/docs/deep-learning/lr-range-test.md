---
title: "Learning Rate Range Test"
description: "Finding optimal learning rates using the LR range test for better training convergence."
date: "2026-06-06"
tags: ["deep-learning", "optimization", "learning-rate"]
---

The learning rate range test helps identify the optimal learning rate range for training neural networks.

## What Is the LR Range Test

The test trains the model for several epochs while gradually increasing the learning rate from a very small to a very large value. By plotting loss vs. learning rate, we can identify:

- **Minimum learning rate**: Below which there's no improvement
- **Optimal learning rate**: Where loss decreases fastest
- **Maximum learning rate**: Where loss starts to diverge

## Implementing the LR Range Test

```python
import matplotlib.pyplot as plt
import numpy as np

def lr_range_test(model, train_loader, device, start_lr=1e-7, end_lr=10, 
                  epochs=1, accumulation_steps=1):
    """
    Run learning rate range test.
    
    Returns: losses and learning rates for plotting.
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
    
    num_batches = len(train_loader) * epochs
    lr_schedule = np.exp(np.linspace(np.log(start_lr), np.log(end_lr), num_batches))
    
    losses = []
    learning_rates = []
    smoothed_losses = []
    
    model.train()
    running_loss = 0.0
    
    epoch_count = 0
    batch_count = 0
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            # Update learning rate
            lr = lr_schedule[batch_count]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_count + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            learning_rates.append(lr)
            losses.append(running_loss)
            running_loss = 0.0
            
            batch_count += 1
        
        epoch_count += 1
    
    # Smooth losses for cleaner visualization
    window_size = 20
    for i in range(len(losses)):
        start = max(0, i - window_size)
        smoothed_losses.append(np.mean(losses[start:i+1]))
    
    return learning_rates, losses, smoothed_losses
```

## Plotting and Analysis

```python
def plot_lr_range_test(lrs, losses, smoothed_losses=None):
    """Plot LR range test results."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(lrs, losses, alpha=0.3, color='blue')
    if smoothed_losses:
        plt.plot(lrs, smoothed_losses, color='blue', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(lrs, np.log(losses), alpha=0.3, color='blue')
    if smoothed_losses:
        plt.plot(lrs, np.log(smoothed_losses), color='blue', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Log Loss')
    plt.title('Log Loss vs Learning Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lr_range_test.png')
    plt.show()


# Usage
lrs, losses, smoothed = lr_range_test(
    model, train_loader, device='cuda',
    start_lr=1e-6, end_lr=10, epochs=1
)
plot_lr_range_test(lrs, losses, smoothed)
```

## Interpreting Results

```
Loss curve typically has three regions:

1. Too Low LR (left flat region)
   - Loss doesn't decrease
   - Learning rate: 1e-7 to 1e-5
   
2. Optimal LR (steepest descent)
   - Loss decreases fastest
   - Learning rate: 1e-4 to 1e-2
   
3. Too High LR (right increasing region)
   - Loss starts to increase
   - Learning rate: > 0.1
```

## Selecting Learning Rate from Test

```python
def find_optimal_lr(lrs, losses, smoothed_losses):
    """Find optimal learning rate from range test."""
    # Use smoothed losses
    losses = smoothed_losses
    
    # Find minimum loss and its index
    min_idx = np.argmin(losses)
    min_loss = losses[min_idx]
    min_lr = lrs[min_idx]
    
    # Find left boundary (where loss starts decreasing significantly)
    for i in range(min_idx, -1, -1):
        if losses[i] > losses[0] * 0.95:  # Within 5% of initial loss
            left_lr = lrs[i]
            break
    
    # Recommended: start at left boundary, peak around 10x min_lr
    max_lr = min_lr * 10
    
    return {
        'min_loss_lr': min_lr,
        'recommended_min_lr': left_lr,
        'recommended_max_lr': max_lr,
        'stochastic_range_min': lrs[min_idx // 2],
        'stochastic_range_max': min_lr * 10
    }


# For one-cycle policy:
# min_lr = recommended_min_lr
# max_lr = recommended_max_lr
# div_factor = max_lr / min_lr  # Usually 10-20
```

## Practical Recommendations

- Run the test for 1 epoch on a subset of data
- Use SGD with momentum for accurate results
- Plot both raw and smoothed losses
- The optimal LR is typically where loss decreases fastest
- For one-cycle policy: max_lr = 10×min_lr
- If loss diverges: decrease end_lr and retest