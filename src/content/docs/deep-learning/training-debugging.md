---
title: "Training Debugging and Validation"
description: "Diagnosing training issues — debugging vanishing gradients, loss not decreasing, and overfitting."
date: "2026-06-06"
tags: ["deep-learning", "debugging", "training"]
---

Training deep networks often encounters issues. Systematic debugging helps identify and fix problems.

## Loss Not Decreasing

```python
def diagnose_loss_not_decreasing(model, train_loader, device):
    """Check common causes of non-converging models."""
    
    # 1. Check learning rate
    for param_group in model.optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']}")
    
    # 2. Check gradient norms
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs.to(device))
        loss = F.cross_entropy(outputs, targets.to(device))
        loss.backward()
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        print(f"Gradient norm: {total_norm:.6f}")
        
        if total_norm < 1e-5:
            print("ISSUE: Vanishing gradients")
        elif total_norm > 100:
            print("ISSUE: Exploding gradients - try gradient clipping")
        
        model.zero_grad()
        break
    
    # 3. Check forward pass
    with torch.no_grad():
        for inputs, _ in train_loader:
            outputs = model(inputs[:1].to(device))
            print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
            print(f"Output mean: {outputs.mean():.4f}")
            if outputs.abs().max() < 0.1:
                print("ISSUE: Outputs too small - check initialization")
            break
```

## Checking Gradients

```python
def check_gradients(model):
    """Inspect gradient statistics."""
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats[name] = {
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item(),
                'min': param.grad.min().item(),
                'max': param.grad.max().item(),
                'has_nan': torch.isnan(param.grad).any().item(),
                'has_inf': torch.isinf(param.grad).any().item()
            }
    
    # Print problematic gradients
    for name, stats in grad_stats.items():
        if stats['has_nan'] or stats['has_inf']:
            print(f"NaN/Inf in {name}")
        if abs(stats['mean']) > 1.0 or stats['std'] > 1.0:
            print(f"Large gradient in {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    return grad_stats
```

## Overfitting Detection

```python
def check_overfitting(train_loader, val_loader, model, device):
    """Check if model is overfitting."""
    model.eval()
    
    # Training accuracy
    train_correct, train_total = 0, 0
    with torch.no_grad():
        for inputs, targets in train_loader:
            outputs = model(inputs.to(device))
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets.to(device)).sum().item()
            train_total += targets.size(0)
    train_acc = 100. * train_correct / train_total
    
    # Validation accuracy
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.to(device))
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(targets.to(device)).sum().item()
            val_total += targets.size(0)
    val_acc = 100. * val_correct / val_total
    
    print(f"Train Acc: {train_acc:.2f}%")
    print(f"Val Acc: {val_acc:.2f}%")
    
    if train_acc - val_acc > 15:
        print("OVERFITTING: Large gap between train and val accuracy")
        print("Suggestions: Add dropout, data augmentation, weight decay, early stopping")
    elif train_acc < 70 and val_acc < 70:
        print("UNDERFITTING: Both train and val accuracy are low")
        print("Suggestions: Increase model capacity, reduce regularization, increase learning rate")
```

## Training Loss Diagnostics

```python
def analyze_loss_curve(loss_history):
    """Analyze training loss curve for issues."""
    if len(loss_history) < 10:
        return
    
    # Calculate rate of decrease
    recent_loss = loss_history[-10:]
    early_loss = loss_history[:10]
    
    avg_recent = sum(recent_loss) / len(recent_loss)
    avg_early = sum(early_loss) / len(early_loss)
    
    if avg_recent > avg_early * 0.95:
        print("Loss not decreasing - consider adjusting learning rate")
    
    # Check for instability
    if len(loss_history) > 100:
        recent_std = np.std(loss_history[-100:])
        if recent_std > 0.5:
            print("Training unstable - consider gradient clipping or smaller LR")
    
    # Check for NaN
    if any(math.isnan(x) for x in loss_history):
        print("NaN loss detected - check input data, reduce learning rate")
```

## Quick Debugging Checklist

1. **Loss stays at NaN**: Reduce LR, check data normalization
2. **Loss oscillates**: Reduce LR, increase batch size
3. **Loss plateaus early**: Increase model capacity, check initialization
4. **Train works, val fails**: Add regularization, use dropout
5. **Gradients are zero**: Check activation functions, learning rate
6. **Output is NaN**: Check data for NaN values, use gradient clipping

Systematic debugging saves time and improves results.