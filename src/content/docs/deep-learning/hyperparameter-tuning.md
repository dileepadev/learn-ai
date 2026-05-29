---
title: "Hyperparameter Tuning for Deep Learning"
description: "Systematic approaches to tuning learning rates, batch sizes, and other hyperparameters for optimal model performance."
date: "2026-06-06"
tags: ["deep-learning", "hyperparameters", "tuning"]
---

Hyperparameter tuning is often the difference between a working model and a good model. Systematic approaches save time and improve results.

## Key Hyperparameters

| Parameter | Typical Range | Impact |
| --- | --- | --- |
| Learning rate | 1e-5 to 1e-1 | Most important |
| Batch size | 16 to 512 | Affects convergence |
| Weight decay | 1e-6 to 1e-2 | Regularization |
| Dropout | 0.0 to 0.7 | Overfitting |
| Hidden dimensions | 64 to 4096 | Model capacity |

## Grid Search

```python
def grid_search(model_class, param_grid, train_loader, val_loader, device):
    """Simple grid search over hyperparameters."""
    results = []
    
    for lr in param_grid['lr']:
        for batch_size in param_grid['batch_size']:
            for wd in param_grid.get('weight_decay', [0.0]):
                model = model_class().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                
                # Train for fixed epochs
                best_val_acc = 0.0
                for epoch in range(param_grid['epochs']):
                    train_one_epoch(model, train_loader, optimizer)
                    val_acc = evaluate(model, val_loader)
                    best_val_acc = max(best_val_acc, val_acc)
                
                results.append({
                    'params': {'lr': lr, 'batch_size': batch_size, 'weight_decay': wd},
                    'val_acc': best_val_acc
                })
    
    return sorted(results, key=lambda x: x['val_acc'], reverse=True)
```

## Random Search

Often more efficient than grid search:

```python
import random

def random_search(model_class, param_distributions, n_trials, train_loader, val_loader, device):
    """Random search over hyperparameters."""
    results = []
    
    for _ in range(n_trials):
        # Sample parameters
        lr = 10 ** random.uniform(-5, -1)
        batch_size = random.choice([16, 32, 64, 128])
        wd = 10 ** random.uniform(-6, -2)
        
        model = model_class().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        # Train and evaluate
        best_val_acc = 0.0
        for epoch in range(20):
            train_one_epoch(model, train_loader, optimizer)
            val_acc = evaluate(model, val_loader)
            best_val_acc = max(best_val_acc, val_acc)
        
        results.append({
            'params': {'lr': lr, 'batch_size': batch_size, 'weight_decay': wd},
            'val_acc': best_val_acc
        })
        print(f"Trial: lr={lr:.6f}, batch={batch_size}, wd={wd:.6f}, acc={best_val_acc:.4f}")
    
    return sorted(results, key=lambda x: x['val_acc'], reverse=True)
```

## Bayesian Optimization with Optuna

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Create model and optimizer
    model = create_model(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Train for fixed epochs
    best_val_acc = 0.0
    for epoch in range(15):
        train_one_epoch(model, train_loader, optimizer)
        val_acc = evaluate(model, val_loader)
        best_val_acc = max(best_val_acc, val_acc)
    
    return best_val_acc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best trial: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_trial.params}")
```

## Learning Rate Tuning Tips

```python
def find_good_lr(model, train_loader, device):
    """Use LR range test to find good learning rate."""
    lr_range_test(model, train_loader, device, start_lr=1e-7, end_lr=10, epochs=1)
    # Analyze the plot to find optimal LR
```

## Hyperparameter Interaction Effects

Some hyperparameters interact:

- **LR and Batch Size**: Larger batches can use larger LRs
- **LR and Weight Decay**: Higher WD often requires lower LR
- **Dropout and Data Augmentation**: Less dropout with strong augmentation

```python
# Common patterns
# Vision: lr=0.001, batch=32, wd=0.01, dropout=0.3
# Transformer: lr=1e-4, batch=32, wd=0.01, dropout=0.1
# Small data: lr=1e-3, batch=16, wd=1e-4, dropout=0.5
```

Systematic tuning with random search or Optuna is recommended for new tasks.