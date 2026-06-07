---
title: "Model Ensembling in Deep Learning"
description: "Combining multiple models to improve performance — averaging, stacking, and diversity strategies."
date: "2026-06-06"
tags: ["deep-learning", "ensembling", "model-composition"]
---

Ensembling combines predictions from multiple models to reduce variance and improve generalization.

## Model Averaging

The simplest form of ensembling:

```python
class ModelAveraging:
    def __init__(self, models, weights=None):
        self.models = models
        self.num_models = len(models)
        self.weights = weights or [1.0 / self.num_models] * self.num_models
    
    def predict(self, x):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred * weight)
        
        return sum(predictions)
    
    def predict_proba(self, x):
        # For classification: average probabilities
        probas = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                proba = torch.softmax(model(x), dim=1)
                probas.append(proba)
        
        return torch.stack(probas).mean(dim=0)


# Usage
ensemble = ModelAveraging([model1, model2, model3], weights=[0.4, 0.3, 0.3])
predictions = ensemble.predict(test_inputs)
```

## Snapshot Ensembling

Train a single model and save weights at different local minima:

```python
class SnapshotEnsembling:
    def __init__(self, model, base_lr=0.1, snapshot_every=10, num_snapshots=5):
        self.model = model
        self.base_lr = base_lr
        self.snapshot_every = snapshot_every
        self.num_snapshots = num_snapshots
        self.snapshots = []
        self.epoch = 0
    
    def cosine_annealing(self, epoch, T_max, lr_min=0.001):
        return lr_min + 0.5 * (self.base_lr - lr_min) * (
            1 + math.cos(math.pi * epoch / T_max)
        )
    
    def train_step(self, train_loader):
        self.epoch += 1
        
        # Cosine annealing learning rate
        lr = self.cosine_annealing(self.epoch, self.snapshot_every * self.num_snapshots)
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        # Training code here...
        # loss = ...
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        
        # Save snapshot
        if self.epoch % self.snapshot_every == 0:
            self.snapshots.append(self.model.state_dict().copy())
            print(f"Snapshot saved at epoch {self.epoch}")
    
    def predict(self, x):
        predictions = []
        for state_dict in self.snapshots:
            self.model.load_state_dict(state_dict)
            self.model.eval()
            with torch.no_grad():
                predictions.append(torch.softmax(self.model(x), dim=1))
        
        return torch.stack(predictions).mean(dim=0)
```

## Diversity Strategies for Ensembling

```python
class DiverseEnsemble:
    """Build ensemble with diverse models."""
    def __init__(self):
        self.models = []
        self.diversity_scores = []
    
    def add_model(self, model, name='default'):
        """Add a model to the ensemble."""
        self.models.append((name, model))
    
    def compute_diversity(self, model_a, model_b, dataloader):
        """Compute diversity between two models."""
        predictions_a = []
        predictions_b = []
        
        for inputs, _ in dataloader:
            with torch.no_grad():
                predictions_a.append(torch.softmax(model_a(inputs), dim=1))
                predictions_b.append(torch.softmax(model_b(inputs), dim=1))
        
        predictions_a = torch.cat(predictions_a)
        predictions_b = torch.cat(predifications_b)
        
        # Disagreement diversity
        disagreement = (predictions_a.argmax(1) != predictions_b.argmax(1)).float().mean()
        
        # Entropy diversity
        avg_proba = (predictions_a + predictions_b) / 2
        entropy = -(avg_proba * torch.log(avg_proba + 1e-10)).sum(dim=1).mean()
        
        return disagreement.item(), entropy.item()
```

## Test-Time Augmentation Ensembling

```python
def test_time_augmentation(model, image, num_augmentations=8):
    """Apply TTA and ensemble predictions."""
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ]
    
    predictions = []
    
    for _ in range(num_augmentations):
        aug_image = image.clone()
        if random.random() > 0.5:
            aug_image = transforms.RandomHorizontalFlip(p=1.0)(aug_image)
        if random.random() > 0.5:
            aug_image = transforms.ColorJitter(brightness=0.1)(aug_image)
        
        model.eval()
        with torch.no_grad():
            pred = torch.softmax(model(aug_image), dim=1)
            predictions.append(pred)
    
    return torch.stack(predictions).mean(dim=0)
```

## Weighted Ensemble by Validation

```python
class WeightedEnsemble:
    def __init__(self, models, val_loader, device):
        self.models = models
        self.weights = self._compute_weights(val_loader, device)
    
    def _compute_weights(self, val_loader, device):
        """Compute ensemble weights based on validation performance."""
        losses = []
        
        for model in self.models:
            model.eval()
            total_loss = 0.0
            count = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    count += inputs.size(0)
            
            losses.append(total_loss / count)
        
        # Weight inversely proportional to loss
        losses = torch.tensor(losses)
        weights = (losses.max() / (losses + 1e-8)).float()
        weights = weights / weights.sum()
        
        return weights.numpy().tolist()
    
    def predict(self, x):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = torch.softmax(model(x), dim=1)
                predictions.append(pred * weight)
        
        return sum(predictions)
```

Ensembling typically improves 1-3% accuracy over single models but requires more compute.