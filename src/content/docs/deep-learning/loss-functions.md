---
title: "Loss Functions for Deep Learning"
description: "Understanding cross-entropy, focal loss, label smoothing, and custom loss functions."
date: "2026-06-06"
tags: ["deep-learning", "loss-functions", "training"]
---

Loss functions measure how well the model predictions match the targets. Choosing the right loss is crucial for training success.

## Cross-Entropy Loss

Standard for classification:

```python
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets):
        # inputs: (batch, num_classes) logits
        # targets: (batch,) class indices
        
        log_softmax = F.log_softmax(inputs, dim=1)
        nll = -log_softmax[range(len(targets)), targets]
        return nll.mean()


# PyTorch built-in
criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax + NLLLoss
```

## Binary Cross-Entropy

For binary or multi-label classification:

```python
# For binary classification
bce_loss = nn.BCEWithLogitsLoss()  # Sigmoid + BCE

# Manual BCE
def binary_cross_entropy(inputs, targets):
    # inputs: (batch,) logits
    # targets: (batch,) 0 or 1
    return -(targets * torch.log_sigmoid(inputs) + 
             (1 - targets) * torch.log(1 - torch.sigmoid(inputs))).mean()
```

## Label Smoothing

Regularization that prevents overconfident predictions:

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        # inputs: (batch, num_classes) logits
        # targets: (batch,) class indices
        
        num_classes = inputs.size(1)
        
        # One-hot encode targets
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Smooth labels
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        # Compute loss
        log_softmax = F.log_softmax(inputs, dim=1)
        return -(one_hot * log_softmax).sum(dim=1).mean()


# Usually use 0.1 smoothing for vision, 0.05-0.1 for transformers
```

## Focal Loss

For class-imbalanced datasets:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha >= 0:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_weight * focal_weight
        
        loss = ce_loss * focal_weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# gamma=2, alpha=1 works well for object detection
```

## Dice Loss

For segmentation and class-imbalanced tasks:

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # inputs: (batch, 1, H, W) probabilities
        # targets: (batch, 1, H, W) binary
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


# Often combined with BCE for segmentation
def combined_loss(inputs, targets, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(inputs, targets)
    dice = DiceLoss()(torch.sigmoid(inputs), targets)
    return bce_weight * bce + (1 - bce_weight) * dice
```

## Triplet Loss

For metric learning and contrastive learning:

```python
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()
```

## Loss Selection Guide

| Task | Loss Function |
| --- | --- |
| Image classification | Cross-entropy + label smoothing |
| Object detection | Focal loss |
| Semantic segmentation | Dice + BCE |
| Binary classification | BCE with logits |
| Metric learning | Triplet loss |
| Imbalanced data | Focal loss or weighted cross-entropy |

Choose loss functions based on your specific task and data characteristics.