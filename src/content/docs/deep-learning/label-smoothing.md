---
title: Label Smoothing - Softening Targets to Reduce Overconfidence
description: Learn how label smoothing replaces hard one-hot targets with softened distributions to regularize classifiers and improve calibration and generalization.
---

Label smoothing is a regularization technique for classification. Instead of training against hard one-hot targets, it moves a small amount of probability mass from the true class to the other classes.

## The Idea

With K classes and smoothing factor epsilon, the target for the correct class becomes `1 - epsilon`, and each other class receives `epsilon / (K - 1)` (some formulations spread it over all K classes):

```text
hard target:     [0, 0, 1, 0]
smoothed (0.1):  [0.033, 0.033, 0.9, 0.033]
```

## Why It Helps

With hard targets, cross-entropy keeps pushing the correct logit toward infinity relative to others, encouraging overconfident predictions and large logit magnitudes. Smoothed targets give the loss a finite optimum, which:

- Reduces overfitting, especially on noisy labels
- Often improves calibration of predicted probabilities
- Encourages tight, well-separated class clusters in representation space

## Implementation

```python
import torch.nn as nn

# Built into PyTorch's cross-entropy loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = criterion(logits, targets)
```

Typical values range from 0.05 to 0.1.

## Where It Is Used

It is a standard ingredient in image classifiers trained on ImageNet, in the original Transformer for machine translation, and in many training recipes for vision transformers.

## Caveats

- It can hurt knowledge distillation: a smoothed teacher produces less informative soft targets, because information about class similarity is erased.
- Smoothed models may be less suitable when you need the raw confidence to carry fine-grained information.
- Too large an epsilon underfits by making the targets overly uninformative.
