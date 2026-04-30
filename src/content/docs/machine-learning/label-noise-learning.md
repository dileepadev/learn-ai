---
title: Learning with Noisy Labels
description: Understand how machine learning models can be trained robustly when training data contains incorrect or unreliable labels — covering noise models, noise-robust loss functions, sample selection methods, label cleaning techniques, and practical strategies for real-world noisy datasets.
---

**Label noise** is ubiquitous in real-world machine learning. Crowdsourced annotations disagree, automated labeling pipelines make errors, domain experts misclassify edge cases, and data collection processes introduce systematic biases. When a model is trained on data with noisy labels, it risks memorizing the errors rather than learning the true underlying pattern — a phenomenon well-documented in the deep learning literature where networks eventually fit even random labels given enough capacity and training time.

The field of **learning with noisy labels** develops methods to train accurate models despite this noise, without requiring fully clean relabeled data.

## Types of Label Noise

### Uniform (Symmetric) Noise

Each label is independently flipped to any other class with probability $\epsilon$:

$$P(\tilde{y} = j \mid y = i) = \begin{cases} 1 - \epsilon & \text{if } j = i \\ \epsilon / (C-1) & \text{if } j \neq i \end{cases}$$

where $C$ is the number of classes and $\epsilon$ is the noise rate. This is the most theoretically tractable form — it is symmetric across classes and independent of the input.

### Class-Dependent (Asymmetric) Noise

Labels are flipped with class-specific probabilities. This models real annotation errors more faithfully — a dog is more likely to be mislabeled as a cat than as an airplane:

$$P(\tilde{y} = j \mid y = i) = T_{ij}$$

where $T$ is a $C \times C$ **noise transition matrix** with rows summing to 1. Estimating $T$ from data is a key challenge.

### Instance-Dependent Noise

The most realistic model: noise depends on both the class and the input features. Hard, ambiguous examples near decision boundaries are more likely to be mislabeled than easy, prototypical examples:

$$P(\tilde{y} = j \mid y = i, \mathbf{x})$$

This is the hardest case to model theoretically but the most common in practice.

## Noise-Robust Loss Functions

### Symmetric Cross-Entropy (SCE)

Standard cross-entropy is asymmetric — predictions confident in wrong labels produce very large gradients that strongly pull the model toward noisy labels. **Symmetric Cross-Entropy** (Wang et al., 2019) adds a reverse cross-entropy term:

$$\mathcal{L}_{\text{SCE}} = \alpha \cdot \mathcal{L}_{\text{CE}}(p, \tilde{y}) + \beta \cdot \mathcal{L}_{\text{CE}}(\tilde{y}, p)$$

where the second term $\mathcal{L}_{\text{CE}}(\tilde{y}, p) = -\sum_c \tilde{y}_c \log p_c$ is the reverse CE (model prediction as "truth", label as "prediction"). The reverse term is bounded and noise-tolerant for one-hot labels.

```python
import torch
import torch.nn.functional as F

def symmetric_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                             alpha: float = 0.1, beta: float = 1.0,
                             num_classes: int = 10) -> torch.Tensor:
    """
    Symmetric Cross-Entropy loss for noise-robust training.
    
    Args:
        logits: (N, C) raw model outputs
        targets: (N,) integer class labels (may be noisy)
        alpha: weight for standard CE (forward)
        beta: weight for reverse CE (backward)
    """
    probs = F.softmax(logits, dim=1)
    
    # Forward CE: standard cross-entropy
    ce_loss = F.cross_entropy(logits, targets)
    
    # Reverse CE: -sum(p * log(y_hat)) — treats labels as distribution
    # Clip to avoid log(0) for one-hot labels with value 0
    targets_one_hot = F.one_hot(targets, num_classes).float()
    A = torch.clamp(targets_one_hot, min=1e-6)
    rce_loss = -torch.mean(torch.sum(probs * torch.log(A), dim=1))
    
    return alpha * ce_loss + beta * rce_loss
```

### Generalized Cross-Entropy (GCE)

GCE (Zhang & Sabuncu, 2018) interpolates between mean absolute error (MAE, noise-robust but slow to converge) and cross-entropy (fast but noise-sensitive) via a parameter $q \in (0, 1]$:

$$\mathcal{L}_{\text{GCE}} = \frac{1 - p_{\tilde{y}}^q}{q}$$

where $p_{\tilde{y}}$ is the predicted probability for the (potentially noisy) label. As $q \to 0$, this approaches CE; as $q \to 1$, it approaches MAE.

```python
def generalized_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                               q: float = 0.7) -> torch.Tensor:
    """
    GCE loss: noise-robust interpolation between CE and MAE.
    q=0.7 is a common default (recommended by original paper).
    """
    probs = F.softmax(logits, dim=1)
    # Gather predicted prob for the (noisy) target class
    p_y = probs[torch.arange(len(targets)), targets]
    loss = (1.0 - p_y ** q) / q
    return loss.mean()
```

## Sample Selection Methods

Rather than modifying the loss, **sample selection** identifies and down-weights (or discards) likely-noisy samples during training.

### Small-Loss Trick

A clean observation: for most tasks, **clean samples have smaller training loss than noisy ones**, especially early in training before the network memorizes noise. This motivates selecting the small-loss subset as a proxy for clean samples.

```python
def small_loss_selection(losses: torch.Tensor, noise_rate: float = 0.2) -> torch.Tensor:
    """
    Select the (1 - noise_rate) fraction of samples with smallest loss.
    Returns a boolean mask of selected (presumed clean) samples.
    """
    n = len(losses)
    threshold_idx = int(n * (1 - noise_rate))
    threshold = torch.sort(losses)[0][threshold_idx]
    return losses <= threshold
```

### Co-training / Co-teaching

**Co-teaching** (Han et al., 2018) trains two networks simultaneously. Each network selects its small-loss samples to teach the *other* network — the disagreement between networks helps filter noise that one network might have already memorized:

```python
class CoTeaching:
    """
    Co-teaching: two networks mutually select clean samples for each other.
    """
    def __init__(self, model1, model2, noise_rate: float = 0.2):
        self.model1 = model1
        self.model2 = model2
        self.noise_rate = noise_rate
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor,
                   optimizer1, optimizer2) -> dict:
        # Forward pass on both models
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        
        losses1 = F.cross_entropy(logits1, y, reduction="none")
        losses2 = F.cross_entropy(logits2, y, reduction="none")
        
        # Each model selects small-loss samples for the OTHER model
        clean_mask1 = small_loss_selection(losses1, self.noise_rate)  # model1 selects for model2
        clean_mask2 = small_loss_selection(losses2, self.noise_rate)  # model2 selects for model1
        
        # Update model1 on model2's selected samples
        loss1 = F.cross_entropy(logits1[clean_mask2], y[clean_mask2])
        optimizer1.zero_grad(); loss1.backward(); optimizer1.step()
        
        # Update model2 on model1's selected samples
        loss2 = F.cross_entropy(logits2[clean_mask1], y[clean_mask1])
        optimizer2.zero_grad(); loss2.backward(); optimizer2.step()
        
        return {"loss1": loss1.item(), "loss2": loss2.item(),
                "clean_fraction": clean_mask1.float().mean().item()}
```

## Label Correction Approaches

### Confident Learning

**Confident Learning** (Northcutt et al., 2021 — also the basis of the `cleanlab` library) estimates the noise transition matrix from the model's predicted probabilities and uses it to identify likely mislabeled examples:

1. Train an initial model with cross-validation to get out-of-sample predicted probabilities for every training example.
2. Estimate the joint distribution $P(\tilde{y}, y)$ — how often is class $i$ mislabeled as class $j$?
3. Flag examples as likely mislabeled when their predicted class differs from their given label with high confidence.

```python
# cleanlab makes confident learning very practical
# pip install cleanlab

from cleanlab.filter import find_label_issues
import numpy as np

# pred_probs: (N, C) out-of-sample predicted probabilities (from cross-val)
# labels: (N,) given (potentially noisy) integer labels
def find_mislabeled_examples(pred_probs: np.ndarray, labels: np.ndarray):
    """
    Returns indices of likely mislabeled examples using confident learning.
    """
    label_issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence"  # rank by how wrong the label looks
    )
    return label_issues

# Usage:
# issues = find_mislabeled_examples(pred_probs, train_labels)
# clean_mask = np.ones(len(train_labels), dtype=bool)
# clean_mask[issues] = False
# retrain on clean_mask subset
```

## Mixup for Noise Robustness

**Mixup** (Zhang et al., 2018) trains on convex combinations of training examples and labels:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

By training on soft, interpolated labels, the model is less able to memorize any individual noisy label — it must learn smooth decision boundaries consistent with the label *distribution*, not individual annotations.

```python
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Return mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

## Noise Rate Estimation

Many methods require knowing (or estimating) the noise rate $\epsilon$. Common approaches:

- **Anchor points**: Find examples the classifier is near-certain about and use them to estimate the noise transition matrix.
- **Cross-validation consistency**: Examples that receive inconsistently predicted labels across CV folds are more likely noisy.
- **Loss curve inflection**: The training loss curve for noisy data shows a characteristic two-phase behavior — a fast initial drop (clean examples) followed by a slower continued decrease (noisy memorization). The transition point estimates noise rate.

## Practical Guidelines

| Noise level | Recommended approach |
|---|---|
| < 10% | Standard training; noise has minor impact |
| 10–30% | GCE or SCE loss; Mixup augmentation |
| 30–50% | Co-teaching or DivideMix; confident learning for cleaning |
| > 50% | Label cleaning is critical before training; semi-supervised methods |

**When to clean vs. train robustly**: If you have budget for manual review, confident learning to identify and fix the top-k suspect labels is usually more effective than noise-robust training alone. For very large datasets where review is impractical, noise-robust loss functions and sample selection are the practical choice.
