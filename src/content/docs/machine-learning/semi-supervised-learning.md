---
title: Semi-Supervised Learning
description: Master semi-supervised learning techniques that leverage large amounts of unlabeled data alongside small labeled sets — covering self-training and pseudo-labeling, consistency regularization, FixMatch, MixMatch, graph-based label propagation, and modern semi-supervised extensions for large-scale vision and NLP tasks.
---

Most real-world ML problems have abundant unlabeled data and scarce labeled data — labeling is expensive, slow, and requires domain expertise. A medical imaging model may have access to millions of X-rays but only a few thousand expert-annotated diagnoses. **Semi-supervised learning (SSL)** bridges this gap by learning from both labeled and unlabeled data, typically achieving performance close to fully supervised models while using a fraction of the labeled data.

## The Core Assumption

SSL methods exploit one or more assumptions about the structure of data:

- **Smoothness assumption**: if two points are close in input space, they likely have the same label — the decision boundary should pass through low-density regions
- **Cluster assumption**: data points in the same cluster tend to share a label
- **Manifold assumption**: high-dimensional data lies on a lower-dimensional manifold where distance is meaningful

## Self-Training and Pseudo-Labeling

The simplest SSL approach is **self-training**: train on labeled data, predict labels for unlabeled data, add high-confidence predictions as pseudo-labeled training data, and iterate.

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def self_training(
    model,
    labeled_loader,
    unlabeled_loader,
    n_rounds: int = 5,
    confidence_threshold: float = 0.95,
    device: str = "cuda",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for round_idx in range(n_rounds):
        # Train on labeled data
        model.train()
        for x_l, y_l in labeled_loader:
            x_l, y_l = x_l.to(device), y_l.to(device)
            loss = F.cross_entropy(model(x_l), y_l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Generate pseudo-labels for unlabeled data
        model.eval()
        pseudo_x, pseudo_y = [], []
        with torch.no_grad():
            for x_u, in unlabeled_loader:
                x_u = x_u.to(device)
                probs = F.softmax(model(x_u), dim=-1)
                conf, pred = probs.max(dim=-1)
                # Keep only high-confidence predictions
                mask = conf >= confidence_threshold
                pseudo_x.append(x_u[mask].cpu())
                pseudo_y.append(pred[mask].cpu())

        if pseudo_x:
            pseudo_dataset = TensorDataset(
                torch.cat(pseudo_x), torch.cat(pseudo_y)
            )
            labeled_loader = DataLoader(
                torch.utils.data.ConcatDataset([
                    labeled_loader.dataset, pseudo_dataset
                ]),
                batch_size=labeled_loader.batch_size, shuffle=True,
            )
        print(f"Round {round_idx+1}: {len(pseudo_y[0]) if pseudo_y else 0} pseudo-labels added")
```

Self-training works well when initial labeled accuracy is high but is prone to **confirmation bias**: early errors compound because wrong pseudo-labels become training data.

## Consistency Regularization

A key insight: a good classifier should produce the same prediction for an input regardless of small perturbations. **Consistency regularization** enforces this by minimizing the distance between model predictions on original and augmented versions of unlabeled inputs:

$$\mathcal{L}_{\mathrm{cons}} = \mathbb{E}_{x_u \in \mathcal{U}}\left[D\!\left(f_\theta(x_u),\, f_\theta(\mathcal{A}(x_u))\right)\right]$$

where $\mathcal{A}$ is a stochastic augmentation function and $D$ is KL divergence or MSE.

### Mean Teacher

Mean Teacher (Tarvainen & Valpola, 2017) uses two networks: a student updated by gradient descent and a teacher updated by an exponential moving average (EMA) of the student weights:

$$\theta'_t = \alpha \theta'_{t-1} + (1 - \alpha) \theta_t$$

The teacher provides a more stable target than the student for consistency loss:

```python
class MeanTeacher:
    def __init__(self, student, ema_decay: float = 0.999):
        self.student = student
        # Teacher is a copy with no gradient updates
        import copy
        self.teacher = copy.deepcopy(student)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)

    def consistency_loss(self, x_u, augment_fn):
        with torch.no_grad():
            teacher_probs = F.softmax(self.teacher(x_u), dim=-1)
        student_probs = F.softmax(self.student(augment_fn(x_u)), dim=-1)
        return F.mse_loss(student_probs, teacher_probs)
```

## FixMatch: Pseudo-Labels with Strong/Weak Augmentation

**FixMatch** (Sohn et al., 2020) combines pseudo-labeling with consistency regularization using an asymmetric augmentation strategy:

1. Apply **weak augmentation** (flip, crop) to get a stable pseudo-label
1. Apply **strong augmentation** (RandAugment, Cutout, CTAugment) to the same image
1. Only use pseudo-labels where the weakly-augmented prediction exceeds a confidence threshold $\tau$

$$\mathcal{L}_{\mathrm{FixMatch}} = \frac{1}{|\mathcal{B}_u|} \sum_{b=1}^{|\mathcal{B}_u|} \mathbf{1}[\max_c p_b^w \geq \tau] \cdot H(\hat{q}_b, p_b^s)$$

where $p_b^w = \text{softmax}(f(A_w(x_b)))$, $p_b^s = \text{softmax}(f(A_s(x_b)))$, $\hat{q}_b = \arg\max p_b^w$ is the one-hot pseudo-label, and $H$ is cross-entropy.

```python
from torchvision import transforms
import torch.nn.functional as F


class FixMatch:
    def __init__(self, model, threshold: float = 0.95, lambda_u: float = 1.0):
        self.model = model
        self.threshold = threshold
        self.lambda_u = lambda_u

    def train_step(self, x_labeled, y_labeled, x_unlabeled_weak, x_unlabeled_strong, optimizer):
        self.model.train()

        # Labeled loss
        logits_l = self.model(x_labeled)
        loss_l = F.cross_entropy(logits_l, y_labeled)

        # Pseudo-labels from weakly augmented unlabeled data (no gradient)
        with torch.no_grad():
            probs_weak = F.softmax(self.model(x_unlabeled_weak), dim=-1)
            confidence, pseudo_labels = probs_weak.max(dim=-1)
            mask = confidence >= self.threshold

        # Consistency loss on strongly augmented data
        logits_strong = self.model(x_unlabeled_strong)
        loss_u = (F.cross_entropy(logits_strong, pseudo_labels, reduction="none") * mask).mean()

        total_loss = loss_l + self.lambda_u * loss_u
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "loss_labeled": loss_l.item(),
            "loss_unlabeled": loss_u.item(),
            "mask_ratio": mask.float().mean().item(),
        }
```

FixMatch achieves 94.9% accuracy on CIFAR-10 with only 40 labels (4 per class) — dramatically outperforming fully supervised training on 40 labeled examples.

## MixMatch and ReMixMatch

**MixMatch** (Berthelot et al., 2019) applies multiple augmentations to unlabeled data, averages the predictions as a sharpened pseudo-label, then mixes labeled and unlabeled data with Mixup:

1. Augment each unlabeled image $K$ times: $\hat{x}^{(k)} = \mathcal{A}(x_u)$
1. Average predictions and sharpen: $\tilde{q} = \text{sharpen}\!\left(\frac{1}{K}\sum_k p^{(k)}\right)$
1. Mix labeled and pseudo-labeled data: $(x', y') = \mathrm{Mixup}(x_l, x_u, y_l, \tilde{q})$
1. Supervised loss on mixed labeled, consistency loss on mixed unlabeled

**ReMixMatch** (Berthelot et al., 2020) improves on MixMatch by adding **distribution alignment** (aligning the marginal prediction distribution on unlabeled data to match the labeled class distribution) and **augmentation anchoring** (using a strongly-augmented target for unlabeled data).

## Graph-Based Label Propagation

When data has a meaningful similarity structure, **label propagation** spreads labels along the graph of similar examples:

$$\mathbf{F} = (1 - \alpha)(\mathbf{I} - \alpha \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2})^{-1}\mathbf{Y}$$

where $\mathbf{W}$ is a weighted adjacency matrix (e.g., from $k$-NN on embedding space), $\mathbf{D}$ is the degree matrix, $\mathbf{Y}$ contains known labels (zero for unlabeled), $\alpha$ controls label propagation vs. clamping to known labels, and $\mathbf{F}$ gives soft label assignments.

```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import numpy as np

# -1 marks unlabeled examples
labels = np.array([0, 1, -1, -1, -1, 2, -1, -1])

lp = LabelSpreading(kernel="rbf", gamma=0.25, alpha=0.2, max_iter=30)
lp.fit(X_all, labels)
predicted = lp.predict(X_all)
```

Label propagation works well when the manifold assumption holds — the labeled and unlabeled data share the same low-dimensional structure.

## Semi-Supervised Learning for NLP

In NLP, the dominant approach is **pretrain then fine-tune**: pretrain a language model on massive unlabeled text (BERT, RoBERTa, T5), then fine-tune on a small labeled task dataset. This is semi-supervised learning at scale — the unlabeled pretraining corpus provides implicit SSL signal.

For specialized domains with scarce labels:

- **Self-training with RoBERTa**: generate pseudo-labels from a teacher model fine-tuned on available labeled data, train a student on pseudo-labeled + labeled data
- **UDA (Unsupervised Data Augmentation)**: apply TF-IDF word replacement, back-translation, or paraphrasing as augmentations for NLP consistency regularization

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def ssl_train_step(model, batch_labeled, batch_unlabeled, lambda_u=1.0):
    # Labeled cross-entropy
    x_l = tokenizer(**batch_labeled["text"], return_tensors="pt", truncation=True, padding=True)
    y_l = batch_labeled["labels"]
    loss_l = F.cross_entropy(model(**x_l).logits, y_l)

    # Consistency between original and augmented unlabeled text
    x_u_orig = tokenizer(**batch_unlabeled["text"], return_tensors="pt", truncation=True, padding=True)
    x_u_aug = tokenizer(**batch_unlabeled["augmented"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        probs_orig = F.softmax(model(**x_u_orig).logits, dim=-1)
    probs_aug = F.softmax(model(**x_u_aug).logits, dim=-1)
    loss_u = F.kl_div(probs_aug.log(), probs_orig, reduction="batchmean")

    return loss_l + lambda_u * loss_u
```

## Summary

Semi-supervised learning extracts value from the vast amounts of unlabeled data that accompany almost every real-world ML problem:

- **Self-training** is simple and effective when initial supervised accuracy is high; confidence thresholding reduces confirmation bias
- **Consistency regularization** (Mean Teacher, UDA) leverages the assumption that predictions should be invariant to realistic augmentations
- **FixMatch** combines weak-augmentation pseudo-labels with strong-augmentation consistency, achieving near-supervised performance with a tiny fraction of labels
- **MixMatch and ReMixMatch** add distribution alignment and Mixup to produce smoother decision boundaries
- **Label propagation** on similarity graphs is effective when data lies on a meaningful manifold
- For NLP, **pretrain-then-fine-tune** is the dominant SSL paradigm — large-scale language model pretraining implicitly provides SSL signal at scale
