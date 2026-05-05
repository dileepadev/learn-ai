---
title: Multi-Label Learning
description: Learn multi-label classification — where each instance can simultaneously belong to multiple classes. Explore problem formulation, evaluation metrics (Hamming loss, subset accuracy, ranking loss), loss functions, label correlation methods (Classifier Chains, label attention), deep learning approaches, and extreme multi-label classification (XMC) for millions of labels.
---

**Multi-label learning** is the general setting where each instance can be assigned zero, one, or many class labels simultaneously. Unlike multi-class classification (one label per instance) or binary classification (yes/no), multi-label problems reflect the natural complexity of many real-world domains: a news article can be about economics, politics, and technology simultaneously; a medical image may show multiple conditions; a song can belong to rock, pop, and alternative simultaneously.

## Problem Formulation

Given an input $x \in \mathcal{X}$ and a label space $\mathcal{Y} = \{y_1, y_2, \ldots, y_L\}$ with $L$ possible labels, the multi-label problem learns:

$$f: \mathcal{X} \rightarrow \{0, 1\}^L$$

Each prediction is a binary vector $\hat{y} \in \{0, 1\}^L$ where $\hat{y}_j = 1$ indicates label $j$ is predicted for instance $x$. This is distinct from multi-class classification (where the output is a one-hot vector or a single integer index) and from regression (where outputs are continuous).

**Label density**: $\text{LD} = \frac{1}{n} \sum_{i=1}^n \frac{|\text{labels}(x_i)|}{L}$ measures the average fraction of labels that are active. Typical values range from 0.01 (sparse, like document tagging) to 0.1 (denser, like image annotation).

## Evaluation Metrics

Multi-label evaluation is more complex than single-label, because a prediction can be partially correct. Key metrics:

**Hamming Loss** — fraction of incorrectly predicted labels (lower is better):

$$\text{HL} = \frac{1}{nL} \sum_{i=1}^n \sum_{j=1}^L \mathbb{1}[\hat{y}_{ij} \neq y_{ij}]$$

**Subset Accuracy (Exact Match Ratio)** — fraction of instances where the entire predicted label set exactly matches the true label set (strictest metric):

$$\text{SA} = \frac{1}{n} \sum_{i=1}^n \mathbb{1}[\hat{y}_i = y_i]$$

**Micro-F1** — compute TP, FP, FN globally across all labels and instances, then compute F1. Dominated by frequent labels.

**Macro-F1** — compute F1 per label, then average. Treats all labels equally regardless of frequency.

**Ranking Loss** — measures how often a relevant label is ranked lower than an irrelevant one:

$$\text{RL} = \frac{1}{n} \sum_{i=1}^n \frac{|\{(j,k): \hat{r}(j) > \hat{r}(k), y_{ij}=1, y_{ik}=0\}|}{|\mathcal{Y}_i^+ \cdot \mathcal{Y}_i^-|}$$

```python
import numpy as np
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    average_precision_score, label_ranking_loss
)

def compute_multilabel_metrics(
    y_true: np.ndarray,      # (n_samples, n_labels) binary ground truth
    y_pred: np.ndarray,      # (n_samples, n_labels) binary predictions
    y_scores: np.ndarray     # (n_samples, n_labels) prediction scores/probabilities
) -> dict[str, float]:
    """
    Comprehensive multi-label evaluation metrics.
    
    y_true: binary matrix — 1 if label j is active for sample i
    y_pred: thresholded binary predictions (threshold at 0.5 typically)
    y_scores: raw prediction scores for ranking-based metrics
    """
    metrics = {}
    
    # Label-based metrics (operate on binary predictions after thresholding)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
    metrics["subset_accuracy"] = accuracy_score(y_true, y_pred)
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["samples_f1"] = f1_score(y_true, y_pred, average="samples", zero_division=0)
    
    # Ranking-based metrics (use raw scores, not thresholded predictions)
    # Mean Average Precision: measures ranking quality per label
    try:
        metrics["mean_average_precision"] = average_precision_score(
            y_true, y_scores, average="macro"
        )
    except ValueError:
        metrics["mean_average_precision"] = float("nan")
    
    # Ranking loss: fraction of label pairs misordered
    try:
        metrics["ranking_loss"] = label_ranking_loss(y_true, y_scores)
    except ValueError:
        metrics["ranking_loss"] = float("nan")
    
    # Coverage: how many ranks needed to cover all true labels
    coverage = 0.0
    for i in range(len(y_true)):
        true_labels = np.where(y_true[i] == 1)[0]
        if len(true_labels) == 0:
            continue
        sorted_scores = np.argsort(-y_scores[i])   # descending
        max_rank = max(np.where(sorted_scores == l)[0][0] for l in true_labels)
        coverage += max_rank + 1
    metrics["coverage"] = coverage / len(y_true)
    
    return metrics
```

## Loss Functions

### Binary Cross-Entropy (BCEWithLogitsLoss)

The simplest and most widely used: treat each label independently as a binary classification problem:

$$\mathcal{L}_\text{BCE} = -\frac{1}{nL} \sum_{i,j} \left[ y_{ij} \log \sigma(z_{ij}) + (1-y_{ij}) \log (1-\sigma(z_{ij})) \right]$$

```python
import torch
import torch.nn as nn

class MultiLabelClassifier(nn.Module):
    """
    Multi-label classifier with configurable backbone and loss function.
    The output layer uses L sigmoid activations (one per label) rather
    than a single softmax — labels are predicted independently.
    """
    
    def __init__(self, backbone: nn.Module, backbone_dim: int, n_labels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone_dim, n_labels)
        
        # No activation here — sigmoid is applied in loss (numerically stable)
        # or explicitly during inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        return self.classifier(features)   # raw logits, shape (B, n_labels)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary label vectors with per-label threshold."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification (Ridnik et al., 2021).
    
    Addresses the severe positive-negative imbalance in multi-label tasks:
    in a dataset with L=80 labels (MS-COCO), the average image has only 3-4
    positive labels and ~76 negative labels. BCELoss treats both equally,
    leading to a model that predicts "absent" for everything.
    
    ASL uses different focusing parameters for positive (γ+) and negative (γ-)
    examples, and probability shifting (m) to hard-discard easy negatives:
    
    For positives:  L+ = (1-p)^γ+ × -log(p)
    For negatives:  L- = (p_m)^γ- × -log(1-p_m)   where p_m = max(p-m, 0)
    
    γ+ < γ- enforces asymmetry: hard positives are upweighted more than hard negatives.
    m > 0 (probability margin) clips easy negative probabilities to 0, ignoring them.
    """
    
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Probability shifting: clip easy negatives
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1)
        else:
            probs_neg = probs
        
        # Focal weight
        probs_pos = probs
        pt_pos = targets * probs_pos
        pt_neg = (1 - targets) * probs_neg
        
        loss_pos = -targets * (1 - pt_pos) ** self.gamma_pos * torch.log(probs_pos + self.eps)
        loss_neg = -(1 - targets) * pt_neg ** self.gamma_neg * torch.log(1 - probs_neg + self.eps)
        
        return (loss_pos + loss_neg).mean()
```

## Label Correlation Methods

Labels in real data are correlated: if an image contains "cat," it's more likely to contain "indoor" than "airplane." Ignoring these correlations (treating labels independently) leaves signal on the table.

### Classifier Chains

**Classifier Chains** (Read et al., 2011) exploits label correlations by training $L$ classifiers sequentially, each receiving the previous labels' predictions as input features:

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import numpy as np

class ClassifierChain(BaseEstimator, ClassifierMixin):
    """
    Classifier Chain (CC) for multi-label classification.
    
    Training: for label j, train a classifier on [X, y_1, ..., y_{j-1}]
    — using true labels from training data (teacher forcing).
    
    Inference: predict labels left-to-right, each using previously
    predicted labels as input features (autoregressive).
    
    The label ordering matters and affects performance.
    Ensemble of Classifier Chains (ECC) uses random orderings and
    majority voting to reduce sensitivity to order choice.
    
    Advantage: captures label correlations without modeling P(Y) explicitly.
    Disadvantage: error propagation — early prediction errors compound.
    """
    
    def __init__(self, base_classifier=None, label_order: list[int] = None,
                 random_state: int = 42):
        self.base_classifier = base_classifier or LogisticRegression(max_iter=1000)
        self.label_order = label_order
        self.random_state = random_state
        self.classifiers_ = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ClassifierChain":
        """
        X: (n_samples, n_features)
        Y: (n_samples, n_labels) binary label matrix
        """
        n_labels = Y.shape[1]
        self.label_order_ = self.label_order or list(range(n_labels))
        self.classifiers_ = []
        
        X_chain = X.copy()
        
        for i, label_idx in enumerate(self.label_order_):
            clf = type(self.base_classifier)(**self.base_classifier.get_params())
            clf.fit(X_chain, Y[:, label_idx])
            self.classifiers_.append(clf)
            
            # Append true label as feature for next classifier (teacher forcing)
            X_chain = np.hstack([X_chain, Y[:, label_idx:label_idx+1]])
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_labels = len(self.label_order_)
        predictions = np.zeros((n_samples, n_labels), dtype=int)
        
        X_chain = X.copy()
        
        for i, (label_idx, clf) in enumerate(zip(self.label_order_, self.classifiers_)):
            pred = clf.predict(X_chain)
            predictions[:, label_idx] = pred
            X_chain = np.hstack([X_chain, pred.reshape(-1, 1)])
        
        return predictions
```

## Extreme Multi-Label Classification

**Extreme Multi-Label Classification (XMC)** is the setting where the label space is enormous — tens of thousands to millions of labels. Examples: Amazon product tagging (70K+ categories), biomedical text labeling (ICD-10 has 69,823 codes), web-scale ad keyword matching.

XMC is too large for dense label matrices or pairwise label correlation models. Key approaches:

- **Parabel / PECOS**: hierarchically partition the label space using a balanced label tree, then train one classifier per tree node. Inference traverses the tree in $\mathcal{O}(\log L)$ rather than $\mathcal{O}(L)$.
- **AttentionXML**: combines hierarchical label trees with a label attention mechanism that selects relevant candidate labels per input using a two-stage retrieve-then-rank approach.
- **XR-Transformer**: uses BERT-style encoders with label clustering to handle 100K+ label spaces efficiently.

## Benchmark Datasets

| Dataset | Samples | Labels | Domain |
| --- | --- | --- | --- |
| MS-COCO | 122,218 | 80 | Image recognition |
| NUS-WIDE | 269,648 | 81 | Web image tagging |
| RCV1 | 804,414 | 103 | News categorization |
| EUR-Lex | 19,348 | 3,956 | Legal documents (EU law) |
| AmazonCat-13K | 1.18M | 13,330 | Product tagging |
| Wiki10-31K | 14,146 | 30,938 | Wikipedia topics |
| Amazon-670K | 490,449 | 670,091 | XMC product categories |

Multi-label learning bridges the gap between idealized benchmark settings and real-world annotation complexity. Its core insight — that prediction targets are often multi-faceted, overlapping, and correlated — extends naturally to structured prediction, multi-task learning, and generative model conditioning.
