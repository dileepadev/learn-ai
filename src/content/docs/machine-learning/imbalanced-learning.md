---
title: Handling Class Imbalance in Machine Learning
description: Explore practical techniques for training effective machine learning models when one class vastly outnumbers others — from resampling and cost-sensitive learning to ensemble methods and evaluation strategies.
---

**Class imbalance** occurs when one or more classes in a classification dataset have significantly fewer samples than others. It is one of the most common practical obstacles in real-world machine learning — and one that standard algorithms are not designed to handle out of the box.

## Why Class Imbalance Is a Problem

Most classifiers minimize overall error. When one class represents 99% of the data, a model that always predicts the majority class achieves **99% accuracy** — yet is completely useless for detecting the minority class.

Common imbalanced scenarios:

- **Fraud detection** — Fraudulent transactions are 0.1–1% of all transactions.
- **Medical diagnosis** — Rare disease cases are far outnumbered by healthy patients.
- **Anomaly detection** — Defective items in manufacturing are rare.
- **Churn prediction** — Churned customers are a minority.

## Measuring the Right Things

Before addressing class imbalance, use metrics that reflect minority class performance:

| Metric | Formula | Use When |
|---|---|---|
| **Precision** | TP / (TP + FP) | Cost of false positives is high |
| **Recall (Sensitivity)** | TP / (TP + FN) | Cost of false negatives is high |
| **F1 Score** | 2 × (P × R) / (P + R) | Balance precision and recall |
| **AUC-PR** | Area under precision-recall curve | Imbalanced binary classification |
| **G-Mean** | √(Sensitivity × Specificity) | Multi-class imbalance |
| **MCC (Matthews)** | Balanced metric for all four quadrants | General-purpose imbalanced eval |

**Avoid accuracy** as the primary metric with imbalanced data.

## Resampling Strategies

### Oversampling the Minority Class

#### Random Oversampling

Duplicate random minority class samples until classes are balanced. Simple but risks **overfitting** to specific minority examples.

#### SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates **synthetic samples** rather than duplicates:

1. For each minority sample, find its $k$-nearest neighbors (also minority).
2. Randomly select one neighbor.
3. Generate a synthetic point on the line segment between them.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**SMOTE variants:**

- **ADASYN** — Generates more samples in harder-to-learn regions.
- **BorderlineSMOTE** — Focuses synthesis near the decision boundary.
- **SVMSMOTE** — Uses an SVM to identify borderline samples for synthesis.

### Undersampling the Majority Class

Remove samples from the majority class to balance the dataset. Risks losing useful information.

- **Random undersampling** — Randomly remove majority samples.
- **Tomek Links** — Remove majority samples that form close pairs with minority samples (cleanup near decision boundary).
- **Edited Nearest Neighbors (ENN)** — Remove majority samples misclassified by their neighbors.
- **NearMiss** — Select majority samples closest to minority samples (version-dependent strategy).

### Combined Approaches

- **SMOTETomek** — Apply SMOTE then clean with Tomek Links.
- **SMOTEENN** — Apply SMOTE then clean with ENN.

These balanced approaches often outperform either resampling strategy alone.

## Cost-Sensitive Learning

Rather than changing the data, modify the **loss function** to penalize minority class errors more heavily.

### Class Weights

Most scikit-learn classifiers and neural network frameworks support `class_weight`:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')

# Or manually specify weights:
model = LogisticRegression(class_weight={0: 1, 1: 10})
```

In neural networks with PyTorch:

```python
import torch
import torch.nn as nn

# Weight for minority class is 10×
pos_weight = torch.tensor([10.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Focal Loss

Originally proposed for object detection (RetinaNet), **focal loss** down-weights easy examples and focuses training on hard, misclassified ones:

$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

When $\gamma = 0$, it reduces to cross-entropy. Higher $\gamma$ focuses more on hard examples. Particularly effective for extreme imbalance.

## Algorithm-Level Approaches

### Threshold Moving

By default, classifiers use a 0.5 decision threshold. Moving the threshold toward 0 increases recall at the cost of precision (and vice versa). Use the precision-recall curve to find the optimal operating point for your use case.

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
# Choose threshold that maximizes F1 or meets your recall requirement
```

### Ensemble Methods for Imbalance

- **BalancedBaggingClassifier** — Train each base estimator on a balanced bootstrap sample.
- **EasyEnsemble** — Multiple balanced subsets, each with a trained classifier; aggregate by voting.
- **RUSBoost** — AdaBoost with random undersampling at each boosting step.

## Deep Learning with Imbalanced Data

- **Class-weighted loss** — Easiest first step for any neural network.
- **Oversampling in DataLoader** — Use a `WeightedRandomSampler` to oversample minority batches during training.
- **Data augmentation** — For images/text, augment minority class samples with synthetic variants (flips, rotations, back-translation, paraphrase).
- **Two-stage training** — Pre-train on imbalanced data, fine-tune on balanced subset.

## Choosing the Right Strategy

| Imbalance Ratio | Recommended Approach |
|---|---|
| Mild (1:3 to 1:10) | Class weights or threshold tuning |
| Moderate (1:10 to 1:100) | SMOTE + class weights |
| Severe (1:100 to 1:1000) | SMOTE + cost-sensitive loss + ensemble methods |
| Extreme (>1:1000) | Anomaly detection framing; focal loss; two-stage training |

## Practical Pitfalls

- **Resample only training data** — Never apply oversampling or undersampling to validation or test sets. Doing so leads to optimistic evaluation.
- **Cross-validation with resampling** — Apply resampling *inside* each fold, not on the full dataset before splitting. Use `imbalanced-learn` Pipeline to enforce this.
- **SMOTE can introduce noise** — Synthetic samples in overlapping regions between classes can hurt model performance. Validate empirically.
- **No universal winner** — Always compare on your specific dataset using proper evaluation metrics.

Handling class imbalance is as much about reframing the problem as it is about technique selection — the most important step is recognizing that accuracy is a misleading guide and committing to minority-class-aware evaluation from the start.
