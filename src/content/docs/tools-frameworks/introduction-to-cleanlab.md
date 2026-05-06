---
title: Introduction to Cleanlab
description: A practical guide to Cleanlab, an open-source Python library for automated data-centric AI that detects and corrects label errors, finds data quality issues, and improves ML model performance by cleaning noisy datasets.
---

# Introduction to Cleanlab

**Cleanlab** is an open-source Python library for **data-centric AI** — the practice of improving ML performance by improving data quality rather than only improving models. It implements *confident learning*, a theoretically grounded framework for estimating noise rates and identifying mislabeled examples in datasets. Cleanlab works with any ML model and any data type (tabular, text, images, multi-label), dramatically improving model performance simply by cleaning training labels.

## The Label Noise Problem

Real-world datasets are noisy. Even carefully curated benchmarks contain significant label errors:

- CIFAR-10: ~3.4% label errors
- ImageNet: ~5.8% label errors
- QuickDraw: ~10%+ estimated
- Medical annotation datasets: 15–30% depending on task difficulty

Training on noisy labels degrades model performance, calibration, and generalization. Cleanlab finds these errors automatically.

## Confident Learning Theory

The core idea in confident learning (Northcutt et al., 2021) is to estimate the **joint distribution** of noisy labels $\tilde{y}$ and true latent labels $y^*$:

$$C_{\tilde{y}, y^*}[i][j] = \text{(estimated number of examples labeled class } i \text{ with true class } j\text{)}$$

This **confusion matrix over latent labels** is estimated from out-of-fold predicted probabilities:

1. Train any model with cross-validation to get $P(\tilde{y} \mid x)$ for each example
2. For each class, compute a per-class threshold $t_j = \text{avg}_{x:\tilde{y}=j} P(\tilde{y}=j \mid x)$
3. An example $x$ is labeled $j$ but likely true class $k$ if $P(\tilde{y}=k \mid x) \geq t_k$
4. Count these assignments to estimate $C_{\tilde{y}, y^*}$, then identify errors where $i \neq j$

## Quick Start: Finding Label Errors

```python
from cleanlab import Datalab
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np

# Simulate noisy labeled data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
# Add 10% label noise
noise_mask = np.random.rand(len(y)) < 0.1
y_noisy = y.copy()
y_noisy[noise_mask] = 1 - y_noisy[noise_mask]

# Get out-of-fold predicted probabilities
from sklearn.model_selection import cross_val_predict

clf = LogisticRegression(max_iter=1000)
pred_probs = cross_val_predict(clf, X, y_noisy, cv=5, method="predict_proba")

# Find all data issues
lab = Datalab(data={"X": X, "label": y_noisy}, label_name="label")
lab.find_issues(pred_probs=pred_probs, issue_types={"label": {}})

# Review issues
lab.report()
print(lab.get_issues("label").head(10))
```

## Classification with `find_label_issues`

```python
from cleanlab.filter import find_label_issues
import numpy as np

# pred_probs: (N, K) out-of-fold probabilities from cross-validation
# labels: (N,) integer class labels

ordered_label_issues = find_label_issues(
    labels=y_noisy,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence",  # or "normalized_margin", "confidence_weighted_entropy"
    filter_by="prune_by_noise_rate",              # or "prune_by_class", "both", "confident_learning"
    frac_noise=0.1,                               # expected noise fraction (optional)
)

print(f"Found {len(ordered_label_issues)} label issues")
print(f"Indices (ranked by severity): {ordered_label_issues[:10]}")
```

## Datalab: Multi-Issue Data Auditing

`Datalab` is Cleanlab's high-level interface that detects multiple types of data quality issues beyond label errors:

```python
from cleanlab import Datalab
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load text dataset
dataset = load_dataset("ag_news", split="train[:5000]")
texts = dataset["text"]
labels = dataset["label"]

# Get embeddings for outlier/duplicate detection
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

# Get pred_probs from any classifier (e.g., fine-tuned BERT)
# pred_probs = ...  (N, num_classes) from cross-validation

lab = Datalab(
    data={"text": texts, "label": labels},
    label_name="label",
)
lab.find_issues(
    pred_probs=pred_probs,
    features=embeddings,
    issue_types={
        "label": {},                   # mislabeled examples
        "outlier": {"threshold": 0.05},  # out-of-distribution examples
        "near_duplicate": {},          # redundant examples
        "non_iid": {},                 # non-IID data distribution issues
    },
)
lab.report()
```

Issue types detected by Datalab:

- **Label issues**: mislabeled examples
- **Outliers**: examples atypical of their class or the whole dataset
- **Near-duplicates**: nearly identical examples that may bias training
- **Non-IID issues**: train/test distribution problems
- **Underperforming groups**: subgroups where the model performs poorly

## Image Data Cleaning

```python
from cleanlab.object_detection import find_label_issues as find_od_issues

# For object detection: list of per-image label dicts and prediction dicts
label_issues = find_od_issues(
    labels=train_labels,          # list of {"bboxes": [...], "labels": [...]} per image
    predictions=model_predictions,  # list of {"pred_label": [...], "pred_bbox": [...], "pred_score": [...]}
)
# Returns per-image quality scores
```

## Cleaning and Retraining

```python
from cleanlab.filter import find_label_issues
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Find issues
issues = find_label_issues(labels=y_noisy, pred_probs=pred_probs)

# 2. Filter out noisy examples
clean_mask = np.ones(len(y_noisy), dtype=bool)
clean_mask[issues] = False
X_clean, y_clean = X[clean_mask], y_noisy[clean_mask]

# 3. Retrain on cleaned data
clf_clean = RandomForestClassifier(n_estimators=100)
clf_clean.fit(X_clean, y_clean)

# 4. Compare performance (using ground truth labels for evaluation)
acc_noisy = accuracy_score(y, RandomForestClassifier().fit(X, y_noisy).predict(X))
acc_clean = accuracy_score(y, clf_clean.predict(X))
print(f"Accuracy on noisy data: {acc_noisy:.3f}")
print(f"Accuracy on cleaned data: {acc_clean:.3f}")
```

## Benchmarks

Cleanlab typically improves accuracy by 2–15% on real-world noisy datasets:

| Dataset | Noise Rate | Baseline Acc | After Cleanlab | Improvement |
|---|---|---|---|---|
| CIFAR-10 (real errors) | 3.4% | 94.2% | 95.1% | +0.9% |
| Amazon Reviews | ~8% | 89.3% | 91.7% | +2.4% |
| WebVision (web images) | ~20% | 73.1% | 78.4% | +5.3% |
| Clinical notes NER | ~15% | 81.2% | 86.8% | +5.6% |

## Integration with Popular Frameworks

Cleanlab works with any model that produces probability estimates:

```python
# With scikit-learn
from cleanlab.classification import CleanLearning
from sklearn.svm import SVC

cl = CleanLearning(clf=SVC(probability=True))
cl.fit(X_train, y_noisy_train)
cl.predict(X_test)

# With PyTorch models — pass pred_probs from model.predict_proba()
# With HuggingFace — use transformers pipeline output probabilities
# With XGBoost / LightGBM — use predict_proba() output
```

## Summary

Cleanlab implements confident learning — a rigorous framework for finding and correcting label errors in any dataset. By estimating the joint distribution of noisy and true labels from out-of-fold predicted probabilities, it identifies mislabeled examples without requiring clean labels for training. The `Datalab` interface extends this to a comprehensive data audit detecting outliers, duplicates, and distribution shift. For any ML project where data quality is uncertain, running Cleanlab before training is one of the highest-leverage data-centric improvements available.
