---
title: Introduction to XGBoost - The Workhorse of Tabular Machine Learning
description: Learn why XGBoost's regularized, efficient implementation of gradient boosting made it a dominant choice for tabular data problems.
---

XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient-boosted decision trees that became the dominant algorithm for structured, tabular data problems, particularly in machine learning competitions, well before deep learning approaches were competitive on this kind of data.

## Gradient Boosting Recap

Gradient boosting builds an ensemble of decision trees sequentially, where each new tree is trained to correct the errors (residuals, or more precisely the gradient of the loss function) of the ensemble built so far:

```text
prediction = tree_1(x)
prediction += learning_rate * tree_2(x)   # trained to correct tree_1's errors
prediction += learning_rate * tree_3(x)   # trained to correct the errors so far
...
```

## What XGBoost Adds

XGBoost adds explicit regularization terms (penalizing tree complexity — number of leaves and leaf weight magnitude) directly into the objective function, which reduces overfitting compared to unregularized gradient boosting. It also uses a second-order Taylor approximation of the loss function when choosing tree splits, giving more informed split decisions than first-order gradient information alone, and implements this efficiently with techniques like approximate split-finding on binned features and native support for missing values, letting it train on very large tabular datasets significantly faster than earlier gradient boosting implementations.

## Basic Usage

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    reg_lambda=1.0
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)
```

Early stopping monitors validation performance during training and halts once it stops improving, which is the standard way to prevent overfitting without manually tuning the number of trees.

## Why Tabular Data Still Favors Gradient Boosting

Despite deep learning's dominance in vision, language, and audio, gradient-boosted trees remain competitive or superior on many tabular datasets, particularly with heterogeneous feature types, missing values, and moderate dataset sizes, because tree-based splits naturally handle these properties without the feature engineering or architecture search that neural networks often need to match them.

## Practical Guidance

Start with XGBoost (or the closely related LightGBM and CatBoost) as a strong default baseline for any tabular prediction problem before reaching for a neural network approach. Tune `max_depth`, `learning_rate`, and regularization parameters together — a smaller learning rate with more trees and early stopping is usually more robust than a large learning rate with a fixed, small number of trees.
