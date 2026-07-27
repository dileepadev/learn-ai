---
title: Feature Engineering Techniques - Shaping Raw Data for Better Models
description: Learn the practical transformations that turn raw, messy data into features that machine learning models can actually learn from.
---

Feature engineering is the process of transforming raw data into inputs that better expose the patterns a model needs to learn. Even with powerful algorithms, poorly constructed features can cap performance well below what better-prepared data would allow.

```text
raw data -> cleaning -> transformation -> encoding -> selection -> model-ready features
```

## Handling Numeric Features

- **Scaling**: many algorithms (SVMs, k-nearest neighbors, gradient descent-based models) are sensitive to feature magnitude, so numeric features are commonly standardized (zero mean, unit variance) or min-max scaled to a fixed range.
- **Binning**: converting a continuous variable into discrete ranges (e.g., age into age groups) can help tree-based models split more effectively and can reduce sensitivity to noisy outliers.
- **Log and power transforms**: skewed distributions (like income or word counts) often become more model-friendly after a log transform, which compresses large values and spreads out small ones.

## Handling Categorical Features

- **One-hot encoding** creates a binary column per category, appropriate when categories have no inherent order and the cardinality is manageable.
- **Target encoding** replaces a category with a statistic of the target variable for that category (e.g., mean target value), useful for high-cardinality categorical features but prone to leakage if not computed carefully with cross-validation.
- **Ordinal encoding** assigns integers based on a meaningful order (e.g., "low", "medium", "high"), appropriate only when that order genuinely reflects the data.

## Interaction and Domain Features

Combining two features can reveal patterns invisible to either alone—for example, multiplying "price" and "quantity" to get "total spend," or computing the ratio of two measurements. Domain expertise often produces the most valuable features: a fraud detection model benefits enormously from a hand-crafted "transactions in the last hour" feature that no generic transformation would discover automatically.

## Handling Missing Data

Missingness itself can be informative—adding a binary "was this value missing" indicator column alongside an imputed value often outperforms imputation alone, since it preserves the signal that a value was absent rather than hiding it.

## Avoiding Leakage

The most common feature engineering mistake is **data leakage**: computing a feature (like a target encoding or a normalization statistic) using information from the full dataset, including validation or test data, that would not be available at prediction time in production. Any statistic used for feature construction should be computed only from training data and applied consistently to validation and test sets.

Feature engineering has become less central for domains where deep learning models learn representations automatically (images, text, audio), but it remains critical for tabular data, where classical models and gradient-boosted trees still often outperform deep learning and depend heavily on well-constructed input features.
