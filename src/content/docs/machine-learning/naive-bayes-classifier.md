---
title: Naive Bayes Classifier - A Fast, Surprisingly Strong Baseline
description: Learn how Naive Bayes applies Bayes' theorem with a simplifying independence assumption, and why it remains a strong baseline for text classification.
---

Naive Bayes is a probabilistic classifier built on Bayes' theorem, with a deliberately simplifying assumption that makes it fast to train and surprisingly effective despite that assumption rarely holding exactly true.

## Bayes' Theorem Applied to Classification

To classify an input with features `x_1, ..., x_n` into class `y`, Naive Bayes picks the class that maximizes the posterior probability:

```text
P(y | x_1, ..., x_n) ∝ P(y) * P(x_1, ..., x_n | y)
```

Computing `P(x_1, ..., x_n | y)` directly requires modeling the joint distribution of all features, which is intractable for high-dimensional data. The "naive" assumption sidesteps this by assuming all features are conditionally independent given the class:

```text
P(x_1, ..., x_n | y) ≈ P(x_1 | y) * P(x_2 | y) * ... * P(x_n | y)
```

This lets the model estimate each `P(x_i | y)` independently from training data, which requires far less data than estimating the full joint distribution.

## Why It Works Despite the Wrong Assumption

Features are rarely truly independent given the class — in text classification, word co-occurrences are correlated — yet Naive Bayes often classifies correctly anyway, because classification only requires the correct class to get the highest posterior score, not an accurate absolute probability estimate. The independence assumption's errors often affect all classes' score estimates in a correlated way, preserving the correct ranking even when the raw probability estimates themselves are poorly calibrated.

## Variants for Different Data Types

Multinomial Naive Bayes models word counts and is the standard choice for text classification and spam filtering. Gaussian Naive Bayes assumes continuous features follow a normal distribution within each class, suited to numeric feature data. Bernoulli Naive Bayes models binary presence/absence features, useful when only whether a word appears matters, not how many times.

## Practical Guidance

Use Naive Bayes as an extremely fast, low-resource baseline for text classification and spam detection before investing in a more complex model — it trains on large datasets in seconds and often gets surprisingly close to more sophisticated methods on simple classification tasks. Apply Laplace (additive) smoothing to feature probability estimates to avoid zero probabilities from words unseen in training data for a given class, which would otherwise zero out the entire posterior for that class regardless of other evidence.
