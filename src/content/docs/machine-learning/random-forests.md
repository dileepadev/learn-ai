---
title: Random Forests - Bagging Decorrelated Decision Trees
description: Learn how random forests combine bootstrap aggregation with random feature selection, and how out-of-bag error and feature importance work in practice.
---

A random forest is an ensemble of decision trees trained on randomized versions of the data. Averaging many deep, high-variance trees yields a model that is accurate, robust, and requires little tuning.

## Two Sources of Randomness

1. **Bagging**: each tree trains on a bootstrap sample (rows drawn with replacement)
2. **Random feature subsets**: at each split, only a random subset of features (often `sqrt(p)` for classification) is considered

The second step decorrelates the trees. Averaging correlated predictions reduces variance less than averaging independent ones, so decorrelation is what makes forests better than plain bagging.

## Prediction

- Classification: majority vote (or averaged class probabilities)
- Regression: mean of tree predictions

## Out-of-Bag (OOB) Error

Each bootstrap sample leaves out about 37% of the rows. Predicting each row using only trees that did not see it gives a built-in validation estimate without a separate hold-out set.

## Usage

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    oob_score=True,
    n_jobs=-1,
    random_state=0,
).fit(X_train, y_train)

print(rf.oob_score_)
```

## Feature Importance

Impurity-based importance is fast but biased toward high-cardinality features. Prefer **permutation importance** on held-out data:

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=0)
```

## Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | More trees reduce variance; diminishing returns |
| `max_features` | Lower values decorrelate more |
| `max_depth`, `min_samples_leaf` | Control individual tree complexity |

## Strengths and Limitations

Strengths: handles mixed feature types, resists overfitting, needs little preprocessing. Limitations: large memory and slower inference than a single tree, poor extrapolation beyond the training range, and usually slightly less accurate than well-tuned gradient boosting on tabular data.
