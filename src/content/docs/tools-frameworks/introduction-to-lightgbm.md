---
title: Introduction to LightGBM - Fast Gradient Boosting at Scale
description: Learn how LightGBM's histogram-based splitting and leaf-wise tree growth make it a faster alternative to XGBoost on large tabular datasets.
---

LightGBM is a gradient boosting framework developed by Microsoft, designed for speed and memory efficiency on large tabular datasets while matching or exceeding the accuracy of earlier implementations like XGBoost.

## Leaf-Wise Tree Growth

Most gradient boosting implementations, including classic XGBoost, grow trees level-wise, expanding all nodes at the current depth before moving deeper. LightGBM grows trees leaf-wise: at each step, it splits whichever leaf gives the greatest reduction in loss, regardless of depth.

```text
Level-wise: expand every leaf at depth 1, then every leaf at depth 2, ...
Leaf-wise:  always expand the single leaf with the highest loss reduction, wherever it is
```

Leaf-wise growth typically reaches lower training loss with fewer leaves than level-wise growth, but it can overfit more easily on smaller datasets since it can produce deeper, more unbalanced trees — `max_depth` and `num_leaves` need more careful tuning as a result.

## Histogram-Based Splitting

LightGBM buckets continuous features into discrete histogram bins before searching for the best split, rather than sorting and scanning every unique feature value. This trades a small amount of split precision for a large reduction in computation and memory, which is what makes LightGBM noticeably faster than histogram-free implementations on large datasets.

## Native Categorical Feature Support

LightGBM can split directly on categorical features without requiring one-hot encoding, using an optimal partitioning strategy for categories based on their relationship to the target variable, which avoids the dimensionality blowup that one-hot encoding causes for high-cardinality categorical columns.

## Basic Usage

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05
)
model.fit(X_train, y_train, categorical_feature=["region", "device_type"])
```

## Practical Guidance

Prefer LightGBM over XGBoost when training speed and memory footprint on large datasets are the priority, and when your data includes high-cardinality categorical features you'd rather not one-hot encode. Because leaf-wise growth is more prone to overfitting on smaller datasets, watch validation loss closely and constrain `num_leaves` and `min_data_in_leaf` more conservatively than you might with a level-wise implementation.
