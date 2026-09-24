---
title: Introduction to CatBoost - Gradient Boosting Built for Categorical Data
description: Learn how CatBoost handles categorical features natively and reduces prediction bias through ordered boosting, and how it compares to XGBoost and LightGBM.
---

CatBoost (Category Boosting) is a gradient boosting library from Yandex, designed specifically to handle categorical features well without extensive preprocessing, while also addressing a subtle bias that affects other gradient boosting implementations.

## Native Categorical Feature Handling

Rather than requiring one-hot or manual target encoding, CatBoost encodes categorical features using ordered target statistics computed only from data preceding each example in a random permutation:

```text
For each categorical value, encode it using the target statistic
computed only from previously seen examples (in permutation order),
not the whole dataset -- avoiding target leakage from an example into its own encoding
```

This ordered encoding is central to why CatBoost handles high-cardinality categorical features (like user IDs or zip codes) more gracefully than naive target encoding, which would otherwise leak target information into each example's own feature value and cause overfitting.

## Ordered Boosting

Standard gradient boosting computes gradients for each training example using a model that was, at some point during training, fit on that very example, creating a subtle "prediction shift" bias between training and inference behavior. CatBoost's ordered boosting mode addresses this by, similarly to its categorical encoding, computing each example's gradient using only a model trained on examples preceding it in a random permutation, avoiding this target leakage in the boosting process itself.

## Basic Usage

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    cat_features=["region", "device_type", "user_segment"]
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

Passing column names or indices via `cat_features` is often the only preprocessing needed for categorical columns — no manual encoding step required.

## CatBoost vs. XGBoost and LightGBM

All three are strong gradient boosting implementations with broadly similar accuracy on many tabular tasks; the practical differences are about default behavior and data shape. CatBoost tends to need less feature engineering upfront when categorical features dominate a dataset, while [[introduction-to-xgboost]] and [[introduction-to-lightgbm]] often train faster on purely numeric, lower-cardinality data.

## Practical Guidance

Reach for CatBoost first when your tabular dataset has many categorical columns, especially high-cardinality ones, since its native handling typically saves meaningful preprocessing effort and often improves accuracy over manual encoding. Benchmark all three major gradient boosting libraries on a held-out validation set for any serious tabular modeling project, since the best performer genuinely varies by dataset characteristics.
