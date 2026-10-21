---
title: The Bias-Variance Tradeoff
description: Learn how prediction error decomposes into bias, variance, and noise, how model complexity moves between them, and how modern overparameterized models complicate the classic picture.
---

The bias-variance tradeoff is a foundational idea for understanding why models underfit or overfit.

## The Decomposition

For squared error, the expected prediction error at a point decomposes as:

```text
E[(y - f_hat(x))^2] = Bias^2 + Variance + Irreducible noise
```

- **Bias**: error from wrong assumptions; how far the average model (over different training sets) is from the true function
- **Variance**: how much predictions change when the training set changes
- **Irreducible noise**: randomness in the data no model can remove

## Intuition

| | Low variance | High variance |
|---|---|---|
| **Low bias** | Ideal | Overfitting |
| **High bias** | Underfitting | Worst case |

A linear model on a curved relationship has high bias. A deep unpruned tree memorizes training noise and has high variance.

## Complexity Curve

As model complexity grows, bias falls and variance rises. Test error is typically U-shaped, with the best generalization near the minimum.

## Diagnosing with Learning Curves

```text
High bias:     train error high, validation error high, curves converge
High variance: train error low, validation error much higher, large gap
```

## Remedies

**For high bias**: more expressive model, richer features, less regularization.

**For high variance**: more training data, regularization, simpler model, early stopping, dropout, ensembling (bagging reduces variance), feature selection.

## Estimating It Empirically

```python
import numpy as np

preds = []
for seed in range(100):
    Xs, ys = resample(X_train, y_train, random_state=seed)
    preds.append(model.fit(Xs, ys).predict(X_test))
preds = np.array(preds)
variance = preds.var(axis=0).mean()
bias_sq = ((preds.mean(axis=0) - y_true) ** 2).mean()
```

## Beyond the Classic View

Large neural networks can interpolate training data yet still generalize, and test error can decrease again past the interpolation threshold, a pattern called **double descent**. The classic U-curve remains a useful guide for classical models but is not the full story for heavily overparameterized ones.
