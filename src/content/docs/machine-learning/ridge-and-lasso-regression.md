---
title: Ridge and Lasso Regression - Regularized Linear Models
description: Learn how L2 and L1 regularization constrain linear model coefficients differently, and why Lasso's sparsity property makes it useful for feature selection.
---

Ridge and Lasso regression both add a penalty term to ordinary linear regression's loss function to constrain coefficient magnitudes, addressing overfitting and instability that arise when features are numerous or highly correlated, but they penalize coefficients in meaningfully different ways with different practical consequences.

## Ridge Regression (L2 Penalty)

Ridge regression adds a penalty proportional to the sum of squared coefficients to the standard squared-error loss:

```text
loss = Σ(y_i - y_pred_i)² + λ * Σ(w_j²)
```

`λ` controls the regularization strength — larger values shrink coefficients more aggressively toward zero, trading some bias for reduced variance, which typically improves generalization when the unregularized model would otherwise overfit or produce unstable coefficient estimates due to correlated features. Ridge shrinks coefficients smoothly toward zero but essentially never sets them to exactly zero, so it keeps every feature in the model with reduced influence rather than removing any entirely.

## Lasso Regression (L1 Penalty)

Lasso replaces the squared-coefficient penalty with a penalty proportional to the sum of absolute coefficient values:

```text
loss = Σ(y_i - y_pred_i)² + λ * Σ|w_j|
```

This seemingly small change in the penalty's mathematical form has an important practical consequence: the L1 penalty's geometry tends to push some coefficients to exactly zero rather than merely shrinking them, effectively performing automatic feature selection as a side effect of regularization — features with coefficients set to exactly zero are functionally removed from the model.

## Why the Penalty Shape Matters

The geometric intuition is that the L1 penalty's constraint region has sharp corners along the coordinate axes, and the loss function's optimal solution tends to land exactly on one of those corners (where some coefficients are exactly zero) more often than the smooth, circular L2 constraint region does, which almost never intersects the loss surface exactly on an axis. This is why Lasso is preferred when you suspect only a subset of available features are actually relevant and want the model itself to identify which ones, while Ridge is preferred when you believe most features contribute at least a little and want to reduce their influence without eliminating any of them entirely.

## Elastic Net

Elastic Net combines both penalties with a mixing parameter, gaining some of Lasso's sparsity-inducing feature selection while retaining more of Ridge's stability when features are highly correlated — Lasso alone tends to arbitrarily pick just one feature from a group of correlated features and zero out the rest, which Elastic Net's blended penalty mitigates somewhat.

## Practical Guidance

Use Ridge when you have many correlated features and want stable, non-sparse coefficient estimates. Use Lasso when you specifically want automatic feature selection alongside regularization. In both cases, choose `λ` via cross-validation rather than a fixed default, since the right regularization strength depends entirely on your specific dataset's size and noise level.
