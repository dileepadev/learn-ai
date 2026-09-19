---
title: Linear Regression Fundamentals
description: Learn how linear regression fits a line through data by minimizing squared error, and the assumptions worth checking before trusting the result.
---

Linear regression models the relationship between input features and a continuous target as a weighted linear combination, remaining the starting point for understanding regression despite decades of more sophisticated methods being developed since.

## The Model

Linear regression predicts a target as a weighted sum of input features plus a bias term:

```text
y_pred = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b
```

Fitting the model means finding the weights `w` and bias `b` that minimize the difference between predictions and actual target values across the training data, almost always measured as mean squared error.

## Ordinary Least Squares

For a well-posed problem, the optimal weights that minimize squared error have a closed-form solution, found by solving the normal equations directly rather than requiring iterative optimization:

```text
w = (X^T X)^-1 X^T y
```

This closed-form solution is one of the few in machine learning where the exact optimum can be computed directly rather than approximated through gradient-based iteration, though for very large or high-dimensional datasets, iterative methods (gradient descent) are often more practical than directly inverting `X^T X`, which becomes computationally expensive or numerically unstable at scale.

## Key Assumptions Worth Checking

Linear regression's validity rests on assumptions that are easy to skip checking but matter for whether the fitted model is trustworthy: a genuinely linear relationship between features and target (checked by examining residual plots for systematic patterns rather than random scatter), homoscedasticity (residual variance staying roughly constant across the range of predictions, rather than growing for larger predicted values), and low multicollinearity among features (highly correlated input features make individual coefficient estimates unstable and hard to interpret, even if overall prediction accuracy remains fine).

## Interpreting Coefficients

A key practical advantage of linear regression over more complex models is direct interpretability: each coefficient represents the expected change in the target for a one-unit change in that feature, holding other features constant — but this interpretation only holds cleanly when features are on comparable, meaningful scales and aren't strongly correlated with each other, which is why standardizing features before fitting is common practice when coefficient interpretation matters.

## Practical Guidance

Use linear regression as a fast, interpretable baseline for any regression problem before reaching for more complex models, and examine residual plots rather than only a single aggregate error metric to catch violated assumptions early. When features are highly correlated or numerous relative to the amount of training data, apply regularization — see [[ridge-and-lasso-regression]] — rather than fitting unregularized ordinary least squares, which becomes unstable in exactly these conditions.
