---
title: Logistic Regression Explained
description: Learn how logistic regression models binary outcomes with the sigmoid function, why it's trained with cross-entropy rather than squared error, and how to interpret it.
---

Despite the name, logistic regression is a classification algorithm, not a regression algorithm in the sense of predicting a continuous value — it predicts the probability that an input belongs to a particular class.

## From Linear Regression to Probabilities

A linear combination of features, as used in [[linear-regression-fundamentals]], can output any real number, but a probability must fall between 0 and 1. Logistic regression solves this by passing the linear combination through the sigmoid function, which squashes any real-valued input into the (0, 1) range:

```text
z = w_1*x_1 + w_2*x_2 + ... + b
P(y=1 | x) = sigmoid(z) = 1 / (1 + e^-z)
```

A predicted probability above a chosen threshold (typically 0.5, though this can be adjusted based on the relative cost of false positives versus false negatives for the specific application) is classified as the positive class.

## Why Cross-Entropy, Not Squared Error

Logistic regression is trained by minimizing cross-entropy loss (also called log loss) rather than mean squared error, because squared error applied to sigmoid outputs produces a non-convex optimization landscape with local minima that gradient descent can get stuck in, while cross-entropy loss combined with the sigmoid function produces a convex loss surface with a single global minimum, making optimization well-behaved and reliable:

```text
loss = -[y * log(p) + (1 - y) * log(1 - p)]
```

This loss penalizes confident wrong predictions much more heavily than uncertain wrong predictions — predicting `p = 0.99` for an example that's actually negative incurs a far larger loss than predicting `p = 0.6` for the same misclassification, which shapes the model toward well-calibrated confidence rather than just correct classification.

## Interpreting Coefficients as Log-Odds

Logistic regression coefficients are most naturally interpreted in terms of log-odds rather than raw probability: a one-unit increase in a feature changes the log-odds of the positive class by that feature's coefficient, and exponentiating a coefficient gives an odds ratio — how much the odds of the positive outcome multiply for a one-unit increase in that feature, holding other features fixed. This log-odds interpretation is less intuitive than linear regression's direct coefficient interpretation but is precise and widely used in fields like epidemiology and social science specifically because it correctly reflects logistic regression's underlying model of probability.

## Practical Guidance

Use logistic regression as a fast, well-calibrated, interpretable baseline for binary classification, particularly when you need to explain which features drive predictions to a non-technical audience or satisfy a regulatory requirement for interpretable decision-making. For classes with a class imbalance, evaluate with precision, recall, and calibration curves specifically, not accuracy alone, since a model can achieve high accuracy on an imbalanced dataset by simply favoring the majority class.
