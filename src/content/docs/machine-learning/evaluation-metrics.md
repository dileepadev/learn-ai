---
title: Evaluation Metrics for Machine Learning
description: A guide to the key metrics used to measure the performance of ML models.
---

How do we know if a Machine Learning model is performing well? Selecting the right evaluation metric is crucial to understanding its strengths and weaknesses.

## Classification Metrics

In classification tasks, we predict discrete categories.

### 1. Accuracy

The simplest metric—the ratio of correct predictions to total predictions.

$$ \text{Accuracy} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}} $$

*Note: Accuracy can be misleading if the classes are imbalanced.*

### 2. Precision and Recall

- **Precision:** Of all positive predictions, how many were actually positive?
  $$ \text{Precision} = \frac{\text{TP}}{\text{TP + FP}} $$
- **Recall (Sensitivity):** Of all actual positive cases, how many did the model find?
  $$ \text{Recall} = \frac{\text{TP}}{\text{TP + FN}} $$

### 3. F1-Score

The harmonic mean of Precision and Recall. It's a better metric when you want a balance between the two.

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

## Regression Metrics

In regression tasks, we predict continuous values.

### 1. Mean Absolute Error (MAE)

The average of the absolute differences between predictions and actual values.

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

### 2. Mean Squared Error (MSE)

The average of the squared differences. It penalizes larger errors more heavily than MAE.

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

### 3. Root Mean Squared Error (RMSE)

The square root of the MSE, which brings the error back to the original units.

$$ \text{RMSE} = \sqrt{\text{MSE}} $$
