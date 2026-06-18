---
title: Handling Class Imbalance
description: Practical techniques for dealing with imbalanced datasets where one class significantly outnumbers others.
---

Class imbalance occurs when one class in a classification dataset has far more examples than another. For example, fraud detection datasets may have 99% legitimate transactions and 1% fraudulent ones. A model that simply predicts "not fraud" every time achieves 99% accuracy but is completely useless — this is the accuracy paradox.

## Why Imbalance Causes Problems

Standard training minimizes overall loss. With imbalanced data, the model is incentivized to focus on the majority class and ignore the minority class because correct predictions on the majority contribute more to reducing loss. The result is high overall accuracy but very poor recall on the minority class — exactly the class you care most about.

## Evaluation Metrics for Imbalanced Data

Accuracy is misleading. Use instead:
- **Precision and Recall:** How many predicted positives are correct, and how many actual positives were found.
- **F1-Score:** Harmonic mean of precision and recall.
- **AUC-ROC:** Area under the ROC curve — measures discrimination ability across all thresholds.
- **AUC-PR (Precision-Recall curve):** More informative than ROC for severely imbalanced datasets.
- **Matthews Correlation Coefficient (MCC):** Balanced metric robust to imbalance.

## Data-Level Techniques

### Oversampling the Minority Class
Duplicate minority class samples or generate synthetic ones.
- **Random oversampling:** Simply duplicate minority examples.
- **SMOTE (Synthetic Minority Oversampling Technique):** Creates synthetic samples by interpolating between existing minority examples in feature space. More effective than random duplication.

### Undersampling the Majority Class
Remove majority class examples to balance the dataset.
- **Random undersampling:** Randomly remove majority examples. Risk: losing useful information.
- **Tomek Links / Edited Nearest Neighbours:** Remove majority examples near the decision boundary for cleaner separation.

### Combining Both: SMOTEENN, SMOTETomek
Hybrid methods that oversample the minority and undersample noisy majority examples simultaneously.

## Algorithm-Level Techniques

### Class Weights
Most classifiers (scikit-learn, XGBoost, PyTorch) accept a `class_weight` parameter. Setting `class_weight='balanced'` automatically scales the loss contribution of each class inversely to its frequency — the minority class contributes more to the loss.

### Threshold Adjustment
By default, classifiers use a 0.5 decision threshold. Lowering the threshold for the positive (minority) class increases recall at the cost of precision. Choose the threshold based on the business requirement using the precision-recall curve.

### Focal Loss
A modification of cross-entropy that downweights easy (correctly classified majority) examples and focuses learning on hard (minority) examples. Particularly effective for object detection and severe imbalance scenarios.

## Practical Recommendations

1. Start with class weights — it's free and often very effective.
2. Evaluate with F1, AUC-PR, or MCC rather than accuracy.
3. Add SMOTE if class weights alone don't achieve the desired recall.
4. Tune the decision threshold after training using the validation set.
5. Collect more minority class data if possible — data beats resampling tricks.
