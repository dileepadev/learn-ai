---
title: Conformal Prediction
description: Learn how conformal prediction provides statistically rigorous prediction intervals and sets with guaranteed coverage — a distribution-free uncertainty quantification framework applicable to any machine learning model.
---

**Conformal prediction** is a framework for constructing **prediction intervals** (for regression) or **prediction sets** (for classification) that come with a formal statistical guarantee: the true label is contained within the predicted set with at least a user-specified probability, regardless of the underlying data distribution and model. Unlike Bayesian credible intervals or bootstrap confidence intervals, conformal prediction requires no distributional assumptions — it is **distribution-free** and works with any trained model as a black box.

In an era where AI systems make consequential decisions in medicine, finance, and safety-critical applications, knowing not just *what* a model predicts but *how confident to be* in that prediction is essential. Conformal prediction provides the principled foundation for calibrated, trustworthy uncertainty quantification.

## The Core Guarantee

Given:

- A calibration dataset $\{(x_i, y_i)\}_{i=1}^n$ drawn exchangeably from the same distribution as future test points.
- A target **coverage level** $1 - \alpha$ (e.g., 90%).

Conformal prediction produces prediction sets $\mathcal{C}(x_{n+1})$ such that:

$$P(y_{n+1} \in \mathcal{C}(x_{n+1})) \geq 1 - \alpha$$

This is a **marginal coverage** guarantee — averaged over the randomness in both the calibration data and the test point. It holds exactly (not approximately) under the exchangeability assumption, for any model, any data distribution, and any sample size $n$.

## The Nonconformity Score

The key ingredient in conformal prediction is the **nonconformity score** $s(x, y)$ — a function that measures how "unusual" or "nonconforming" a label $y$ is for a given input $x$. Higher scores indicate greater nonconformity.

For regression, a natural choice is the absolute residual:

$$s(x, y) = |f(x) - y|$$

For classification, a common choice is one minus the predicted class probability:

$$s(x, y) = 1 - \hat{p}(y \mid x)$$

Any nonconformity score can be used — including learned scores from neural networks.

## Split Conformal Prediction

**Split conformal prediction** (also called **inductive conformal prediction**) is the most practical variant:

1. **Train** a model $f$ on a training set (separate from the calibration set).
2. **Calibrate**: Compute nonconformity scores $s_1, s_2, \ldots, s_n$ for all $n$ calibration examples using the trained model.
3. **Quantile**: Find the $(1-\alpha)$ empirical quantile of the calibration scores:

$$\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^n;\, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

1. **Predict**: For a new test point $x_{n+1}$, produce the prediction set:

$$\mathcal{C}(x_{n+1}) = \{y : s(x_{n+1}, y) \leq \hat{q}\}$$

### Regression Example

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Split data: train / calibration / test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Train model on training set
model = GradientBoostingRegressor().fit(X_train, y_train)

# Calibration: compute nonconformity scores (absolute residuals)
cal_scores = np.abs(model.predict(X_cal) - y_cal)

# Find (1-alpha) quantile with finite-sample correction
alpha = 0.1  # Target 90% coverage
n = len(cal_scores)
q_level = np.ceil((n + 1) * (1 - alpha)) / n
q_hat = np.quantile(cal_scores, q_level, method="higher")

# Predict: generate prediction intervals for test set
y_pred = model.predict(X_test)
lower = y_pred - q_hat
upper = y_pred + q_hat

# Evaluate empirical coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"Target coverage: {1-alpha:.1%}, Empirical coverage: {coverage:.1%}")
```

### Classification Example

```python
from sklearn.neural_network import MLPClassifier

# Train softmax classifier
clf = MLPClassifier().fit(X_train, y_train)

# Calibration scores: 1 - predicted probability of true class
cal_probs = clf.predict_proba(X_cal)
cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]

# Quantile
q_hat = np.quantile(cal_scores, q_level, method="higher")

# Prediction sets: include all classes with score <= q_hat
test_probs = clf.predict_proba(X_test)
prediction_sets = test_probs >= (1 - q_hat)

# Coverage: true class is included in prediction set
coverage = np.mean(prediction_sets[np.arange(len(y_test)), y_test])
avg_set_size = prediction_sets.sum(axis=1).mean()
print(f"Coverage: {coverage:.1%}, Average set size: {avg_set_size:.2f}")
```

## Conformal Risk Control

**Conformal Risk Control (CRC)** (Angelopoulos et al., 2022) generalizes conformal prediction beyond simple coverage. Instead of controlling coverage probability, CRC controls any **monotone risk function** $\ell(y, \mathcal{C}(x))$:

$$\mathbb{E}[\ell(y, \mathcal{C}(x))] \leq \alpha$$

Examples of risk functions:

- **False negative rate**: Fraction of test examples where the true label is excluded from the set.
- **Graph distance**: For structured prediction, the maximum acceptable graph distance between prediction and ground truth.
- **Expected calibration error**: Controlling miscalibration rather than coverage.

CRC enables tailored guarantees for specific downstream objectives — for example, controlling false negative rate in medical diagnosis rather than overall coverage.

## Adaptive Conformal Prediction

**Adaptive prediction sets (APS)** / **RAPS** produce sets whose sizes adapt to input difficulty:

- Easy inputs get small, tight prediction sets.
- Ambiguous or out-of-distribution inputs get larger, more cautious sets.

RAPS (Regularized Adaptive Prediction Sets, Angelopoulos et al., 2020) adds a penalty for larger prediction sets to the nonconformity score, encouraging tighter sets on easy examples while maintaining coverage guarantees.

The set size serves as an uncertainty signal — when RAPS returns a prediction set containing many classes, the model is genuinely uncertain about that input.

## Conformal Prediction for LLMs

Applying conformal prediction to language model outputs is an active research area:

### Selective Prediction

**Abstain when uncertain**: A conformal selective prediction system abstains from answering when the model's uncertainty (measured via nonconformity score) exceeds the calibrated threshold — controlling the fraction of erroneous responses.

### Factuality Sets

**LM-Conformal** and related approaches construct answer sets that contain the correct answer with guaranteed probability — useful for question answering where the model may not be well-calibrated:

```python
# For open-ended QA: score candidate answers using log-probabilities
# nonconformity score = negative log-probability of generating the true answer
# Calibrate threshold on examples with known correct answers
# At test time: include all candidate answers with score <= threshold
```

### Confidence Regions for Structured Outputs

For structured generation tasks (JSON, code, tables), conformal prediction can define regions of acceptable outputs with coverage guarantees — important for safety-critical code generation.

## Mondrian (Conditional) Conformal Prediction

Standard conformal prediction guarantees **marginal** coverage — coverage averaged over all inputs. For some applications, **conditional coverage** — the guarantee holds for each specific input $x$ — is more desirable.

**Mondrian conformal prediction** approximates conditional coverage by grouping calibration examples into classes (e.g., by predicted confidence) and computing separate quantiles per class. This provides coverage guarantees conditional on class membership — tighter than marginal guarantees for heterogeneous inputs.

True conditional coverage without additional assumptions requires an infinite calibration set, so Mondrian conformal prediction is the practical approximation.

## Coverage Guarantees: What They Mean in Practice

Understanding the scope of conformal guarantees is important:

- **Marginal, not conditional**: The guarantee is averaged over all inputs — it does not promise 90% coverage specifically for rare subgroups.
- **Exchangeability, not i.i.d.**: The guarantee requires exchangeability (weaker than i.i.d.) — it breaks under distribution shift between calibration and deployment.
- **Coverage, not calibration**: Coverage guarantees the true label is *included*; it doesn't guarantee correct probability estimates for specific predictions.

Despite these limitations, conformal prediction's distribution-free finite-sample guarantees are stronger than those of most alternative uncertainty quantification methods, making it an essential tool for responsible ML deployment.

## Comparison with Other Uncertainty Methods

| Method | Distributional assumptions | Coverage guarantee | Scalability |
| --- | --- | --- | --- |
| Conformal prediction | None (exchangeability) | Exact finite-sample | High |
| Bayesian credible intervals | Requires prior + likelihood | Asymptotic | Low-medium |
| Bootstrap | Weak (smoothness) | Asymptotic | Medium |
| MC Dropout | Model uncertainty only | None | High |
| Ensemble intervals | None | Empirical only | Medium |

Conformal prediction provides the strongest theoretical guarantees with the fewest assumptions — at the cost of requiring a held-out calibration set and producing marginal rather than conditional coverage.
