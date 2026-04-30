---
title: Shapley Values and Feature Attribution in ML
description: Understand how Shapley values from cooperative game theory provide theoretically principled feature attributions for any machine learning model — covering SHAP, TreeSHAP, KernelSHAP, LIME, and practical implementation for model interpretability.
---

**Feature attribution** answers a deceptively simple question: *how much did each input feature contribute to this prediction?* For regulated industries, safety-critical systems, and debugging complex models, this explanation is as important as the prediction itself. Shapley values — imported from cooperative game theory — provide the only attribution method with a complete set of desirable theoretical guarantees.

## The Feature Attribution Problem

Consider a credit scoring model that rejects a loan application. The applicant asks why. A black-box model cannot explain its decision; a feature attribution method can say: "Your debt-to-income ratio contributed −42 points, your credit history contributed +28 points, and your employment length contributed +8 points to the decision."

Unlike feature importance in tree models (which is global and aggregated), attribution methods are **local** — explaining a single prediction for a specific instance.

## Shapley Values: The Game-Theoretic Foundation

Shapley values originate from Lloyd Shapley's 1953 work on fair payoff distribution in cooperative games. The setup:

- **Players**: Input features $x_1, x_2, \ldots, x_d$
- **Game**: The model prediction $f(\mathbf{x})$
- **Payoff**: How much of $f(\mathbf{x}) - \mathbb{E}[f(\mathbf{x})]$ (the difference from baseline) should each feature receive?

The Shapley value $\phi_i$ for feature $i$ is:

$$\phi_i = \sum_{S \subseteq \mathcal{F} \setminus \{i\}} \frac{|S|!\,(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \left[ f_{S \cup \{i\}}(\mathbf{x}) - f_S(\mathbf{x}) \right]$$

where $S$ ranges over all subsets of features not containing $i$, and $f_S(\mathbf{x})$ is the model prediction when only features in $S$ are "known" (others are marginalized out).

This is the **only attribution method** satisfying all four axioms simultaneously:

| Axiom | Meaning |
|---|---|
| **Efficiency** | Attributions sum exactly to $f(\mathbf{x}) - \mathbb{E}[f(\mathbf{x})]$ |
| **Symmetry** | Equally contributing features receive equal attribution |
| **Dummy** | A feature with no influence receives zero attribution |
| **Linearity** | Attributions combine linearly for sums of models |

## SHAP: Shapley Additive Explanations

**SHAP** (Lundberg & Lee, 2017) is the dominant practical implementation of Shapley values for ML. It frames attribution as finding additive explanation functions:

$$g(\mathbf{z}') = \phi_0 + \sum_{i=1}^{d} \phi_i z'_i$$

where $\mathbf{z}' \in \{0,1\}^d$ indicates which features are "present." SHAP unifies multiple prior explanation methods (LIME, DeepLIFT, SHAP) under this additive framework and proves they are all approximations to Shapley values.

### TreeSHAP: Exact Computation for Tree Models

For decision trees, random forests, and gradient boosting (XGBoost, LightGBM, CatBoost), exact Shapley values can be computed in polynomial time via the TreeSHAP algorithm:

```python
import shap
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Train model
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute exact SHAP values — O(TLD^2) where T=trees, L=leaves, D=depth
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Local explanation: waterfall plot for one prediction
shap.plots.waterfall(explainer(X_test)[0])

# Global explanation: feature importance via mean |SHAP|
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Dependence plot: feature value vs. SHAP value (reveals interactions)
shap.dependence_plot("worst radius", shap_values, X_test)
```

TreeSHAP computes exact Shapley values in $O(TLD^2)$ time, where $T$ is number of trees, $L$ is number of leaves, and $D$ is max depth — making it practical for large ensembles.

### KernelSHAP: Model-Agnostic Approximation

For any model (neural networks, SVMs, pipelines), KernelSHAP approximates Shapley values by treating the model as a black box:

```python
import shap
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# KernelSHAP works with any model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Use a background sample to marginalize missing features
background = shap.sample(X_train, 100)  # 100-sample background

# KernelSHAP: model-agnostic, slower than TreeSHAP
explainer = shap.KernelExplainer(model.predict_proba, background)

# Explain a subset (KernelSHAP is slow for large datasets)
shap_values = explainer.shap_values(X_test[:50])

print(f"Attribution sum check: {shap_values[1][0].sum():.4f}")
print(f"Prediction - baseline: {model.predict_proba(X_test[:1])[0,1] - explainer.expected_value[1]:.4f}")
```

KernelSHAP samples random feature coalitions, evaluates the model on each, and solves a weighted linear regression problem to estimate Shapley values. The key hyperparameter is the number of coalition samples (more = more accurate, slower).

### DeepSHAP and GradientSHAP for Neural Networks

```python
import shap
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = SimpleNet(X_train.shape[1])
X_train_tensor = torch.FloatTensor(X_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)

# GradientSHAP: combines integrated gradients with Shapley sampling
background = X_train_tensor[:100]
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_test_tensor[:20])
```

## LIME: Local Interpretable Model-Agnostic Explanations

LIME (Ribeiro et al., 2016) approximates the model locally around a prediction with an interpretable surrogate:

1. Sample perturbations of the input.
2. Weight each perturbation by proximity to the original instance.
3. Fit a sparse linear model to the perturbed samples.
4. Use the linear model's coefficients as explanations.

```python
import lime
import lime.lime_tabular
import numpy as np

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["malignant", "benign"],
    discretize_continuous=True
)

# Explain one prediction
instance = X_test.iloc[0].values
explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=10
)

explanation.show_in_notebook()
```

**LIME vs. SHAP trade-offs**:

| Property | LIME | SHAP (KernelSHAP) |
|---|---|---|
| Theoretical guarantee | Local approximation only | Shapley axioms satisfied |
| Consistency | Can vary between runs | More stable |
| Speed | Fast | Slower (more samples needed) |
| Model access | Black-box | Black-box |
| Attribution sum | No guarantee | Sums to $f(x) - E[f(x)]$ |

## Global Interpretability from Local Explanations

Aggregating local SHAP values across the dataset reveals global model behavior:

```python
import pandas as pd
import shap

# Global feature importance: mean absolute SHAP value
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=False)

print("Top-5 globally important features:")
print(mean_shap.head())

# SHAP interaction values — detect feature interactions
shap_interaction = explainer.shap_interaction_values(X_test)
# shap_interaction[i, j, k] = interaction between features j and k for instance i

# Beeswarm plot: global overview, local detail
shap.summary_plot(shap_values, X_test)
```

## Practical Considerations

**Baseline choice matters**: SHAP marginalizes missing features by expectation over the background dataset. A poor background (wrong distribution) produces misleading attributions. Use a representative sample of training data.

**Correlated features**: When features are highly correlated, SHAP distributes attribution across them. This is technically correct (Shapley values handle correlation properly) but can feel unintuitive — a feature "taking credit" from a correlated but causally unrelated feature.

**Computational cost**: For KernelSHAP, cost scales as $O(2^d)$ (exponential in features) — approximated by sampling. For $d > 30$ features, use TreeSHAP when possible.

**Causal vs. correlational**: SHAP measures statistical contribution, not causal effect. High SHAP attribution for a spuriously correlated feature does not imply it causes the outcome.

SHAP has become the de facto standard for ML model explanation in production — integrated into MLflow, Azure ML, Amazon SageMaker Clarify, and every major ML platform. For any model affecting high-stakes decisions, computing and reviewing SHAP values should be a standard part of the evaluation pipeline.
