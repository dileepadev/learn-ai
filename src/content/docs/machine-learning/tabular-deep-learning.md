---
title: Tabular Deep Learning
description: Understand why tabular data remains uniquely challenging for deep learning, explore modern architectures like TabNet, SAINT, and TabPFN, and learn when to use neural networks versus gradient boosting for structured data.
---

Tabular data — spreadsheets, databases, feature tables — is the most common data format in industry, yet it is the domain where deep learning has historically lagged behind classical methods. Understanding why, and how modern architectures are closing the gap, is essential knowledge for any applied ML practitioner.

## Why Tabular Data Is Different

Deep learning excels at learning from **raw, high-dimensional, homogeneous** inputs (pixels, tokens, audio samples). Tabular data breaks these assumptions:

- **Heterogeneous features:** A single table may mix continuous values, categorical codes, dates, IDs, and booleans
- **Low-to-medium dimensionality:** Hundreds of features, not millions — less signal for complex architectures to exploit
- **Sparse, irregular patterns:** Missing values, rare categories, and skewed distributions are endemic
- **Feature importance is explicit:** Domain experts already know which columns matter; there's less need for learned representations
- **Small to medium dataset sizes:** Many business datasets have thousands to hundreds of thousands of rows

## The Gradient Boosting Dominance

Gradient Boosted Decision Trees (GBDT) — particularly **XGBoost**, **LightGBM**, and **CatBoost** — consistently match or beat deep learning on tabular benchmarks:

| Property | GBDT | Deep Neural Network |
|---|---|---|
| Handles mixed types natively | Yes | Requires preprocessing |
| Robust to outliers | Yes | Sensitive |
| No tuning of learning rate schedule | Largely | Complex |
| Handles low data | Well | Poorly |
| Parallel training at scale | Moderate | Excellent |
| Learns from raw features | Yes | Prefers normalized inputs |

Studies like *Why do tree-based models still outperform deep learning on tabular data?* (Grinsztajn et al., 2022) showed GBDT outperforming deep networks in the majority of benchmarks, particularly when rotational invariance of the feature space is not appropriate.

## Modern Tabular Deep Learning Architectures

### TabNet

TabNet (Arik & Pfister, 2019) uses **sequential attention** to select relevant features at each decision step, mimicking tree-based feature selection while being differentiable.

Key ideas:

- Sparsity regularization encourages the model to use few features per step
- Instance-wise feature selection adapts to each row
- Transparent: attention masks reveal which features drove each prediction

### SAINT: Self-Attention and Intersample Attention Transformer

SAINT applies Transformer-style self-attention in **two dimensions**:

- **Row-wise self-attention:** Attends over features within each sample (like a standard Transformer)
- **Column-wise intersample attention:** Attends over samples for each feature — capturing global value distributions and cross-sample patterns

This allows SAINT to exploit both feature relationships and dataset-level statistics simultaneously.

### FT-Transformer: Feature Tokenization Transformer

FT-Transformer (Gorishniy et al., 2021) tokenizes each feature (both numerical and categorical) into an embedding vector, then applies a standard Transformer encoder:

$$x_j^\text{embed} = \mathbf{W}_j^\text{num} \cdot x_j + \mathbf{b}_j \quad \text{(numerical feature)}$$
$$x_j^\text{embed} = \mathbf{E}[v_j] \quad \text{(categorical feature)}$$

A `[CLS]` token aggregates all feature embeddings for the final prediction. FT-Transformer is one of the strongest pure deep learning baselines on tabular benchmarks.

### TabPFN: Tabular Prior-Fitted Networks

TabPFN (Hollmann et al., 2022) takes a radically different approach: it is a **prior-fitted Transformer** trained once on millions of **synthetic tabular datasets** generated from Bayesian priors over data-generating processes.

At inference time, TabPFN processes the **entire training set as context** (similar to in-context learning in LLMs) and produces predictions without any gradient updates:

```
Input: [training rows | test row]  →  TabPFN  →  P(y | x_test, D_train)
```

On small datasets (≤1,000 rows), TabPFN often **matches or beats XGBoost in under a second** of compute — making it exceptional for rapid prototyping and AutoML pipelines.

### GRANDE: Gradient-Based Decision Tree Ensembles

GRANDE (Marton et al., 2023) differentiates the entire decision tree ensemble training procedure end-to-end using soft splits, enabling backpropagation through tree structure while maintaining interpretability comparable to GBDT.

## When to Use Deep Learning Over GBDT

Deep learning for tabular data pays off when:

- **You have large datasets** (>100K rows) where neural networks can exploit more capacity
- **Multi-modal data:** The table is joined with images, text, or time series (deep learning handles this naturally)
- **Online learning / streaming:** Neural networks update incrementally; GBDT requires full retraining
- **Embeddings are needed downstream:** The learned representations are inputs to another model
- **Multi-task learning:** Predicting multiple targets simultaneously in a shared architecture

## Feature Engineering Considerations

Even for deep learning, feature engineering remains critical for tabular data:

- **Embedding categorical features:** Trainable embeddings consistently beat one-hot encoding for high-cardinality categoricals
- **Cyclical encoding for time:** Use $\sin(2\pi t / T)$ and $\cos(2\pi t / T)$ for hour-of-day, day-of-week
- **Target encoding with regularization:** Mean-encode categorical features against the label with cross-validation to avoid leakage
- **Missing value indicators:** A binary flag indicating missingness + imputed value outperforms imputation alone

## Benchmark Landscape

Key benchmarks for evaluating tabular methods:

- **OpenML-CC18:** 72 curated binary classification datasets
- **Tabular Benchmark (Grinsztajn et al.):** 45 datasets with careful preprocessing
- **AutoML Benchmark:** Evaluates full pipelines including hyperparameter tuning
- **CARTE:** Addresses relational tables with string-valued columns

## Practical Recommendations

1. **Start with LightGBM or XGBoost** — they are fast, robust, and hard to beat without large datasets
2. **Try FT-Transformer or SAINT** when dataset size exceeds ~50K rows
3. **Use TabPFN** as a strong default for small datasets and rapid experimental iteration
4. **Combine GBDT + neural embeddings** for high-cardinality categorical features (e.g., entity embeddings)
5. **Always validate with cross-validation** — tabular models are sensitive to train/test distribution shifts

## Further Reading

- Grinsztajn et al. (2022), *Why do tree-based models still outperform deep learning on tabular data?*
- Gorishniy et al. (2021), *Revisiting Deep Learning Models for Tabular Data*
- Hollmann et al. (2022), *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*
- Arik & Pfister (2019), *TabNet: Attentive Interpretable Tabular Learning*
