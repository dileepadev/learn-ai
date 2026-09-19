---
title: Principal Component Analysis - Finding the Directions That Matter
description: Learn how PCA finds the directions of maximum variance in data, how to interpret components, and where PCA fits before modeling or visualization.
---

Principal Component Analysis (PCA) is a linear dimensionality reduction technique that finds new axes — principal components — ordered by how much variance in the data they capture, letting you represent data with fewer dimensions while retaining as much information as possible.

## What PCA Computes

PCA finds the directions in feature space along which the data varies most. The first principal component is the direction of maximum variance; the second is the direction of maximum remaining variance orthogonal to the first, and so on.

```text
original data: n features, correlated
PCA: compute covariance matrix -> eigenvectors (components) and eigenvalues (variance explained)
project data onto top-k components -> k new uncorrelated features, ranked by variance explained
```

Because components are computed as eigenvectors of the covariance matrix, they are guaranteed to be mutually orthogonal (uncorrelated with each other), which is why PCA is also used as a preprocessing step to remove multicollinearity before feeding features into models sensitive to correlated inputs.

## Choosing How Many Components to Keep

A scree plot shows variance explained by each successive component; a common heuristic is to keep enough components to explain some target cumulative variance (often 90-95%) or to look for an "elbow" where additional components stop contributing meaningfully. Keeping too few components loses real signal; keeping too many defeats the purpose of dimensionality reduction.

## Standardization Matters

PCA is sensitive to feature scale, since it operates on variance, and a feature measured in larger units will dominate the components purely due to its scale, not its actual informativeness. Standardizing features (zero mean, unit variance) before applying PCA is standard practice unless features are already on genuinely comparable scales.

## Limitations

PCA only captures linear relationships between features, so it can fail to compress data that has meaningful nonlinear structure — a dataset lying on a curved manifold in feature space may need far more linear components to represent than a nonlinear method (like t-SNE or UMAP for visualization, or an autoencoder for general nonlinear dimensionality reduction) would require. PCA components also are not always individually interpretable, since each is a weighted combination of many original features rather than a single meaningful variable.

## Practical Guidance

Use PCA to speed up training and reduce overfitting risk on high-dimensional data with many correlated features, and as a quick 2D or 3D visualization tool for exploring cluster structure before committing to a specific model. Do not use PCA blindly before a model that explicitly benefits from interpretable original features (like a decision tree meant to be inspected by domain experts), since the transformed components lose the direct feature meaning.
