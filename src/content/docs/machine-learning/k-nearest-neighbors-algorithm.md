---
title: K-Nearest Neighbors - Classification and Regression by Proximity
description: Learn how k-nearest neighbors makes predictions from the closest labeled examples, and the practical tradeoffs in choosing k and a distance metric.
---

K-nearest neighbors (KNN) is one of the simplest machine learning algorithms: to classify or predict a value for a new point, look at the `k` closest labeled points in the training data and combine their labels or values.

## How Prediction Works

For classification, KNN takes a majority vote among the `k` nearest neighbors' labels. For regression, it typically averages their target values, sometimes weighted by inverse distance so closer neighbors count more.

```text
new point -> compute distance to every training point
          -> select k closest points
          -> classification: majority vote of their labels
          -> regression: (weighted) average of their values
```

There is no explicit training phase beyond storing the data — all computation happens at prediction time, which is why KNN is called a "lazy learning" method, in contrast to algorithms that build an explicit model during a distinct training phase.

## Choosing K and a Distance Metric

A small `k` (like `k=1`) makes predictions very sensitive to individual noisy points and can overfit sharply to local data quirks. A large `k` smooths predictions but risks underfitting by averaging over points that aren't actually similar to the query point, and can be dominated by a majority class in imbalanced datasets. `k` is typically chosen via cross-validation. The distance metric matters just as much: Euclidean distance is standard for continuous numeric features, but categorical or mixed-type features need a different metric (Hamming distance for categorical features, or a combined distance function), and features should be scaled to comparable ranges since unscaled features with larger numeric ranges dominate distance calculations regardless of their actual relevance.

## The Curse of Dimensionality

In high-dimensional feature spaces, the distance between the nearest and farthest points tends to converge, making "nearest" a less meaningful concept and degrading KNN's performance — this is one of the clearest practical illustrations of the curse of dimensionality. Dimensionality reduction (like PCA) or feature selection before applying KNN often substantially improves results on high-dimensional data.

## Practical Guidance

Use KNN as an interpretable, assumption-light baseline for small-to-medium datasets, especially when the decision boundary is genuinely irregular and not well captured by a linear or tree-based model. For large datasets, exact KNN's prediction-time cost (computing distance to every training point) becomes prohibitive — use approximate nearest neighbor structures (KD-trees, ball trees, or the same approximate search techniques underlying vector databases) rather than brute-force search.
