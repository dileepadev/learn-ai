---
title: Isolation Forest - Anomaly Detection by Random Partitioning
description: Learn how Isolation Forest detects anomalies by measuring how quickly random splits isolate a point, with scoring details and practical usage in scikit-learn.
---

Isolation Forest is an unsupervised anomaly detection algorithm built on a simple observation: anomalies are few and different, so they are easier to isolate from the rest of the data than normal points.

## Core Idea

Build random binary trees by repeatedly choosing a random feature and a random split value between that feature's min and max. A point in a dense region needs many splits to be separated from its neighbors, while an outlier is isolated after only a few.

```text
Normal point:  deep in the tree   (long path)
Anomaly:       near the root      (short path)
```

## Anomaly Score

The score uses the average path length E[h(x)] across trees, normalized by c(n), the expected path length of an unsuccessful search in a binary search tree of n samples:

```text
s(x, n) = 2 ^ ( - E[h(x)] / c(n) )
```

Scores close to 1 indicate anomalies; scores well below 0.5 indicate normal points.

## Why It Is Efficient

Each tree is built on a small random subsample (256 points is a common default), which reduces swamping and masking effects and keeps training near linear in the number of samples. No distance calculations are required.

## Usage

```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=0)
clf.fit(X_train)

labels = clf.predict(X_test)         # 1 = normal, -1 = anomaly
scores = clf.score_samples(X_test)   # lower = more anomalous
```

## Practical Tips

- `contamination` sets the threshold; if unknown, threshold on `score_samples` using a validation set
- Works well in moderate-to-high dimensions but struggles when anomalies hide in feature subspaces among many irrelevant features
- Axis-parallel splits can create artifacts; Extended Isolation Forest uses random hyperplanes to reduce this
- Scale is not required, but categorical features need encoding
