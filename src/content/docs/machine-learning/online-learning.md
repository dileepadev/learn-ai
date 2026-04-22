---
title: Online Learning
description: Explore online and incremental learning — algorithms that update models continuously from data streams without storing the full dataset, covering stochastic gradient descent, concept drift detection, and real-world streaming ML systems.
---

**Online learning** is a machine learning paradigm in which a model is trained **incrementally** — updating its parameters one example (or one small batch) at a time as data arrives, rather than training on a fixed dataset in multiple passes. The model continuously incorporates new information without requiring retraining from scratch or access to previously seen data.

Online learning is the natural choice for systems where data arrives as a continuous stream, where the data distribution changes over time, where storage of the full dataset is infeasible, or where fast adaptation to new information is required.

## Online Learning vs. Batch Learning

In **batch learning** (the standard ML workflow):

1. Collect a complete dataset.
2. Train a model on the entire dataset.
3. Deploy the model.
4. Periodically retrain on new data (requires re-running steps 1–3).

In **online learning**:

1. Deploy an initial model (possibly randomly initialized or pretrained).
2. For each incoming example: update the model → make a prediction → receive feedback.
3. The model evolves continuously as data arrives.

Key differences:

| Dimension | Batch Learning | Online Learning |
|-----------|----------------|-----------------|
| Data availability | Full dataset required | One example at a time |
| Training passes | Multiple epochs | One pass (usually) |
| Memory | Full dataset in memory | Bounded memory |
| Adaptation speed | Slow (periodic retraining) | Immediate |
| Stability | Stable | Potentially unstable with drift |

## Stochastic Gradient Descent as Online Learning

**Stochastic gradient descent (SGD)** is the foundational online learning algorithm. For each incoming example $(x_t, y_t)$:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t; x_t, y_t)$$

The model parameters $\theta$ are updated using the gradient of the loss on a single example, rather than the full dataset. This makes SGD inherently online — it can be applied to streaming data with a bounded computational cost per update.

**Mini-batch SGD** processes small batches of $b$ examples at each step, balancing the low-variance updates of batch gradient descent with the speed and streaming compatibility of true online SGD.

## The Prediction-Then-Learn Protocol

Online learning theory studies algorithms in an **adversarial protocol**:

1. At time $t$, the learner observes example $x_t$.
2. The learner predicts $\hat{y}_t$.
3. The true label $y_t$ is revealed.
4. The learner suffers loss $\ell(\hat{y}_t, y_t)$ and updates its model.

The performance measure is **regret** — the difference between the algorithm's cumulative loss and the cumulative loss of the best fixed model in hindsight:

$$R_T = \sum_{t=1}^{T} \ell(\hat{y}_t, y_t) - \min_{\theta^*} \sum_{t=1}^{T} \ell(f_{\theta^*}(x_t), y_t)$$

A learning algorithm is considered good if its regret grows sub-linearly with $T$ (i.e., $R_T = o(T)$), meaning its per-step loss converges to that of the optimal fixed model.

## Key Online Learning Algorithms

### Perceptron

The **Perceptron** (Rosenblatt, 1958) is the original online classification algorithm:

- If the prediction is correct, no update.
- If the prediction is wrong: $\theta \leftarrow \theta + y_t x_t$ (add the misclassified example to the weight vector).

The Perceptron converges to a separable solution in a finite number of mistakes when the data is linearly separable. The **mistake bound** depends on the margin of separation.

### Winnow

**Winnow** is an online algorithm designed for problems with many features where only a few are relevant (sparse relevant features). It uses multiplicative updates:

- Correct prediction: no update.
- Wrong prediction: multiply weights of features present in the example by $2$ (positive class) or $\frac{1}{2}$ (negative class).

Winnow's mistake bound depends logarithmically on the number of features — much better than Perceptron when the number of relevant features is small relative to the total.

### Online Gradient Descent (OGD)

**OGD** applies gradient descent updates in the online protocol, projecting back onto a convex feasible set $\mathcal{K}$ after each update:

$$\theta_{t+1} = \Pi_{\mathcal{K}}(\theta_t - \eta_t g_t)$$

where $g_t = \nabla \ell(\theta_t; x_t, y_t)$ is the subgradient of the loss. With appropriately decaying step sizes $\eta_t = O(1/\sqrt{t})$, OGD achieves $O(\sqrt{T})$ regret — optimal for convex losses.

### Follow the Regularized Leader (FTRL)

**FTRL** at each step chooses parameters that would have minimized the sum of all past losses plus a regularization term:

$$\theta_{t+1} = \underset{\theta}{\arg\min} \left[ \sum_{s=1}^{t} \ell_s(\theta) + R(\theta) \right]$$

In practice, this reduces to a closed-form update. FTRL with adaptive learning rates (FTRL-Proximal, used in Google's large-scale ad click prediction) achieves sparse solutions (via L1 regularization) suitable for models with billions of features.

### Adaptive Methods: AdaGrad, Adam

**AdaGrad** adapts the learning rate per parameter based on the history of gradients — parameters that receive large gradients get smaller learning rates (dampening rapid oscillation) while rarely updated parameters get larger learning rates:

$$\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} g_{t,i}$$

where $G_{t,ii} = \sum_{s=1}^{t} g_{s,i}^2$ is the sum of squared gradients for parameter $i$.

AdaGrad is particularly effective for sparse online learning problems (text, click prediction) where some features are very rare.

**Adam** extends AdaGrad with momentum and bias correction and is the dominant optimizer for deep neural networks in online settings.

## Concept Drift

A central challenge in online learning is **concept drift** — the phenomenon where the statistical properties of the data distribution change over time. A model trained on historical data may become outdated as the world changes.

### Types of Concept Drift

**Sudden drift**: The distribution changes abruptly at a single time point. Examples: product launches, market events, seasonal transitions.

**Gradual drift**: The old concept transitions slowly to a new one over an extended period.

**Incremental drift**: Small, cumulative changes that accumulate over time (e.g., slow changes in user preferences).

**Recurring drift**: Old concepts re-emerge periodically (e.g., annual seasonal patterns, weekly cycles).

### Detecting Concept Drift

**CUSUM (Cumulative Sum)**: Monitors the cumulative sum of deviations from the expected error rate. A detection is triggered when the CUSUM statistic exceeds a threshold.

**Page-Hinkley Test**: Similar to CUSUM — detects when the mean of an observed variable changes significantly.

**DDM (Drift Detection Method)**: Monitors the error rate $p_t$ and its standard deviation $s_t = \sqrt{p_t(1 - p_t)/n}$. A warning is issued when $p_t + s_t > p_{min} + 2 s_{min}$; a drift is detected when $p_t + s_t > p_{min} + 3 s_{min}$.

**ADWIN (Adaptive Windowing)**: Maintains a variable-length sliding window over the data stream. A drift is detected when the mean of any two subwindows differs by more than a statistical threshold, triggering window shrinkage.

### Responding to Concept Drift

- **Forgetting mechanisms**: Use exponentially weighted moving averages or sliding windows so older examples contribute less to the current model.
- **Ensemble with drift detection**: Maintain a pool of models trained on different time windows; activate the model most appropriate to current conditions.
- **Reset and retrain**: When severe drift is detected, discard the current model and retrain from recent data.
- **Fine-tuning**: When mild drift is detected, continue training on recent data with a reduced learning rate.

## Online Learning for Large-Scale Systems

### Click-Through Rate Prediction

Online advertising click prediction is perhaps the most mature large-scale online learning application. Systems at Google, Facebook/Meta, Alibaba, and others:

- Process billions of impression-click events daily.
- Use FTRL-Proximal or similar algorithms to update feature weights in near-real-time.
- Maintain models with billions to trillions of sparse features.
- Must adapt rapidly to new advertisers, campaigns, and seasonal patterns.

The Hogwild! algorithm enables lock-free parallel asynchronous SGD updates, allowing multiple processors to update a shared model without synchronization — achieving near-linear speedups for sparse models where parameter collisions are rare.

### Recommendation Systems

Streaming recommendation systems use online learning to:

- Incorporate user feedback (clicks, purchases, ratings) within seconds.
- Update user and item embeddings continuously.
- Adapt to new items with no historical interaction data.

**Bandit algorithms** (contextual bandits) naturally integrate exploration (trying new recommendations) with exploitation (recommending known good content) — a better fit for recommendation than pure supervised learning.

### Fraud Detection

Fraud patterns evolve continuously as fraudsters adapt to deployed defenses. Online learning enables fraud models to:

- Update immediately when new fraud patterns are labeled.
- Downweight older transactions that may reflect outdated fraud patterns.
- Detect emerging attacks before a full retraining cycle could respond.

## River: A Python Library for Online Learning

**River** (formerly Creme) is the leading Python library for online machine learning. It provides:

- Online implementations of linear models, decision trees, Naive Bayes, ensembles.
- Drift detection methods (ADWIN, DDM, HDDM).
- Online feature engineering (windowed statistics, count encoders).
- Streaming evaluation (progressive validation).
- Compatible API with scikit-learn.

```python
from river import linear_model, preprocessing, metrics

model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
metric = metrics.Accuracy()

for x, y in dataset:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)
```

The `|` operator chains transformers and models into an online pipeline that processes one example at a time.

## Progressive Validation

Standard train/test splits are not directly applicable to online learning (there is no separate training phase). **Progressive validation** (also called prequential evaluation) evaluates online learning systems by:

1. For each incoming example, make a prediction *before* seeing the label.
2. Record the prediction error.
3. Update the model with the true label.
4. Report the rolling average of prediction errors.

This gives an unbiased estimate of the model's performance as it would be experienced in production — measuring the true online error rather than a held-out batch error.

## Online Learning vs. Continual Learning

These terms are often conflated but refer to related but distinct problems:

**Online learning**: Concerned with computational and statistical efficiency for streaming data. The focus is on algorithms that process one example at a time with bounded memory and sub-linear regret. May involve stationary or non-stationary distributions.

**Continual learning**: Specifically concerned with learning a **sequence of distinct tasks** while retaining performance on previous tasks — overcoming **catastrophic forgetting** in neural networks. The focus is on preventing the new task from overwriting knowledge of previous tasks.

Online learning algorithms are often applied to the continual learning setting, but the core research concerns are different: continual learning focuses on forgetting; online learning focuses on efficiency and adaptation.
