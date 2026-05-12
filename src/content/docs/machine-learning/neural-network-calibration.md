---
title: Neural Network Calibration
description: Learn how to calibrate neural network confidence scores — covering the Expected Calibration Error metric, reliability diagrams, temperature scaling, Platt scaling, histogram binning, isotonic regression, and why modern deep networks are systematically overconfident.
---

A well-calibrated model produces confidence scores that reflect the true likelihood of being correct. When a calibrated classifier says it is 80% confident, it should be correct 80% of the time on inputs where it reports 80% confidence. In practice, modern deep neural networks are systematically **overconfident** — a ResNet trained on CIFAR-100 that outputs 90% confidence is often correct far less than 90% of the time. Calibration methods fix this discrepancy, making confidence scores reliable for downstream decision-making.

## Why Calibration Matters

Calibrated confidence estimates are essential in:

- **Medical diagnosis**: a diagnostic AI that outputs 95% confidence when it is actually 60% accurate can lead to missed diagnoses or inappropriate treatment decisions.
- **Safety-critical systems**: autonomous vehicles, industrial control, and financial systems require reliable uncertainty estimates to trigger human oversight.
- **Downstream pipelines**: multi-stage AI systems that use one model's output as input to another must pass calibrated confidence scores to avoid error propagation.
- **Selective prediction**: systems that abstain from prediction when uncertainty is too high only function correctly if confidence scores accurately reflect uncertainty.

## The Overconfidence Problem

**Guo et al. (2017)** demonstrated a striking trend: modern deep neural networks trained with standard cross-entropy loss and strong regularization (batch normalization, weight decay, dropout) are significantly more overconfident than older, shallower networks. ResNet-110 trained on CIFAR-100 has a gap between accuracy and confidence of ~6 percentage points — it reports 80% average confidence when its actual accuracy is ~74%.

The causes are interconnected:

- **Cross-entropy loss** does not directly penalize miscalibrated confidence — only misclassification. A model can achieve low cross-entropy by predicting high-probability correct classes without being well-calibrated.
- **Batch normalization** shifts the effective model capacity and changes how logit magnitudes scale with depth, systematically increasing output confidence.
- **Network depth**: deeper networks consistently exhibit worse calibration than shallower ones at equivalent accuracy, suggesting that representational power amplifies overconfidence rather than mitigating it.

## Measuring Calibration

### Expected Calibration Error (ECE)

ECE measures the average gap between confidence and accuracy, estimated by binning predictions:

1. Divide predictions into $M$ equally spaced confidence bins $B_1, \ldots, B_M$ (e.g., [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]).
1. For each bin $B_m$, compute:
   - $\text{acc}(B_m)$: fraction of correctly classified samples in the bin.
   - $\text{conf}(B_m)$: mean predicted confidence in the bin.
1. Compute the weighted average gap:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

ECE ranges from 0 (perfectly calibrated) to 1 (maximally miscalibrated). Values below 0.02 are considered well-calibrated in practice.

### Maximum Calibration Error (MCE)

MCE is the worst-case gap across all bins:

$$\text{MCE} = \max_{m=1,\ldots,M} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

MCE is more relevant for high-stakes applications where even occasional large miscalibration is unacceptable.

### Reliability Diagrams

A **reliability diagram** plots $\text{acc}(B_m)$ vs. $\text{conf}(B_m)$ for each bin. A perfectly calibrated model produces a diagonal line from (0, 0) to (1, 1). Overconfident models show accuracy below the diagonal at high confidence levels; underconfident models show accuracy above the diagonal.

## Temperature Scaling

**Temperature scaling** is the simplest and most effective post-hoc calibration method (Guo et al., 2017). A single scalar temperature $T > 0$ is applied to the logits before the softmax:

$$\hat{p}_c = \frac{\exp(z_c / T)}{\sum_{c'} \exp(z_{c'} / T)}$$

- $T = 1$: original model output (no calibration).
- $T > 1$: softer distribution (lower confidence, flatter probabilities) — corrects overconfidence.
- $T < 1$: sharper distribution (higher confidence) — corrects underconfidence.

$T$ is optimized on a held-out validation set by minimizing the **negative log-likelihood (NLL)** of the calibrated predictions:

$$T^* = \arg\min_T -\sum_i \log \hat{p}_{y_i}(T)$$

This is a one-dimensional optimization (trivially solved by line search or gradient descent) that does not change the model's accuracy — it only rescales the confidence scores. Despite its simplicity, temperature scaling consistently achieves ECE close to 0.01 on ImageNet and CIFAR benchmarks, outperforming more complex methods.

### Why Temperature Scaling Works

Temperature scaling effectively changes the **entropy** of the model's output distribution. Overconfident models produce low-entropy outputs (probability mass concentrated on one class); $T > 1$ increases entropy, spreading probability more evenly. The NLL objective on the validation set finds the entropy level that best matches actual accuracy — the fundamental calibration condition.

## Platt Scaling

**Platt scaling** (Platt, 1999) was originally developed for SVMs and applies a logistic regression on top of the model's raw scores. For binary classification:

$$\hat{p}(y=1 \mid x) = \sigma(a \cdot f(x) + b)$$

where $f(x)$ is the model's score and $a, b$ are learned on a validation set. For multi-class classification, Platt scaling generalizes to a softmax with per-class learned parameters. Platt scaling is more flexible than temperature scaling (2 parameters per class instead of 1 global parameter) but can overfit on small validation sets.

## Histogram Binning

**Histogram binning** is a non-parametric calibration method:

1. Divide predictions into $M$ bins based on predicted confidence.
1. For each bin, replace all confidence values with the empirical accuracy in that bin.

This is equivalent to a piecewise-constant mapping from predicted confidence to calibrated confidence. Histogram binning is flexible (no parametric assumption) but requires sufficient validation data per bin. With $M = 15$ bins and a validation set of 1,000 examples, each bin has only ~67 examples — leading to noisy estimates.

## Isotonic Regression

**Isotonic regression** is a non-parametric method that fits a piecewise-linear monotone function mapping raw confidences to calibrated ones. The monotonicity constraint ensures that higher raw confidence always maps to higher calibrated confidence — a necessary property for a valid calibration function.

Isotonic regression is more flexible than histogram binning (variable bin sizes, continuous function) and typically outperforms histogram binning with sufficient data.

## Multi-Class Calibration Methods

For multi-class problems, single-probability calibration (calibrating only the top-class probability) is insufficient — the full probability vector should be calibrated. Methods include:

### Top-Label Calibration

Calibrate only the confidence assigned to the predicted class $\hat{y} = \arg\max_c \hat{p}_c$. Temperature scaling achieves this efficiently and is the standard approach.

### Dirichlet Calibration

Fit a Dirichlet distribution to the softmax outputs on the validation set. The Dirichlet calibration applies a linear transformation in log-space to the logits:

$$\hat{p} = \text{softmax}(W \log \hat{p} + b)$$

This generalizes temperature scaling (where $W = \frac{1}{T} I$) and can correct off-diagonal miscalibration (where the model confuses specific pairs of classes systematically).

## Calibration During Training

Post-hoc calibration requires a held-out validation set and re-calibration when the model is updated. **Training-time calibration** methods integrate calibration directly into the learning objective:

### Label Smoothing

Replace one-hot targets $y_i \in \{0, 1\}^C$ with soft targets:

$$\tilde{y}_c = (1 - \varepsilon) \cdot y_c + \frac{\varepsilon}{C}$$

Label smoothing prevents the model from driving softmax probabilities to 1 for the correct class, improving calibration. With $\varepsilon = 0.1$, the target for the correct class is 0.9 and for incorrect classes is 0.1/C, reducing the effective confidence the model is trained to produce.

### Focal Loss

The focal loss (Lin et al., 2017), originally proposed for object detection, down-weights easy examples:

$$\mathcal{L}_\text{focal} = -\sum_i (1 - \hat{p}_{y_i})^\gamma \log \hat{p}_{y_i}$$

By reducing the loss contribution of high-confidence correct predictions, focal loss discourages the model from becoming overconfident, implicitly improving calibration.

## Calibration in Large Language Models

LLM calibration differs from classification calibration in important ways:

- **Token-level calibration**: language model perplexity measures token-level calibration, but downstream task confidence (e.g., multiple-choice answers) requires task-specific evaluation.
- **Verbalized confidence**: LLMs can be prompted to verbalize their uncertainty ("I am 80% confident that..."). These verbalized confidences are poorly calibrated — LLMs tend to state high confidence regardless of actual accuracy.
- **Selective prediction with LLMs**: calibrated LLMs should be able to abstain from questions outside their knowledge ("I don't know") at a rate matching their actual error rate — a research-active capability.

Temperature scaling applied to LLM logits improves calibration for multiple-choice tasks; for open-ended generation, Monte Carlo sampling (generating multiple answers and measuring consistency) provides an alternative uncertainty estimate.

## Summary

Neural network calibration ensures that confidence scores reflect true accuracy rather than overconfident artifacts of cross-entropy training. ECE and reliability diagrams measure miscalibration. Temperature scaling — optimizing a single scalar on a validation set — is the most practical calibration method, consistently reducing ECE below 0.01 with negligible computational cost. More complex methods (Platt scaling, isotonic regression, Dirichlet calibration) offer greater flexibility at the cost of requiring more validation data. Training-time methods (label smoothing, focal loss) reduce the initial calibration gap. Reliable confidence estimates are foundational for high-stakes deployment, selective prediction, and multi-stage AI pipelines.
