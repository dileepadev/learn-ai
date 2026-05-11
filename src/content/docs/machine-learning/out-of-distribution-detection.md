---
title: Out-of-Distribution Detection
description: Learn how to detect out-of-distribution (OOD) inputs in machine learning systems — covering energy-based scoring, the Mahalanobis distance method, ODIN post-hoc detection, ReAct feature clipping, deep ensembles for uncertainty, and evaluation using AUROC on standard OOD benchmarks.
---

Machine learning models are trained on a specific data distribution and are expected to perform well on test examples drawn from that same distribution. In deployment, however, models inevitably encounter **out-of-distribution (OOD)** inputs — samples that differ significantly from the training distribution. A skin lesion classifier trained on clinical photographs should recognize when presented with a smartphone photo taken in poor lighting; a fraud detection model should flag transaction patterns that differ from its training set. Without OOD detection, models silently produce confident but meaningless predictions on inputs they are not equipped to handle.

**OOD detection** is the problem of determining whether a new input $x$ belongs to the training distribution $p_\text{in}(x)$ or comes from an unknown distribution $p_\text{out}(x)$, using only the trained model's internal representations — without access to out-of-distribution examples during training.

## Problem Formulation

An OOD detector outputs a **confidence score** $s(x)$ for each input:

- High $s(x)$: the model is confident $x$ is in-distribution (ID) → proceed with prediction.
- Low $s(x)$: the model suspects $x$ is OOD → flag for human review or refuse to predict.

The detector is evaluated by its ability to separate ID and OOD test sets using $s(x)$ as a discriminator. Standard metrics:

- **AUROC** (Area Under ROC Curve): probability that a random ID example has higher score than a random OOD example. Perfect detector: 1.0; random detector: 0.5.
- **FPR95** (False Positive Rate at 95% True Positive Rate): the fraction of OOD examples incorrectly classified as ID when 95% of ID examples are correctly classified. Lower is better.

## Baseline: Maximum Softmax Probability

The earliest OOD detection approach (Hendrycks and Gimpel, 2017) uses the maximum of the softmax output as the confidence score:

$$s_\text{MSP}(x) = \max_c \text{softmax}(\text{logit}_c(x))$$

The intuition: a model confident about the class of an input (high softmax probability) is likely on ID data; uncertain predictions (low softmax) suggest OOD. Despite its simplicity, MSP establishes a strong baseline that many sophisticated methods only marginally improve.

**Limitation**: modern neural networks are overconfident on OOD data. ReLU networks produce arbitrarily high logits for inputs far from the training data in the direction of the weight vectors, and the softmax converts these to near-1.0 probabilities.

## ODIN: Out-of-Distribution Detector for Neural Networks

**ODIN** (Liang et al., 2018) applies two modifications to improve OOD detection without retraining:

### Temperature Scaling

Dividing logits by a temperature $T > 1$ before softmax smoothes the distribution and improves the separation between ID and OOD confidence scores:

$$s_\text{ODIN}(x) = \max_c \text{softmax}(\text{logit}_c(x) / T)$$

Optimal $T$ is found by validation on a small set of known OOD examples.

### Input Preprocessing

Add a small perturbation to $x$ in the direction that **increases** the model's confidence:

$$\tilde{x} = x + \varepsilon \cdot \text{sign}\!\left(\nabla_x \max_c \text{logit}_c(x)\right)$$

This perturbation is similar to the FGSM adversarial example construction but applied to maximize (not confuse) prediction confidence. ID examples respond strongly to this perturbation (confidence increases significantly); OOD examples respond less (confidence was already inflated by geometry rather than learned features).

## Energy-Based OOD Detection

**Energy scores** (Liu et al., 2020) replace the softmax with a more principled measure derived from the energy-based model framework:

$$E(x) = -T \cdot \log \sum_c \exp(\text{logit}_c(x) / T)$$

The energy $E(x)$ is low for in-distribution inputs (where the model assigns high unnormalized probability to some class) and high for OOD inputs (where logits are uniformly low or uninformative).

The OOD score uses negative energy:

$$s_\text{energy}(x) = -E(x) = T \cdot \log \sum_c \exp(\text{logit}_c(x) / T)$$

Energy scoring consistently outperforms MSP without any architectural change, because the LogSumExp aggregation over all logits (the log-sum-exp is the soft-maximum) captures the overall "certainty" of the model's output better than the maximum alone.

### Energy-Bounded Training

Fine-tuning the model with an energy margin loss further improves detection by explicitly pushing OOD examples (approximated from auxiliary outlier datasets) toward higher energy:

$$\mathcal{L}_\text{energy} = \mathbb{E}_{x_\text{in}}[\max(0, E(x_\text{in}) - m_\text{in})^2] + \mathbb{E}_{x_\text{out}}[\max(0, m_\text{out} - E(x_\text{out}))^2]$$

where $m_\text{in} < m_\text{out}$ are margin hyperparameters. This trains ID examples to have energy below $m_\text{in}$ and OOD examples to have energy above $m_\text{out}$.

## Mahalanobis Distance Method

The **Mahalanobis distance** method (Lee et al., 2018) uses the feature space geometry of a trained classifier for OOD detection:

1. **Fit class-conditional Gaussians**: for each class $c$, compute the mean $\mu_c$ and shared covariance $\Sigma$ of penultimate-layer features over the training set.
1. **Compute Mahalanobis distance**: for a test input $x$, the OOD score is the minimum Mahalanobis distance to any class mean:

$$s_M(x) = \max_c \left[ -(h(x) - \mu_c)^\top \Sigma^{-1} (h(x) - \mu_c) \right]$$

where $h(x)$ is the feature vector (negative distance, so higher is more ID-like).

The Mahalanobis distance accounts for feature correlation via the inverse covariance matrix, making it more robust than Euclidean distance to anisotropic feature distributions. It performs strongly on near-OOD detection (where OOD data is semantically similar to ID data but from different classes) and benefits from **feature ensemble** — computing the score from multiple intermediate layers and combining them.

## ReAct: Rectified Activations

**ReAct** (Sun et al., 2021) addresses a specific cause of overconfidence on OOD data: **unit activation abnormalities**. Neural networks trained on ID data have bounded activation magnitudes for ID inputs, but OOD inputs can produce extreme activation values that drive high confidence.

ReAct truncates activations at a threshold $c$ before the classification head:

$$h_\text{react}(x) = \min(h(x), c)$$

where $c$ is chosen as a high percentile (e.g., 90th percentile) of ID activation values. The truncation prevents OOD-induced activation extremes from producing inflated confidence, while preserving ID predictions (which rarely have activations near the threshold).

ReAct is a simple post-hoc modification requiring no retraining and achieves competitive AUROC with more complex methods.

## Deep Ensembles

**Deep ensembles** (Lakshminarayanan et al., 2017) train $M$ models with different random initializations and use disagreement across the ensemble as an uncertainty signal:

$$s_\text{ensemble}(x) = 1 - \text{Var}_m[\hat{p}_m(y^* \mid x)]$$

where $\hat{p}_m$ is the predictive distribution from model $m$. Ensemble members agree strongly on ID inputs (reducing variance) and disagree on OOD inputs (increasing variance), providing a reliable OOD signal.

Ensembles are the most reliable OOD detection method in practice but require 5-10× the inference cost. For production systems where latency matters, single-model methods like energy scoring or ReAct are preferred.

### Monte Carlo Dropout

A cheaper approximation: apply dropout at inference time and sample $M$ stochastic forward passes. The variance across passes approximates Bayesian predictive uncertainty. MC Dropout is faster than a full ensemble but less effective at OOD detection — stochastic dropout introduces less diversity than independent training.

## KNN-Based Detection

**KNN OOD detection** (Sun et al., 2022) uses the distance to the $k$-th nearest neighbor in the feature space of the training set:

$$s_\text{KNN}(x) = -d_k(h(x), \mathcal{Z}_\text{train})$$

where $d_k$ is the Euclidean distance to the $k$-th nearest training feature. ID inputs have small nearest-neighbor distances (they are similar to training examples); OOD inputs have large distances.

KNN detection achieves strong performance without distribution assumptions, but requires storing all training features (memory cost) and efficient approximate nearest neighbor search (e.g., FAISS) for low-latency serving.

## Evaluation Benchmarks

Standard OOD benchmarks for image classification:

| In-Distribution | Out-of-Distribution Sets |
| --- | --- |
| CIFAR-10 | CIFAR-100, SVHN, Textures, Places365 |
| CIFAR-100 | CIFAR-10, SVHN, Textures, Places365 |
| ImageNet | iNaturalist, SUN, Places, Textures |

**Near-OOD** benchmarks (OOD examples semantically similar to ID):

- CIFAR-10 vs. CIFAR-100: both are natural image datasets; distinguishing requires fine-grained feature understanding.
- ImageNet vs. ImageNet-O: OOD examples that strongly activate ImageNet classifiers despite being from unseen classes.

Methods that perform well on far-OOD (textures vs. photographs) often fail on near-OOD, making near-OOD benchmarks the more stringent evaluation.

## OOD Detection vs. Anomaly Detection

OOD detection and anomaly detection are related but distinct:

- **OOD detection**: binary decision — is this input from $p_\text{in}$ or not? Applied at inference time to a trained classifier.
- **Anomaly detection**: finds unusual examples within a single distribution (e.g., fraudulent transactions among normal ones). Often unsupervised, trained without normal class labels.

One-class classification methods (deep SVDD, PatchCore for visual anomaly detection) address anomaly detection; OOD detection methods address the deployment safety problem for supervised classifiers.

## Summary

Out-of-distribution detection enables models to recognize when they are operating outside their competence — a critical requirement for safe deployment in high-stakes applications. Maximum softmax probability provides a baseline; energy scoring improves on it by using the log-sum-exp of all logits; Mahalanobis distance leverages feature space geometry; ReAct eliminates OOD-induced activation extremes with a simple clipping heuristic; deep ensembles provide the most reliable uncertainty estimates at higher computational cost. Near-OOD benchmarks on semantically similar datasets remain challenging for all methods, motivating continued research into feature-level representations that more faithfully encode distributional membership.
