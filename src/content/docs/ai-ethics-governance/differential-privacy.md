---
title: Differential Privacy
description: Protecting individual privacy in machine learning — techniques to train models on sensitive data while providing formal privacy guarantees.
---

**Differential Privacy (DP)** is a mathematical framework for ensuring that machine learning models preserve individual privacy while still learning from sensitive data. The key idea: adding controlled noise during training such that the model's output reveals minimal information about any individual's data, while remaining accurate for general patterns.

This is critical for training on healthcare records, financial data, demographic information, or any sensitive personal data — enabling useful AI without violating privacy.

## The Privacy Problem

Traditional machine learning reveals private information in two ways:

1. **Model overfitting**: A model can memorize training data. Given a data point, we can ask: "Was this person's data in your training set?" — a membership inference attack. Some models even allow extracting specific training records.

2. **Model inversion**: Given a model's predictions, an attacker can infer properties of training individuals. For example, if a model predicts income from demographic data, an attacker might infer the income of someone similar to a training individual.

Differential privacy addresses these risks rigorously.

## Formal Definition

**Differential Privacy (ε, δ)**: An algorithm is (ε, δ)-differentially private if, for any two adjacent datasets $D$ and $D'$ (differing by one record), and any possible output $O$:

$$P(\text{Algorithm}(D) = O) \leq e^\epsilon \cdot P(\text{Algorithm}(D') = O) + \delta$$

Intuition:
- Changing one person's data changes the probability of any output by at most a factor of $e^\epsilon$ (plus a small $\delta$ term).
- **Lower ε**: Stronger privacy (more noise needed).
- **Higher ε**: Weaker privacy (less noise, more accuracy).
- **Lower δ**: Lower probability of failure; typically set to be smaller than $1/n$ where $n$ is dataset size.

A typical setting: ε = 1, δ = 10^-5 (strong privacy with manageable accuracy loss).

## Mechanisms for Differential Privacy

### Laplace Mechanism

Add noise drawn from a Laplace distribution:

$$\text{Output} = \text{Query}(D) + \text{Lap}(0, b)$$

where $b = \Delta Q / \epsilon$ and $\Delta Q$ is the **sensitivity** of the query (how much the output changes if one data point changes).

**Example**: To release the count of individuals with a disease (sensitivity = 1), add Laplace noise with scale $1/\epsilon$.

### Gaussian Mechanism

For multiple queries, add noise from a Gaussian distribution:

$$\text{Output} = \text{Query}(D) + \mathcal{N}(0, \sigma^2)$$

where $\sigma = \frac{\sqrt{2} \Delta Q}{\epsilon}$ (approximately).

The Gaussian mechanism is better for sequential queries due to composition properties.

### Exponential Mechanism

For non-numeric outputs (e.g., selecting a category), sample outputs proportionally to $\exp(\epsilon \cdot \text{score} / 2\Delta)$, where score is higher for more desirable outputs.

This enables differentially private recommendations: return an item likely to be favored, without revealing the exact score.

## Differentially Private Machine Learning

### DP-SGD (Differentially Private Stochastic Gradient Descent)

**DP-SGD** (Abadi et al., 2016) makes standard SGD differentially private:

1. **Gradient clipping**: Clip each individual's gradient to a maximum norm $C$ — reduces sensitivity.
2. **Add noise**: After clipping, add Gaussian noise to the gradient.
3. **Aggregate**: Sum clipped + noisy gradients.

$$\theta_{t+1} = \theta_t - \eta \left( \text{sum of clipped gradients} + \mathcal{N}(0, \sigma^2 I) \right)$$

This ensures the update reveals minimal information about any individual's contribution.

### Privacy Budget Accounting

Each training step consumes privacy budget. After $T$ steps with noise scale $\sigma$:
- Total privacy: approximately $(\frac{T}{\sigma^2})^{1/2}$ for ε (simplified).
- Smaller $\sigma$ or more steps → tighter privacy guarantee (smaller ε or δ).

Practitioners track privacy loss and stop training once a privacy target is reached.

### Amplification by Sampling

If mini-batches are sampled randomly (not deterministic), privacy is amplified:
- Sampling probability $\gamma = B / N$ (batch size / dataset size).
- Equivalent ε is reduced by a factor related to $\gamma$.

This allows better privacy-accuracy tradeoffs in practice.

## Privacy-Accuracy Tradeoff

Adding noise for privacy hurts accuracy:

- **High privacy (low ε)**: More noise, significant accuracy loss. Useful for very sensitive settings.
- **Moderate privacy (ε ≈ 1)**: Manageable accuracy loss, strong privacy guarantees.
- **Weak privacy (high ε > 10)**: Minimal accuracy impact, but privacy is weak.

Practical DP-SGD training:
- **Non-private accuracy**: 95% on CIFAR-10.
- **ε = 8, δ = 10^-5**: ~92% accuracy (3% loss).
- **ε = 1, δ = 10^-5**: ~75% accuracy (significant loss).

## Applications

### Medical Data

Train models on patient records (diagnoses, treatments, genetic data) while ensuring no individual patient is identifiable. DP enables collaborative learning across hospitals without sharing raw data.

### Census Data

Release aggregate statistics (e.g., population counts) while protecting individual privacy. The 2020 U.S. Census used differential privacy.

### Federated Learning

Train on decentralized data (e.g., smartphones). Each device sends only gradients (via DP-SGD), never raw data. The server aggregates differentially private gradients.

### Recommendation Systems

Train recommendation models on user behavior while ensuring users' individual preferences are not memorized.

## Practical Challenges

### Accuracy Loss

Differential privacy reduces model accuracy, especially for complex models and strict privacy targets. Tradeoffs must be carefully negotiated between privacy and utility.

### Tuning Hyperparameters

DP-SGD introduces additional hyperparameters (gradient clipping norm, noise scale) that interact with standard learning rate and batch size. Tuning is complex.

### Composition and Multiple Queries

If a dataset is used for multiple analyses, privacy budgets combine (a.k.a. composition). Repeated use of the same data quickly exhausts privacy budgets. Mechanisms like **zero-knowledge proofs** or **composition accounting** help but add complexity.

### Scalability

Gradient clipping and noise addition add computational overhead. For very large models or datasets, DP-SGD training can be 2–10x slower.

## Current Research Directions

- **Federated learning + DP**: Combine distributed training with privacy for decentralized systems (e.g., keyboard prediction on mobile phones).
- **Relaxed notions**: Concepts like **local differential privacy** (where each user adds noise before sending data) and **approximate differential privacy** balance privacy and utility.
- **Membership inference attacks**: Formal evaluation of privacy achieved — testing whether trained models leak membership information.
- **Efficient DP**: Reducing computational overhead and accuracy loss through better mechanisms and algorithms.

Differential privacy is moving from theory to practice: large-scale systems (Google, Apple, U.S. Census) now deploy DP in production, making privacy a first-class concern in machine learning.
