---
title: Bayesian Deep Learning
description: Understand how Bayesian methods bring principled uncertainty quantification to deep neural networks — covering approximate inference techniques like variational inference, Monte Carlo Dropout, and deep ensembles, and why calibrated uncertainty matters in real-world AI systems.
---

Bayesian Deep Learning (BDL) marries the representational power of deep neural networks with the principled uncertainty quantification of Bayesian inference. Instead of learning a single set of weights, BDL treats model parameters as **random variables** with probability distributions — allowing the model to express not just *what* it predicts but *how confident* it is.

## Why Uncertainty Matters

A standard neural network produces a softmax output for classification — often interpreted as a probability. But this is a **calibration confidence**, not a true uncertainty measure. A network trained on cats and dogs may confidently classify a zebra as a dog simply because "zebra" is outside its training distribution.

Two types of uncertainty are critical:

- **Aleatoric uncertainty:** Irreducible noise in the data (e.g., label noise, sensor measurement error). No amount of additional data reduces this.
- **Epistemic uncertainty:** Uncertainty due to lack of knowledge — insufficient or ambiguous training data. This can be reduced by collecting more data or better models.

Bayesian methods distinguish and quantify both types. This is essential in:
- **Safety-critical systems:** Medical diagnosis, autonomous driving — the model should be uncertain near its decision boundary
- **Active learning:** Query the most uncertain examples for labeling
- **Out-of-distribution detection:** Flag inputs the model has never seen similar to
- **Scientific discovery:** Report prediction intervals, not point estimates

## The Bayesian Framework

In Bayesian inference over model parameters $\theta$:

$$P(\theta | \mathcal{D}) = \frac{P(\mathcal{D} | \theta) P(\theta)}{P(\mathcal{D})}$$

- $P(\theta)$ is the **prior** — beliefs about parameters before seeing data
- $P(\mathcal{D} | \theta)$ is the **likelihood** — how well parameters explain the data
- $P(\theta | \mathcal{D})$ is the **posterior** — updated beliefs after seeing data

For predictions on new input $x^*$:

$$P(y^* | x^*, \mathcal{D}) = \int P(y^* | x^*, \theta) P(\theta | \mathcal{D})\, d\theta$$

This **posterior predictive distribution** is a probability distribution over possible outputs — the full characterization of what the model knows and doesn't know.

**The challenge:** For deep networks with millions of parameters, computing the exact posterior $P(\theta | \mathcal{D})$ is computationally intractable. All practical BDL methods are **approximations**.

## Monte Carlo Dropout (MC Dropout)

**MC Dropout** (Gal & Ghahramani, 2016) showed that a neural network trained with dropout and evaluated with dropout **active at test time** is equivalent to approximate Bayesian inference in a deep Gaussian process.

**Implementation:**

```python
import torch

class BayesianMLP(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(512, 256)
        self.drop = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))

model.train()  # Keep dropout active at test time!

# Monte Carlo sampling
preds = torch.stack([model(x) for _ in range(50)], dim=0)
mean = preds.mean(0)       # Predictive mean
variance = preds.var(0)    # Predictive uncertainty
```

**Pros:** Zero architectural changes to existing models; minimal code change.
**Cons:** Requires many forward passes (30–100) for reliable uncertainty; dropout rate is a hyperparameter that must be tuned for calibration.

## Deep Ensembles

**Deep Ensembles** (Lakshminarayanan et al., 2017) train $M$ independent networks from different random initializations and aggregate their predictions:

$$\bar{p}(y | x) = \frac{1}{M} \sum_{m=1}^M p_m(y | x)$$

The **variance of ensemble predictions** provides an uncertainty estimate. When networks disagree, the model is uncertain.

**Why it works:** Different random initializations lead to models that occupy different modes of the weight posterior. Their disagreement reflects genuine model uncertainty.

**Empirical results:** Despite not being Bayesian in a strict theoretical sense, deep ensembles consistently achieve the **best calibration** among all practical uncertainty methods. They are the de-facto standard in production safety-critical applications.

**Cost:** $M$ separate forward passes — typically $M = 5$ is a good tradeoff.

## Variational Inference: Bayes by Backprop

Variational Bayes approximates the intractable posterior $P(\theta | \mathcal{D})$ with a tractable family $q_\phi(\theta)$ (typically a diagonal Gaussian) by minimizing the KL divergence:

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(\theta)} [\log P(\mathcal{D} | \theta)]}_{\text{Expected data fit}} - \underbrace{\text{KL}[q_\phi(\theta) \| P(\theta)]}_{\text{Prior regularization}}$$

**Bayes by Backprop** (Blundell et al., 2015) uses the **reparameterization trick** to backpropagate through sampled parameters:

$$\theta = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Each parameter gets a **mean** $\mu$ and **standard deviation** $\sigma$ (doubled parameter count). During forward passes, weights are sampled from the learned distribution.

**Pros:** Principled uncertainty; directly estimates parameter posteriors.
**Cons:** 2× parameters; optimization is harder than standard training; often underperforms deep ensembles in practice.

## Laplace Approximation

The **Laplace approximation** fits a Gaussian centered at the MAP estimate $\theta_\text{MAP}$ with covariance equal to the inverse Hessian of the loss:

$$q(\theta) = \mathcal{N}(\theta_\text{MAP},\, (H + \lambda I)^{-1})$$

The key advantage: **train normally**, then apply the approximation post-hoc. For large models, the full Hessian is intractable but last-layer Laplace (fitting the approximation only to the final layer's weights) is efficient and effective.

**Last-Layer Laplace** (Daxberger et al., 2021) is popular for adding calibrated uncertainty to pre-trained models without retraining.

## Calibration: Are Uncertainty Estimates Trustworthy?

A model is **well-calibrated** if its predicted confidence matches actual accuracy: when it says "90% confident," it should be right 90% of the time.

**Evaluation metrics:**
- **Expected Calibration Error (ECE):** Mean absolute difference between confidence and accuracy across probability bins
- **Reliability diagrams:** Plot confidence vs. accuracy; diagonal is perfect calibration
- **Negative Log-Likelihood (NLL):** Penalizes both wrong predictions and overconfidence

**Calibration methods:**
- **Temperature scaling:** A single scalar $T$ applied to logits before softmax: $p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$. Tuning $T > 1$ spreads the probability mass (less confident); $T < 1$ sharpens. Highly effective and parameter-efficient.

## Uncertainty in Practice

### Out-of-Distribution (OOD) Detection
High epistemic uncertainty is the ideal signal for detecting OOD inputs. Methods:
- Maximum softmax probability (MSP)
- Energy-based scores
- Ensemble disagreement
- Mahalanobis distance in feature space

### Selective Prediction
Abstain from predictions when uncertainty exceeds a threshold, routing them to a human reviewer — critical in clinical and legal applications.

### Active Learning
Query labels for training examples where the model is most uncertain:
- **Query by committee:** Select examples on which ensemble members disagree most
- **BALD:** Maximize mutual information between model predictions and parameters

## Summary of Methods

| Method | Training Change | Inference Cost | Calibration |
|---|---|---|---|
| MC Dropout | Minimal (enable dropout) | Medium (×30–100) | Moderate |
| Deep Ensembles | Train M models | High (×M) | Best |
| Bayes by Backprop | Significant (2× params) | Medium | Good |
| Laplace Approximation | None (post-hoc) | Low-Medium | Good |
| Temperature Scaling | None (post-hoc) | None | Good (calibration only) |

## Further Reading

- Gal & Ghahramani (2016), *Dropout as a Bayesian Approximation*
- Lakshminarayanan et al. (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*
- Wilson & Izmailov (2020), *Bayesian Deep Learning and a Probabilistic Perspective of Generalization*
- Daxberger et al. (2021), *Laplace Redux — Effortless Bayesian Deep Learning*
