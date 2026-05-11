---
title: Learning to Optimize
description: Discover Learning to Optimize (L2O) — the paradigm of using neural networks to replace hand-designed optimizers. Covers LSTM-based L2O, VeLO, meta-learning optimizer frameworks, learned learning rates, the generalization problem, and how L2O competes with Adam and SGD on real tasks.
---

Optimization is the computational engine of machine learning: gradient descent, Adam, and their variants are carefully hand-engineered algorithms designed with theoretical guarantees and empirical intuitions. **Learning to Optimize (L2O)** asks a more ambitious question: can we train a neural network to discover optimization algorithms automatically — algorithms that outperform hand-designed ones, especially in the specific distribution of problems a practitioner cares about?

## The Core Idea

Standard optimization has a fixed update rule. For gradient descent:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

In L2O, the update rule itself is a parameterized function $m_\phi$ — an "optimizer network" — that maps the history of gradients and parameter states to an update:

$$\theta_{t+1} = \theta_t + m_\phi(g_t, g_{t-1}, \ldots, \theta_t, \ldots)$$

The optimizer parameters $\phi$ are trained (meta-trained) across a distribution of optimization problems, so that $m_\phi$ learns to take good steps for that distribution. At test time, $m_\phi$ is frozen and applied to new, unseen optimization problems.

## LSTM Optimizer (Andrychowicz et al., 2016)

The seminal L2O paper replaced the update rule with an **LSTM**:

- Input: the current gradient $g_t$ for each parameter (and optionally its history).
- Hidden state: the LSTM's hidden state $h_t$ tracks momentum-like information.
- Output: the parameter update $\Delta\theta_t$.

The LSTM is applied **coordinate-wise**: to handle optimizees of arbitrary dimension, the same LSTM is applied to each scalar gradient independently, sharing weights across all coordinates and all time steps. This makes the optimizer invariant to the number of parameters and their ordering.

### Meta-Training Procedure

The LSTM optimizer is meta-trained by unrolling the optimization of a random problem from a distribution (e.g., small neural networks on simple tasks) for $T$ steps:

$$\phi^* = \arg\min_\phi \mathbb{E}_{f \sim p(\mathcal{F})} \left[ \sum_{t=1}^T w_t \mathcal{L}(f, \theta_t) \right]$$

where $\theta_t$ is the iterates produced by $m_\phi$, and $w_t$ is a time-weighting (often exponential decay to focus on final loss). The gradient of this objective with respect to $\phi$ is computed through the unrolled optimization steps via **backpropagation through time (BPTT)**.

### What the LSTM Learns

Trained LSTM optimizers exhibit behaviors that mirror — and sometimes improve upon — hand-designed algorithms:

- **Adaptive learning rates**: the LSTM automatically adjusts step size per coordinate, similar to Adam.
- **Momentum**: the LSTM's hidden state carries gradient history across steps, implementing implicit momentum.
- **Curvature adaptation**: by observing gradient sequences, the LSTM implicitly estimates second-order information without explicit Hessian computation.

On problems similar to the training distribution, LSTM optimizers converge significantly faster (fewer gradient evaluations) than SGD, Adam, or RMSProp. On out-of-distribution problems, however, they can diverge catastrophically.

## Generalization Challenge

The core limitation of early L2O work is **poor generalization across problem distributions**:

- An LSTM trained on small 2-layer networks fails on 5-layer networks.
- An optimizer trained for 100 steps collapses when run for 1,000 steps (the unrolling horizon).
- An optimizer trained on quadratic functions fails on non-convex neural network loss landscapes.

Several techniques address this:

### Curriculum Training

Progressively increase the difficulty of meta-training problems: start with small networks and short unrolling horizons, gradually introduce larger networks and longer horizons. This prevents the LSTM from overfitting to specific problem geometry.

### Gradient Clipping and Preprocessing

Preprocessing raw gradients before feeding them to the LSTM (e.g., $\log(|g|)$ and $\text{sign}(g)$ separately) improves numerical stability and generalization. LSTM optimizers trained on raw gradients are sensitive to gradient scale, which varies enormously across problem types.

### Imitation Learning from Adam

Instead of meta-training purely by loss, pre-train the LSTM optimizer to **imitate Adam** on a diverse set of problems. The LSTM learns Adam's behavior as a prior, then fine-tunes to discover improvements. This dramatically reduces the training distribution mismatch for practical deployment.

## VeLO: Versatile Learned Optimizer

**VeLO** (Metz et al., Google Brain, 2022) is the largest and most general learned optimizer to date:

- **Training scale**: trained across a mixture of 4,000+ diverse tasks (image classifiers, NLP models, RL agents, reinforcement learning, numerical simulations) for 4,000 TPU-months of compute.
- **Architecture**: a transformer-based optimizer that processes the current parameter vector, gradient history, and optimizer state jointly, producing updates for all parameters simultaneously.
- **Generalization**: by training on an extraordinarily diverse task distribution, VeLO achieves competitive performance with Adam on new, unseen tasks drawn from qualitatively different domains.

VeLO can be used as a drop-in replacement for Adam by loading its pretrained weights and running its forward pass to compute updates. On held-out benchmarks spanning NLP, vision, and RL, VeLO achieves faster convergence than Adam in 40-60% of tasks — a major advance over earlier L2O methods that only worked on narrow task distributions.

### Computational Cost

Running VeLO adds overhead compared to Adam because the optimizer network itself requires computation. For small models, this overhead is significant. For large models, the optimizer compute is negligible compared to the forward/backward pass — making VeLO most attractive for large-scale training.

## Learned Learning Rate Schedules

A simpler form of L2O learns only the **learning rate schedule** rather than the full update rule. A small network predicts the learning rate $\eta_t$ at each step given the optimization history:

$$\eta_t = \text{LRNet}_\phi(\mathcal{L}_{t-1}, \mathcal{L}_{t-2}, \ldots, g_{t-1}, g_{t-2}, \ldots)$$

This is dramatically cheaper to meta-train than a full LSTM optimizer (the learning rate is a scalar; no coordinate-wise application needed) while still outperforming hand-designed cosine or polynomial schedules on the meta-training task distribution.

## Population-Based Training (PBT)

**Population-Based Training** (Jaderberg et al., DeepMind, 2017) is a related approach that adapts hyperparameters (including learning rate, momentum, regularization) during training by maintaining a population of agents and using evolutionary selection:

1. Train a population of $K$ models in parallel with different hyperparameters.
1. Periodically evaluate performance; underperforming agents **copy** the weights of a top-performing agent.
1. The copied agent **mutates** its hyperparameters (random perturbation) and continues training.

PBT discovers adaptive hyperparameter schedules without meta-training: the schedule emerges from the evolutionary dynamics. On Atari and protein structure prediction, PBT discovers learning rate schedules that outperform fixed schedules chosen by grid search.

## Optax: Composable Gradient Transformations

Rather than learned optimizers, **Optax** (DeepMind) provides a composable library of gradient transformations for JAX that enables rapid construction of new optimizers by chaining existing primitives:

```python
import optax

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),   # gradient clipping
    optax.scale_by_adam(),             # Adam-style adaptive scaling
    optax.scale_by_learning_rate(3e-4), # learning rate application
    optax.add_decayed_weights(1e-2),   # weight decay
)
```

This compositional approach reduces the gap between L2O research and practical use: learned components can be inserted as custom gradient transformations within the Optax framework.

## Hypernetworks for Optimizer Generation

**Hypernetworks** trained on a distribution of tasks can generate task-specific optimizer parameters:

- A meta-network maps task descriptors (e.g., model architecture, dataset statistics) to optimizer hyperparameters $(\eta, \beta_1, \beta_2, \epsilon)$.
- This is a one-shot prediction (unlike LSTM optimizers that run for $T$ steps) — computationally cheap at test time.
- Learns that vision tasks benefit from different Adam parameters than language tasks, or that residual networks need different momentum than attention networks.

## L2O in Practice: When to Use It

| Scenario | Recommendation |
| --- | --- |
| Training on the same architecture repeatedly | L2O with task-specific meta-training |
| Diverse tasks across domains | VeLO as Adam drop-in |
| Need theoretical guarantees | Stick with Adam/SGD |
| Hyperparameter sensitivity | PBT for adaptive scheduling |
| Resource-constrained deployment | Learned LR schedule only |

L2O offers the largest gains in **repeated optimization of similar problems** — training the same type of model many times (e.g., neural architecture search, meta-learning inner loops, hyperparameter optimization) where the cost of meta-training amortizes quickly.

## Summary

Learning to Optimize replaces hand-designed update rules with neural networks that learn to optimize from experience. LSTM-based coordinatewise optimizers demonstrated that learned algorithms can match and exceed Adam on in-distribution tasks. VeLO extended this to practical generalization by training across 4,000+ diverse tasks at massive compute scale. PBT provides evolutionary hyperparameter adaptation without explicit meta-training. The generalization gap — the tendency of L2O methods to fail outside their training distribution — remains the primary challenge, driving continued research into more robust meta-training distributions and more expressive optimizer architectures.
