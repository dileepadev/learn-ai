---
title: Loss of Plasticity in Deep Networks
description: Explore the loss of plasticity phenomenon — where neural networks progressively lose their ability to learn new information during training — including dormant neurons, dead units, and the continual backpropagation solution.
---

Loss of plasticity refers to the gradual degradation of a neural network's capacity to learn new patterns during training. Initially identified in the context of reinforcement learning (Dohare et al., 2021; Nikishin et al., 2022), it has since been recognized as a fundamental challenge in continual learning, online learning, and long-horizon training of large models.

## What Is Plasticity?

Plasticity is a network's ability to change its behavior in response to new gradient signals — to learn. A fully plastic network can fit any new dataset quickly from its current state. As training progresses, many networks lose this property: they become rigid, responding weakly to gradients even when their loss on new data is high.

This is distinct from **catastrophic forgetting**, where a network loses previously learned knowledge. A network with low plasticity may still retain old representations but simply cannot acquire new ones efficiently — even with strong gradient signals.

## Dormant Neurons

One major mechanism of plasticity loss is the emergence of **dormant neurons** (Sokar et al., 2023). A dormant neuron is one whose activations are near-zero across virtually all inputs:

$$\text{Dormancy}(h_i) = \frac{\mathbb{E}_x[|h_i(x)|]}{\mathbb{E}_x[\max_j |h_j(x)|] + \epsilon} \approx 0$$

Dormant neurons neither contribute to forward predictions nor receive meaningful gradients during backpropagation — they are effectively dead. In deep RL training, up to 70–80% of neurons in later layers can become dormant after extended training.

### Why Neurons Go Dormant

Several interacting mechanisms drive dormancy:

- **ReLU dead units**: A ReLU neuron receiving only negative inputs will produce zero activations and receive zero gradients, potentially forever
- **Gradient starvation**: In sparse activation regimes, many neurons rarely receive gradient updates
- **Norm growth**: Weight norms growing without bound cause saturation in downstream neurons
- **Negative learning transfer**: In non-stationary settings (RL, continual learning), previously useful features become harmful and are driven toward zero

## Measuring Plasticity

Several metrics quantify plasticity during training:

### Effective Rank

The **effective rank** of a layer's activation matrix $H \in \mathbb{R}^{B \times d}$ (over a batch of $B$ inputs) measures how many dimensions are meaningfully used:

$$\text{eff-rank}(H) = \exp\left(-\sum_i \tilde{\sigma}_i \log \tilde{\sigma}_i\right)$$

where $\tilde{\sigma}_i$ are the normalized singular values of $H$. Low effective rank means the network is compressing all inputs into a low-dimensional subspace, reducing its expressivity.

### Weight Magnitude and Dead Units

Tracking the fraction of neurons with mean activation below a threshold $\tau$ across a validation batch gives a direct dormancy estimate. In practice, $\tau = 0.01 \times \max$ activation is commonly used.

### Plasticity Loss Test

A direct test: after $T$ steps of training on the main task, train for $T'$ steps on a new, unrelated task (e.g., random labels or a different domain). The speed of loss reduction on the new task measures retained plasticity.

## Plasticity Loss in Reinforcement Learning

RL provides the clearest demonstrations of plasticity loss because:

- The data distribution is non-stationary (as the policy improves, it generates different experiences)
- The same network must adapt continuously to a moving target
- Long-horizon training (millions of steps) amplifies gradual degradation

Nikishin et al. (2022) showed that networks trained with the DQN algorithm on Atari games progressively lose plasticity. **Periodic resets** — reinitializing the network's parameters while keeping the replay buffer — restored plasticity and improved final performance, at the cost of transient performance drops.

The **primacy bias** (Nikishin et al., 2022) describes a related issue: early experiences disproportionately shape the network's representations because the network is most plastic early in training. These early representations then persist even when they become suboptimal.

## Continual Backpropagation

Dohare et al. (2023) proposed **continual backpropagation** (CBP) as a principled solution that maintains plasticity throughout training without requiring periodic hard resets.

### Algorithm

CBP identifies low-utility units and selectively reinitializes them:

1. Track a utility score $u_i$ for each unit $i$:

$$u_i^{(t)} = (1 - \beta) u_i^{(t-1)} + \beta \cdot \overline{|h_i|}$$

where $\overline{|h_i|}$ is the mean absolute activation over recent inputs and $\beta$ is a decay factor.

1. Periodically identify the fraction $\rho$ of units with the lowest utility scores

1. Reinitialize those units' incoming and outgoing weights to their initial distribution values (e.g., He initialization)

1. Continue training normally

The key insight is that low-utility units are not contributing to current predictions, so reinitializing them sacrifices little while restoring gradient flow and representational capacity.

### Properties

- **Gradual**: avoids the discontinuous performance drops of full resets
- **Targeted**: reinitializes only dormant units, preserving learned representations
- **Ongoing**: runs throughout training, not just at predetermined checkpoints
- **Hyperparameter-light**: only $\rho$ (fraction reinitialized) and $\beta$ (utility decay) need tuning

## Regenerative Regularization

An alternative to reinitialization is to regularize weights toward small magnitudes, which counteracts norm growth and preserves gradient flow. **L2 regularization toward initialization** (L2-Init) adds:

$$\mathcal{L}_{\text{reg}} = \frac{\lambda}{2} \|\theta - \theta_0\|^2$$

This pulls weights back toward their initial values, maintaining the sparse activation patterns and effective rank that enable plasticity. In practice, this simple technique is competitive with more complex methods on many RL benchmarks.

## Relationship to Feature Rank Collapse

**Feature rank collapse** (Kumar et al., 2021) is closely related: the feature matrix $\Phi(X) \in \mathbb{R}^{B \times d}$ (last layer activations over a batch) converges to low rank during training. This collapse:

- Reduces the expressivity of the learned representation
- Causes the linear head on top to overfit to the low-rank features
- Is exacerbated by TD learning and bootstrapping in RL

**Spectral normalization** of weight matrices and **explicit rank regularization** (adding a penalty for low effective rank) both mitigate feature rank collapse.

## Plasticity in Large Language Models

Plasticity concerns extend to large-scale language model training:

### Continual Pre-training

When continuing pre-training on new data (domain adaptation), plasticity loss in early layers can prevent the model from learning new vocabulary or syntax patterns. Layer-selective learning rate schedules — higher rates for later layers — partially compensate.

### Fine-tuning

Repeated fine-tuning across tasks (multi-task sequential fine-tuning) shows progressive performance degradation consistent with plasticity loss. Methods like **LoRA** (which adds low-rank updates without modifying base weights) preserve base model plasticity by keeping the original weight matrices intact.

## Comparison of Mitigation Strategies

| Method | Mechanism | Disruption | Overhead |
| --- | --- | --- | --- |
| Periodic reset | Reinitialize all weights | High (performance drop) | Low |
| Continual backprop | Reinitialize dormant units | Low | Low |
| L2-Init regularization | Pull weights toward initialization | None | Negligible |
| Spectral normalization | Constrain weight singular values | None | Low |
| Shrink-and-perturb | Scale + noise injection | Moderate | Low |
| ReDo (Sokar 2023) | Reset dormant neurons | Low | Low |

## Implications for Practice

### Training Duration

Plasticity loss accelerates with training duration. For very long training runs, periodic monitoring of dormancy fraction and effective rank is valuable.

### Architecture Choice

- **Layer normalization** mitigates activation collapse compared to batch normalization in non-stationary settings
- **Residual connections** help gradient flow to earlier layers, slowing plasticity loss
- **Smaller networks** tend to maintain plasticity longer in RL (less redundancy to become dormant)

### Optimizer Choice

Adam's adaptive learning rates partially compensate for plasticity loss by scaling up learning rates for consistently low-gradient parameters. However, Adam's momentum buffers accumulate stale gradient information that can actually worsen primacy bias.

## Summary

Loss of plasticity is a fundamental challenge in neural network training that becomes acute in non-stationary settings like reinforcement learning and continual learning. Dormant neurons, feature rank collapse, and weight norm explosion are the primary mechanisms. Solutions range from periodic hard resets to targeted continual backpropagation and simple regularization techniques. As AI systems are increasingly deployed in settings requiring lifelong adaptation, maintaining plasticity throughout training is becoming a first-class design concern.
