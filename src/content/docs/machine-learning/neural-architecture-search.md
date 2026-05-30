---
title: "Neural Architecture Search (NAS)"
description: "Learn how Neural Architecture Search automates the design of neural network architectures, from early grid search approaches to modern differentiable and hardware-aware NAS methods."
---

Designing neural network architectures has traditionally required deep expertise and extensive manual experimentation. **Neural Architecture Search (NAS)** automates this process, using algorithms to discover architectures that outperform human-designed ones on specific tasks and hardware targets.

## The NAS Problem

Given a search space of possible architectures, an objective function (validation accuracy, latency, memory), and a computational budget, find the architecture that maximizes the objective.

The challenge: the search space is enormous (10^18+ possible architectures for some spaces), and evaluating each candidate requires training it — which is expensive.

## Search Strategies

### Random Search
Surprisingly competitive baseline. Sample architectures randomly and evaluate them. Simple but requires many evaluations.

### Reinforcement Learning
A controller network generates architecture descriptions as sequences of tokens. The controller is trained with RL using validation accuracy as the reward. Used in the original NASNet paper (Google, 2017), which found architectures that outperformed hand-designed ones — but required 500 GPUs for weeks.

### Evolutionary Algorithms
Maintain a population of architectures. Mutate and recombine the best performers. More sample-efficient than RL in practice.

### Differentiable NAS (DARTS)
The key insight: instead of searching over discrete architectures, relax the search space to be continuous. Assign a learnable weight to each possible operation at each position. Train the architecture weights and model weights jointly with gradient descent. At the end, discretize by selecting the highest-weight operations.

DARTS reduced NAS from weeks to hours but introduced stability issues (the architecture weights can collapse to degenerate solutions).

## One-Shot NAS and Supernets

Train a single **supernet** that contains all possible architectures as subgraphs. Individual architectures are evaluated by sampling paths through the supernet without retraining. This amortizes training cost across all candidate architectures.

**Once-for-All (OFA)** trains a supernet that can be sliced to produce architectures optimized for different hardware targets (mobile, edge, server) without retraining.

## Hardware-Aware NAS

Accuracy alone is insufficient for deployment. Hardware-aware NAS optimizes for:
- Latency on specific hardware (mobile CPU, GPU, NPU).
- Energy consumption.
- Memory footprint.

**MobileNetV3**, **EfficientNet**, and **EfficientDet** were all found or refined using hardware-aware NAS.

## NAS for Transformers

NAS has been applied to transformer architectures to find optimal:
- Number of attention heads.
- Feed-forward layer dimensions.
- Layer depth and width ratios.

**AutoFormer** and **HAT (Hardware-Aware Transformers)** are examples of NAS applied to vision and language transformers.

## Current Limitations

- **Proxy tasks**: NAS often searches on small datasets/models and transfers results to larger settings, which doesn't always work.
- **Search space design**: The quality of NAS results is bounded by the quality of the search space — garbage in, garbage out.
- **Reproducibility**: NAS results are sensitive to implementation details and random seeds.

Despite these limitations, NAS-discovered architectures are now standard in mobile and edge AI deployment.
