---
title: Scaling Laws in Deep Learning - Predicting Performance Before You Train
description: Understand the empirical power-law relationships that let researchers predict model performance from compute, data, and parameter count.
---

Scaling laws describe how a model's test loss changes as you increase model size, dataset size, or training compute, holding other factors reasonably fixed. Remarkably, these relationships follow smooth power laws across many orders of magnitude, letting researchers extrapolate from small, cheap experiments to predict the performance of models that haven't been trained yet.

```text
loss(N) ≈ a * N^(-alpha) + c        (N = number of parameters)
loss(D) ≈ b * D^(-beta)  + c        (D = number of training tokens)
```

As `N` or `D` grows, loss decreases predictably and continues improving well beyond the scales where earlier researchers expected returns to plateau.

## The Chinchilla Insight: Compute-Optimal Training

An influential 2022 study (the "Chinchilla" paper) found that many earlier large models were **undertrained** relative to their size—they used too many parameters for the amount of training data seen. For a fixed compute budget, there is an optimal balance between model size and dataset size that minimizes loss:

$$N_{opt} \propto C^{a}, \quad D_{opt} \propto C^{b}$$

where `C` is the total training compute (roughly proportional to parameters times tokens processed). Their key finding: model size and training tokens should scale roughly in proportion to each other as compute increases—meaning many contemporary models could have achieved lower loss for the same compute budget simply by training a smaller model on more data.

## Why Scaling Laws Are Useful

- **Budget planning**: teams can decide, before committing a large compute budget, roughly how big a model to train and how much data to gather.
- **Early stopping decisions**: a small-scale experiment can indicate whether a change in architecture or data mix is likely to help at full scale, without needing the full run.
- **Capability forecasting**: scaling trends have helped anticipate qualitative capability jumps, though predicting exactly when specific skills emerge remains unreliable.

## Limits of Scaling Laws

Power-law fits from one regime don't always extrapolate perfectly to very different scales or data mixes, and data quality, architecture choices, and training stability all shift the constants in ways the simple formulas don't capture. Some capabilities also appear to "emerge" somewhat discontinuously rather than following the smooth loss curve, and there is ongoing debate over whether apparent emergence is a real phenomenon or an artifact of the specific metrics used to measure it. Additionally, high-quality training data is finite, which has shifted research attention toward data efficiency and synthetic data generation as pure data scaling becomes harder to sustain.
