---
title: Discrete Diffusion Models
description: Explore discrete diffusion models — generative models that apply the diffusion framework to categorical data such as text, graphs, and protein sequences — covering D3PM, masked diffusion, absorbing states, and their applications.
---

Diffusion models dominated continuous data (images, audio) by learning to reverse a Gaussian noising process. **Discrete diffusion models** extend this framework to categorical domains — text, code, protein sequences, graphs, and any data represented as finite-alphabet tokens. Rather than adding Gaussian noise, discrete diffusion corrupts data through categorical noise: random substitutions, masking (absorbing states), or uniform corruption. The generative process then learns to iteratively denoise toward the original discrete tokens.

## Why Discrete Diffusion?

Most language modeling uses autoregressive generation (GPT-style), which generates tokens left-to-right. Discrete diffusion offers a compelling alternative:

- **Non-autoregressive generation**: all tokens are generated simultaneously across diffusion steps, enabling parallel decoding
- **Flexible conditioning**: arbitrary positions can be masked and filled (cloze-style generation, infilling)
- **Bidirectional context**: the denoising model sees all positions simultaneously at each step
- **Controllable generation**: fine-grained control over which tokens are regenerated

## The Forward Process on Discrete Data

For continuous diffusion, the forward process adds Gaussian noise:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

For discrete data with $K$ categories, the forward process is defined via a **transition matrix** $Q_t \in \mathbb{R}^{K \times K}$:

$$q(x_t | x_{t-1}) = \text{Cat}(x_t; p = x_{t-1} Q_t)$$

where $x_{t-1}$ is a one-hot vector and $Q_t$ specifies how each token transitions to a noisy state.

### Marginals and Posterior

The key property exploited in training is the tractable marginal and posterior:

$$q(x_t | x_0) = \text{Cat}\left(x_t; p = x_0 \bar{Q}_t\right), \quad \bar{Q}_t = Q_1 Q_2 \cdots Q_t$$

And the posterior (used to compute the training objective):

$$q(x_{t-1} | x_t, x_0) \propto q(x_t | x_{t-1}) q(x_{t-1} | x_0)$$

Both are analytically tractable with categorical distributions, enabling efficient training via the variational lower bound.

## D3PM: Structured Denoising Diffusion for Discrete State Spaces

**Austin et al. (2021)** introduced D3PM (Discrete Denoising Diffusion Probabilistic Models), a general framework supporting multiple transition types.

### Transition Types

**Uniform noise**: Each token independently becomes any of $K$ categories with equal probability:

$$Q_t = (1 - \beta_t) I + \frac{\beta_t}{K} \mathbf{1}\mathbf{1}^T$$

At $T \to \infty$, the stationary distribution is uniform over all tokens — complete information loss.

**Absorbing (mask) state**: A special token `[MASK]` acts as an absorbing state. Once masked, a token stays masked:

$$Q_t = (1 - \beta_t) I + \beta_t \mathbf{e}_m \mathbf{1}^T$$

where $\mathbf{e}_m$ is the mask token one-hot vector. At $T \to \infty$, all tokens are masked. Generation starts from a fully masked sequence and progressively reveals tokens.

**Token-specific noise**: Transitions use a similarity-based matrix informed by embedding distances, so semantically similar tokens are more likely substitutions than unrelated ones.

### Training Objective

D3PM is trained via the evidence lower bound (ELBO), which decomposes as:

$$\mathcal{L} = \mathbb{E}_{q}\left[\sum_{t=1}^T D_{\text{KL}}\left(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)\right)\right] + \text{reconstruction}$$

An auxiliary cross-entropy loss $\mathcal{L}_{\text{CE}} = -\mathbb{E}[\log p_\theta(x_0 | x_t)]$ is added to directly supervise $x_0$ prediction, which improves training stability.

## Masked Diffusion Language Models (MDLMs)

A highly popular special case focuses entirely on the **absorbing mask state**. **Sahoo et al. (2024)** — MDLM — and related work by **Shi et al. (2024)** simplified discrete diffusion by:

1. Using only the absorbing/mask transition (no uniform noise, no token-specific noise)
1. Predicting $x_0$ (the original token) directly from $x_t$ (the masked sequence)
1. Using a monotone masking schedule analogous to the continuous noise schedule

The forward process simply masks each token independently with probability $\bar{\alpha}_t$:

$$q(x_t | x_0) = \text{Bernoulli-Mask}(x_0, \bar{\alpha}_t)$$

At $t = T$, all tokens are masked. At $t = 0$, no tokens are masked.

### Connection to BERT

Masked diffusion is conceptually related to BERT's masked language modeling (MLM), but with a key difference: BERT masks a fixed fraction (15%) and trains once, while masked diffusion is trained across all masking rates $\bar{\alpha}_t \in [0, 1]$ and generates by iteratively unmasking.

## MDLM and Continuous-Time Limits

Diffusion can be formulated in continuous time with a rate matrix $R_t$ governing the instantaneous transition rates. For the absorbing diffusion:

$$\frac{d}{dt} q(x_t | x_0) = q(x_t | x_0) R_t$$

The reverse process has an analogous continuous-time form. Continuous-time formulations allow flexible sampling schedules and connection to score-based generative models in discrete spaces.

**Campbell et al. (2022)** (CTMC diffusion) formalized the continuous-time Markov chain framework for discrete diffusion, enabling:

- Arbitrary (non-uniform) time steps during sampling
- Connection between discrete score matching and discrete diffusion
- Improved sampling efficiency

## Sampling Strategies

### Ancestral Sampling

Standard sampling follows the learned reverse process step by step from $t = T$ to $t = 0$:

1. Start with fully noised/masked sequence $x_T$
1. For each step $t$ from $T$ down to $1$: sample $x_{t-1} \sim p_\theta(x_{t-1} | x_t)$
1. Return $x_0$

This is slow (hundreds of steps) but faithful to the generative model.

### $\tau$-Leaping

Borrowed from stochastic chemistry simulation, $\tau$-leaping takes larger steps by approximating multiple transitions at once, trading accuracy for speed.

### Nucleus Sampling and Temperature

At each step, the predicted $p_\theta(x_0 | x_t)$ can be filtered with top-$p$ or temperature scaling before sampling, analogous to language model decoding.

## Discrete Diffusion for Graphs

Graphs — nodes and edges with discrete types — are natural candidates for discrete diffusion. **DiGress (Vignac et al., 2023)** applies discrete diffusion to molecular graph generation:

- Node and edge types are treated as categorical tokens
- A graph transformer architecture predicts denoised graphs at each step
- Marginals are defined over the joint distribution of node/edge adjacency matrices
- Structural validity is enforced via graph-conditioned denoising

DiGress achieves state-of-the-art performance on molecular graph benchmarks (ZINC, QM9) by leveraging discrete diffusion's ability to handle mixed categorical data.

## Protein Sequence Diffusion

Proteins are discrete sequences over a 20-amino-acid alphabet. Discrete diffusion enables:

- Generating valid protein sequences conditioned on structure
- Designing sequences for target functions using classifier guidance
- Combining with continuous structure diffusion for joint sequence-structure generation

**EvoDiff (Alamdari et al., 2023)** applies masked discrete diffusion to protein sequences at scale, showing that sequence-only models can generate diverse, evolutionarily plausible proteins without requiring structural inputs.

## Comparison with Autoregressive Models

| Property | Autoregressive (GPT) | Discrete Diffusion |
| --- | --- | --- |
| Generation order | Left-to-right | All positions simultaneously |
| Context at generation | Left context only | Full bidirectional context |
| Speed (parallel) | Sequential | Parallelizable per step |
| Infilling/editing | Awkward (requires tricks) | Natural (mask positions) |
| Quality at scale | State-of-the-art | Competitive but lagging |
| Training | Next-token prediction | Variational lower bound |

At current scale, autoregressive models still dominate on language quality benchmarks, but discrete diffusion is competitive on structured generation tasks (molecules, proteins, code infilling).

## MDLM vs. Autoregressive: The Speed Argument

One key motivation for discrete diffusion is **parallel decoding**. In masked diffusion, all tokens are partially revealed at each step. With $T$ steps and sequence length $L$:

- Autoregressive: $O(L)$ sequential forward passes
- Masked diffusion: $O(T)$ forward passes, each processing all positions simultaneously

In practice, $T \ll L$ is achievable with fewer steps than tokens, giving a wall-clock speedup. **MDLM** demonstrated that 10–20 sampling steps can produce quality competitive with much more expensive autoregressive decoding.

## Current Challenges

**Quality gap**: At large scale, discrete diffusion language models still trail GPT-class autoregressive models on perplexity and downstream benchmarks.

**Training efficiency**: The ELBO has higher variance than next-token cross-entropy; balancing the per-timestep KL terms requires careful weighting.

**Exposure bias in reverse steps**: Each step conditions on previously denoised tokens, but errors accumulate differently than in autoregressive models.

**Discrete score matching**: Unlike continuous diffusion where the score $\nabla \log p(x)$ is well-defined, the discrete analog (ratio estimators or jump rate ratios) is less established.

## Summary

Discrete diffusion models bring the diffusion paradigm to categorical data, enabling bidirectional, non-autoregressive generation for text, graphs, proteins, and more. The key ingredients are:

- A forward noising process defined by categorical transition matrices (uniform, absorbing, token-specific)
- A neural network trained to reverse the noising process, predicting $x_0$ or $x_{t-1}$ from $x_t$
- A tractable ELBO training objective derived from the categorical marginals

Masked diffusion (absorbing state) has emerged as the dominant variant, offering a clean connection to BERT-style masked language modeling while enabling iterative generation. As scale increases and training stabilizes, discrete diffusion is a strong candidate to complement or eventually challenge autoregressive generation for certain domains.
