---
title: Multi-Token Prediction in LLMs
description: How training language models to predict multiple future tokens simultaneously improves sample efficiency, representation quality, and inference speed — covering the MTP objective, its use in LLaMA 3, the connection to speculative decoding, and the theoretical motivations behind predicting beyond the next token.
---

**Multi-token prediction (MTP)** is a training objective where a language model is tasked with predicting not just the next token, but the next $n$ tokens simultaneously from a single forward pass. This seemingly simple modification to the standard causal language model objective has been shown to improve both training efficiency and downstream task performance — and it enables new inference strategies like self-speculative decoding.

## The Standard Next-Token Prediction Objective

Virtually all modern autoregressive language models are trained with the **next-token prediction** (NTP) objective:

$$\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{< t})$$

At each position, the model predicts a probability distribution over the vocabulary for the immediately following token. This is simple, scalable, and proven — but it leaves potential on the table. A token's prediction is influenced only by its immediate neighbors; the model has no explicit incentive to plan ahead or develop representations useful for longer-range structure.

## The Multi-Token Prediction Objective

In MTP, $n$ additional output heads are added to the base transformer, each predicting a token further into the future:

$$\mathcal{L}_{\text{MTP}} = -\sum_{t=1}^{T} \sum_{k=1}^{n} \lambda_k \log P_\theta^{(k)}(x_{t+k} \mid x_{< t})$$

Where $P_\theta^{(k)}$ is the $k$-th prediction head and $\lambda_k$ are loss weights (often equal or decaying with $k$).

The architecture typically uses:

- **Shared trunk**: The main transformer processes the input sequence once, producing a shared hidden representation.
- **Independent prediction heads**: Each future-token head is a shallow MLP or small transformer on top of the shared representation.

The additional heads are used **only during training**. At inference time, the model uses only the first head (standard next-token prediction), so there is no inference overhead.

## LLaMA 3 and Multi-Token Prediction

**LLaMA 3** (Meta, 2024) incorporated multi-token prediction during training, predicting 4 tokens ahead in addition to the standard next-token loss. The paper reported:

- **Higher benchmark scores** on coding tasks (HumanEval, MBPP) with MTP vs. without.
- **Improved reasoning** on mathematical tasks.
- No degradation on language modeling perplexity.

The improvement is hypothesized to come from forcing the model's representations to encode information useful for planning ahead — not just resolving the immediate next token.

## Why Multi-Token Prediction Helps

### Better Gradient Signal

Each training token now contributes to $n$ prediction tasks rather than one. The total gradient signal per token is richer and more informative. This is particularly valuable in **data-limited regimes** — MTP is, in effect, a form of self-supervised data augmentation on the same corpus.

### Forcing Longer-Range Representations

With NTP, the model's hidden states only need to encode information sufficient to predict the next token. With MTP, the hidden states at position $t$ must simultaneously support predicting $x_{t+1}, x_{t+2}, \ldots, x_{t+n}$. This encourages representations that capture **higher-level structure** — sentence-level intentions, code logic, mathematical reasoning chains — rather than purely local n-gram statistics.

### Code and Structured Data Benefits

Structured outputs like code, JSON, and mathematical expressions have strong dependencies between tokens that are spaced several positions apart. For example, a function name declared on line 1 must match its call on line 10. MTP forces the model to develop representations that anticipate these relationships.

Empirically, coding benchmarks show the largest MTP gains, consistent with this hypothesis.

## Multi-Token Prediction for Faster Inference

Beyond training, MTP heads can be repurposed at inference time for **self-speculative decoding**:

1. The main model generates $n$ draft tokens in a single forward pass using the $n$ prediction heads.
2. A verification step runs a single forward pass checking all $n$ draft tokens simultaneously.
3. Tokens are accepted or rejected based on consistency with the true distribution.

This is the same algorithmic principle as speculative decoding — but the draft model is built into the same weights as the target model, eliminating the need for a separate smaller draft model.

**Practical speedup**: Accepting even 2–3 draft tokens per verification step can yield 1.5–2× wall-clock inference speedup on hardware where memory bandwidth is the primary bottleneck (which is true for most autoregressive LLM inference scenarios).

## Connection to Speculative Decoding

Traditional speculative decoding requires:

- A **draft model** (small, fast): Generates candidate token sequences.
- A **target model** (large): Verifies the draft in parallel.

MTP self-speculative decoding replaces the separate draft model with **future-token prediction heads** attached to the main model. This simplifies deployment (one model file, one loading step) and avoids the distribution mismatch between two separately-trained models.

The acceptance rate — fraction of draft tokens that pass verification — determines the speedup. MTP heads trained jointly with the main model tend to achieve higher acceptance rates than externally distilled draft models, because they share identical internal representations.

## Variants and Extensions

### Independent Heads vs. Sequential Heads

**Independent heads**: Each head $k$ independently predicts $x_{t+k}$ from the shared representation at $t$. Simple, parallelizable.

**Sequential (causal) heads**: The $k$-th head conditions on all previous heads' outputs, predicting $x_{t+k}$ given the context and $x_{t+1}, \ldots, x_{t+k-1}$ from prior heads. More powerful but requires sequential computation.

DeepMind's research found that sequential heads better model the joint distribution over future tokens, at the cost of compute.

### Consistency-Based MTP

Rather than training separate heads from scratch, **consistency models** for language enforce that the model's internal representations at step $t$ and $t+k$ are self-consistent — a soft form of MTP based on representation alignment rather than explicit prediction heads.

### Diffusion Language Models

Non-autoregressive **diffusion-based language models** (MDLM, PLAID) inherently predict multiple tokens simultaneously by denoising masked positions. MTP for autoregressive models can be seen as bridging toward this regime — learning to plan beyond the immediately next step.

## Implementation Notes

Adding MTP to an existing transformer training setup is straightforward:

```python
# After the main transformer produces hidden states `h`
main_logits = lm_head(h)                  # shape: [B, T, V]
mtp_logits_2 = mtp_head_2(h)             # predict x_{t+2}
mtp_logits_3 = mtp_head_3(h)             # predict x_{t+3}

# Shift labels for each head
labels_1 = tokens[:, 1:]                  # standard next-token
labels_2 = tokens[:, 2:]                  # two ahead
labels_3 = tokens[:, 3:]                  # three ahead

loss = (
    cross_entropy(main_logits[:, :-1], labels_1)
    + λ * cross_entropy(mtp_logits_2[:, :-2], labels_2)
    + λ * cross_entropy(mtp_logits_3[:, :-3], labels_3)
)
```

The additional heads add a small number of parameters (typically < 5% of total model size) and a modest training compute overhead (< 10%).

## Relationship to Other Self-Supervised Objectives

MTP is part of a broader family of richer pretraining objectives:

| Objective | What is Predicted | Key Benefit |
| --- | --- | --- |
| NTP (standard) | Next token | Simple, scales well |
| MTP | Next $n$ tokens | Richer gradient, faster inference |
| Masked LM (BERT) | Masked tokens (bidirectional) | Better encoder representations |
| Span corruption (T5) | Contiguous spans | Better seq2seq transfer |
| Contrastive (CLIP) | Cross-modal alignment | Multimodal representations |

MTP is unique in that it improves the autoregressive objective without changing the inference-time model structure — making it a free lunch for training that also enables new inference capabilities.

## Practical Takeaways

- MTP is a **low-cost training upgrade**: few extra parameters, modest compute overhead, measurable gains.
- Benefits are **most pronounced for code and structured output** tasks.
- MTP heads **enable self-speculative decoding** without a separate draft model — a practical inference speedup.
- The optimal value of $n$ appears to be **4–8** for current model scales; beyond this, marginal gains diminish.
- MTP is now a **standard component** in competitive open-source model training recipes (LLaMA 3, DeepSeek V3).
