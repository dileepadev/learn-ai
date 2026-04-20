---
title: Mixture of Agents (MoA)
description: How layering multiple LLMs together as proposers and aggregators produces outputs that surpass any individual model — exploring the Mixture of Agents architecture, its collaborative reasoning principles, and real-world applications.
---

**Mixture of Agents (MoA)** is an inference-time ensemble architecture where multiple large language models collaborate in a layered pipeline to produce a final response that outperforms any single model acting alone. Introduced by researchers at Together AI (Wang et al., 2024), MoA leverages **collaborative reasoning** across diverse LLMs rather than model architecture changes or additional fine-tuning.

MoA is distinct from **Mixture of Experts (MoE)**, which is an internal model architecture where different subsets of a single model's parameters are activated per token. In MoA, each agent is an entirely separate model.

## The Core Insight: Complementary Strengths

Different LLMs have different strengths, training distributions, and failure modes. A question that trips up GPT-4o may be handled well by Claude, and vice versa. By exposing each model to the outputs of other models — even from weaker models — the generating model can **reference, critique, and synthesize** a richer set of perspectives.

Empirically, it was found that:

- LLMs produce better outputs when given responses from other LLMs as reference context, even when those reference responses are lower quality than what the model could generate alone.
- The improvement from adding references is **consistent across model families** (GPT, Claude, Gemini, Llama, etc.).

## Architecture Overview

MoA operates in **layers**, each containing multiple **proposer agents**:

```text
Layer 1:  [Model A] [Model B] [Model C]
              ↓         ↓         ↓
          Response_A Response_B Response_C
                      ↓
Layer 2:  [Model D (Aggregator)]
          Input: Original prompt + Response_A + Response_B + Response_C
              ↓
          Synthesized Response
                      ↓
Layer 3:  [Model E (Final Aggregator)]
              ↓
          Final Answer
```

### Proposers

Proposer agents in layer $i$ receive the original prompt and — in all layers beyond the first — the responses from all proposers in layer $i-1$. Each proposer generates its own independent response.

### Aggregators

An aggregator also receives all previous-layer outputs and synthesizes them into a single, cohesive response. Aggregators tend to be the strongest available model since synthesis requires high capability.

## The Collaborative Effect

The key mechanism is that reference outputs serve as **implicit chain-of-thought scaffolding**. The aggregator can:

- **Identify the correct answer** when multiple proposers agree.
- **Spot errors** when one proposer disagrees with the majority.
- **Combine partial knowledge** — one model may get the reasoning right while another gets domain-specific facts right.
- **Improve formatting and structure** by seeing multiple presentation styles.

This mirrors how human expert panels work: individual opinions may be flawed, but a synthesizer who reads all opinions can arrive at a better conclusion than any individual.

## Performance Results

In the original MoA paper, a configuration using multiple open-source models (Llama-3, Qwen, WizardLM, etc.) as proposers with GPT-4o as the final aggregator achieved:

- **Higher AlpacaEval 2.0 scores** than GPT-4o alone.
- **Higher MT-Bench scores** than any single model.
- Competitive performance with GPT-4 Turbo at a fraction of the cost when using open-source proposers.

The result held across categories: writing, coding, reasoning, and knowledge tasks.

## MoA vs. Self-Consistency

**Self-consistency** (Wang et al., 2022) is a related technique where a single model generates multiple responses and then selects or aggregates the most consistent one. MoA differs in that:

| Aspect | Self-Consistency | Mixture of Agents |
| --- | --- | --- |
| Diversity source | Sampling temperature | Different model families |
| Failure modes | Correlated — same model biases | Uncorrelated — different biases |
| Cost | Cheaper (one model) | More expensive (multiple APIs) |
| Ceiling | Bounded by single model capability | Can exceed any individual model |

## Design Considerations

### Choosing Proposers

- Use **diverse model families** to maximize uncorrelated perspectives (e.g., GPT, Claude, Gemini, open-source Llama).
- Include **specialized models** for domain-specific tasks (code, math, multilingual).
- Weaker or smaller models still add value as reference providers.

### Choosing Aggregators

- The aggregator should be the **strongest general-purpose model** available.
- The aggregator prompt should explicitly instruct the model to reference and synthesize the provided responses rather than ignore them.

### Number of Layers

- Most tasks benefit from **1–2 aggregation layers**.
- Deeper pipelines increase latency and cost without proportional gains for typical tasks.
- Complex, multi-step tasks (research synthesis, code generation with review) may benefit from additional layers.

### Latency and Cost

MoA multiplies API calls by the number of proposers per layer. Strategies to manage this:

- **Parallel proposer calls** — all layer-1 proposers can be called simultaneously.
- **Cheaper proposer models** — use cost-efficient models as proposers; reserve the powerful model for final aggregation.
- **Selective activation** — only activate MoA for high-stakes or difficult queries.

## Practical Applications

- **High-stakes content generation**: Legal drafts, medical summaries, technical documentation where accuracy matters more than latency.
- **Code generation**: Multiple models generate solutions; an aggregator selects and refines the best.
- **Factual synthesis**: Research summaries where different models may have different knowledge cutoffs or specializations.
- **Evaluation pipelines**: Using MoA as a judge to evaluate other LLM outputs more reliably than a single model judge.

## Limitations

- **Latency**: Sequential layers introduce unavoidable latency; even with parallel proposers, aggregation adds time.
- **Cost**: Running 5+ models per query is expensive at scale.
- **Aggregator bottleneck**: If the aggregator model has a blind spot or bias, it filters all outputs through that lens.
- **Context length**: Combining multiple full responses as context can exceed context windows for complex tasks.

Mixture of Agents represents a compelling approach to squeezing higher quality from existing models without retraining — a practical engineering solution to the limits of any single model.
