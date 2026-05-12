---
title: Thinking Tokens and Budget Forcing
description: Understand how modern reasoning models use thinking tokens — extended chains of internal deliberation generated before a final answer — and how budget forcing controls the amount of test-time computation. Covers the think-before-answer paradigm, token budget mechanisms, scaling laws for thinking, budget forcing techniques in o1, DeepSeek-R1, and QwQ, and the tradeoffs between reasoning depth and latency.
---

**Thinking tokens** (also called reasoning tokens, scratchpad tokens, or internal chain-of-thought) are sequences of tokens that a language model generates as intermediate deliberation before producing a final answer. Rather than mapping input directly to output, reasoning models are trained to first "think" — working through the problem step by step in a latent scratchpad — and only then emit the final response. **Budget forcing** refers to mechanisms that control the length and depth of this internal reasoning, trading off computation cost against answer quality.

This paradigm was popularized by OpenAI's o1 model series (2024) and subsequently implemented in DeepSeek-R1, QwQ-32B, Gemini Thinking, and Claude 3.7 Sonnet Extended Thinking.

## The Think-Before-Answer Paradigm

Standard language models are trained to generate the next token immediately from the input, relying on in-context chain-of-thought (CoT) only when users explicitly prompt for it. Reasoning models internalize CoT as a first-class training objective:

1. **Reasoning trace**: the model generates a long sequence of thinking tokens, typically enclosed in `<think>...</think>` tags or suppressed from the user-visible output.
1. **Final answer**: after the reasoning trace concludes, the model generates the user-facing answer using insights accumulated during thinking.

The key property: **thinking tokens are generated autoregressively** and thus undergo the same attention-based computation as any other token generation. More thinking tokens = more forward passes = more FLOP at inference time.

## Training Reasoning Models

### Process Reward Model (PRM) Supervision

Reasoning models are often trained with process-level supervision — a PRM scores the correctness of each reasoning step (not just the final answer). Training data consists of long chain-of-thought solutions to hard problems (math competition problems, competitive programming tasks, logical puzzles) with step-level labels.

### GRPO and Reinforcement Learning from Outcome Verification

**DeepSeek-R1** uses Group Relative Policy Optimization (GRPO) with outcome-level reward: the model generates multiple reasoning traces for each problem, and traces that lead to correct final answers receive positive reward. This incentivizes the model to learn reasoning patterns that reliably produce correct outcomes — without requiring manually labeled reasoning traces.

The training objective maximizes:

$$\mathcal{J}_\text{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^G} \left[ \sum_{i=1}^G \frac{r_i - \bar{r}}{\text{std}(r)} \cdot \log \pi_\theta(o_i \mid q) \right]$$

where $G$ rollouts are sampled per question $q$, and reward $r_i$ reflects answer correctness (and format compliance).

### Emergent Reasoning Behaviors

A remarkable finding from DeepSeek-R1-Zero (trained purely via RL with no SFT warm-up): the model spontaneously develops reasoning behaviors including:

- **Self-verification**: checking its own intermediate conclusions.
- **Backtracking**: recognizing an error and restarting a sub-chain.
- **Reflection**: explicitly noting "wait, I made a mistake" and correcting course.
- **Exploration**: trying multiple solution strategies before committing.

These behaviors emerge from RL incentives without being explicitly programmed — suggesting that extended thinking is a learned strategy for improving answer quality under verifiable feedback.

## Scaling Laws for Thinking

Reasoning models exhibit a distinct scaling law at inference: **answer quality improves with more thinking tokens**, up to a saturation point. Empirically:

- On AIME (math competition) problems, accuracy improves roughly logarithmically with the number of thinking tokens.
- On coding benchmarks (SWE-bench, LiveCodeBench), longer traces correlate with higher pass rates.
- The marginal benefit diminishes as thinking tokens grow very long — the model "overthinks" and sometimes reverses correct reasoning.

This creates a **test-time compute scaling curve** analogous to model scaling curves: quality vs. inference cost (measured in tokens or FLOP). The optimal operating point depends on task difficulty and latency budget.

## Budget Forcing

**Budget forcing** is the mechanism for controlling where on the test-time compute curve a model operates. The goal is to spend more compute on hard problems and less on easy ones — or to cap compute usage for cost-sensitive applications.

### Minimum Thinking Budget

Some deployments require a **minimum number of thinking tokens** to ensure the model doesn't short-circuit to a shallow answer. This is enforced by:

- Blocking the `</think>` end-of-thinking tag until a minimum token count is reached.
- Training the model with a mixture of problems at varying thinking lengths, encouraging deep thinking on hard examples.

### Maximum Thinking Budget

A **thinking token limit** (budget cap) can be set to bound inference cost. When the model approaches the limit:

- It is prompted to begin wrapping up: a system-level message like "you have N tokens left, finalize your answer" is injected.
- The model learns to summarize its partial reasoning and emit the best available answer given remaining compute.

OpenAI's o1 API exposes a `max_completion_tokens` parameter that caps total output tokens (thinking + answer). Higher caps enable more thorough reasoning on hard problems at higher cost.

### Adaptive Budget Allocation

Rather than a fixed cap, **adaptive budget forcing** allocates thinking tokens based on estimated problem difficulty:

1. The model generates a short "meta-thinking" phase to estimate how hard the problem is.
1. Based on this estimate, it allocates a thinking budget and proceeds.
1. Alternatively, an external classifier routes simple queries to fast (no thinking) paths and complex queries to extended thinking paths.

**s1: Simple Test-Time Scaling** (Muennighoff et al., 2025) demonstrated that supervised fine-tuning on a small set of 1K carefully selected reasoning problems (s1K) followed by budget forcing at inference matches or exceeds o1-preview on competition math — showing that data quality and compute control matter more than raw scale.

### Budget Tokens as Input Conditioning

**QwQ-32B** and experimental reasoning models condition on a target thinking length as part of the input: "think for approximately 2000 tokens." The model adapts the depth of its reasoning to the specified budget, spending more tokens on exploration when given a generous budget and compressing reasoning when given a tight one.

This can be formalized as conditioning the model on a budget variable $b$:

$$p_\theta(\text{answer} \mid \text{question}, b) = \int p_\theta(\text{answer} \mid \text{reasoning}) \cdot p_\theta(\text{reasoning} \mid \text{question}, b) \, d(\text{reasoning})$$

## Visible vs. Hidden Thinking

Models differ in whether thinking tokens are shown to users:

- **Hidden thinking (o1, o3)**: OpenAI suppresses the reasoning trace from API responses by default. Users see only the final answer; the scratchpad is not exposed, partly for safety (reasoning traces could be steered toward harmful content more easily) and partly for UX.
- **Visible thinking (DeepSeek-R1, QwQ, Claude Extended Thinking)**: reasoning traces are returned to the user, enabling inspection, debugging, and trust calibration. Users can observe the model checking its work, backtracking, and synthesizing its findings.
- **Optional visibility (Claude 3.7 Sonnet)**: the Extended Thinking feature allows users to choose whether to show or hide the reasoning summary.

## Thinking in Multimodal and Agentic Settings

Thinking tokens extend naturally to multimodal and agentic contexts:

- **Multimodal reasoning**: the model reasons about image content (e.g., counting objects, interpreting charts, spatial reasoning) using thinking tokens before describing or answering.
- **Agentic loops**: in multi-step agent workflows, each action step can include a thinking phase — the agent plans what tool to call and why before executing, improving the reliability of tool use.
- **Code generation**: the model reasons about algorithm design, edge cases, and time complexity in the thinking trace before writing code, substantially reducing bugs in difficult coding tasks.

## Tradeoffs

| Property | More Thinking Tokens | Fewer Thinking Tokens |
| --- | --- | --- |
| Answer accuracy | Higher (for hard tasks) | Lower |
| Latency | Higher | Lower |
| API cost | Higher (billed per token) | Lower |
| Suitable for | Math, coding, logic, planning | Summarization, classification, simple Q&A |
| Risk | Overthinking, reversed correct answers | Shallow reasoning, missed steps |

## Summary

Thinking tokens allow language models to perform extended internal deliberation before committing to a final answer, dramatically improving performance on hard reasoning tasks. Models like o1, DeepSeek-R1, and QwQ are trained with RL-based objectives that incentivize correct, verifiable final answers — causing the model to spontaneously develop behaviors like self-verification, backtracking, and multi-strategy exploration. Budget forcing controls the amount of thinking at inference time through minimum and maximum token limits, adaptive allocation, or direct budget conditioning. The result is a new axis of model capability — **test-time compute scaling** — where harder problems receive proportionally more computation, and inference cost becomes a tunable parameter rather than a fixed property of model size.
