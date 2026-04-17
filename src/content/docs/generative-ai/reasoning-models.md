---
title: Reasoning Models
description: Explore the new class of LLMs that think before answering — covering chain-of-thought at inference time, o1/o3 style process reward models, DeepSeek-R1, and how extended thinking improves performance on hard reasoning tasks.
---

**Reasoning models** are a category of large language models that dedicate significant compute at **inference time** to an extended internal thinking process before producing a final answer. Unlike standard LLMs that generate responses in a single left-to-right pass, reasoning models produce a hidden **chain of thought** that explores multiple approaches, backtracks, self-critiques, and verifies intermediate steps.

The defining property: **scaling inference compute** (not just training compute) improves performance on hard reasoning tasks.

## The Limits of Standard LLMs

Standard LLMs are remarkably capable, but they have a fundamental structural limitation for complex reasoning: they must produce the answer in a single autoregressive pass with a fixed computation budget per token. This constrains them to "System 1" thinking — fast, intuitive pattern matching — making multi-step logical deduction, math proofs, and complex planning difficult.

Human experts approach hard problems differently: they think for longer, try multiple approaches, catch mistakes, and refine their reasoning. Reasoning models replicate this "System 2" deliberative process.

## Extended Chain-of-Thought (CoT)

The foundation of reasoning models is **extended chain-of-thought** — allowing the model to produce a long internal scratchpad before the final answer. Unlike standard prompting CoT (a few reasoning sentences), reasoning models may generate hundreds to thousands of tokens of internal reasoning.

This extended thinking enables:

- **Decomposing** complex problems into manageable sub-problems.
- **Backtracking** when a line of reasoning leads to a contradiction.
- **Verifying** intermediate calculations before proceeding.
- **Exploring alternatives** before committing to an answer.

## OpenAI o1 and o3

**OpenAI o1** (September 2024) introduced the reasoning model paradigm to mainstream use. Key characteristics:

- Trained with **reinforcement learning** to generate useful thinking traces.
- The internal reasoning ("thinking") is hidden from users; only the final answer is shown.
- Dramatically outperforms GPT-4o on competition math (AIME), coding (Codeforces), and PhD-level STEM questions (GPQA).
- **o3** (early 2025) extended this significantly, reaching human-expert-level performance on frontier benchmarks including ARC-AGI.

**Scaling law:** For reasoning models, performance on hard tasks scales with the **amount of inference compute** (thinking tokens) in a way that standard models do not exhibit. More thinking → better answers, up to a point.

## DeepSeek-R1

**DeepSeek-R1** (January 2025) demonstrated that strong reasoning capabilities can be achieved through **pure reinforcement learning** on mathematical and coding problems — without supervised fine-tuning on human-written reasoning traces.

### Training Pipeline

1. **Cold start**: A small amount of high-quality reasoning examples are used for initial SFT.
2. **GRPO training** (Group Relative Policy Optimization): The model generates multiple candidate solutions; those that match the ground truth receive positive reward. No process-level supervision — only outcome signals.
3. **Rejection sampling SFT**: Collect high-quality reasoning traces from the RL-trained model and fine-tune on them.
4. **Final RL stage**: Further reinforcement learning on mixed reasoning and general tasks.

**Key finding:** Reasoning behaviors (backtracking, self-verification, "aha moments") **emerge naturally** from RL training with only correctness rewards — they are not injected by supervised examples.

DeepSeek-R1 matched or exceeded o1's performance on key benchmarks and was released as an open-weights model — a pivotal moment for accessible reasoning AI.

## Process Reward Models (PRMs)

Standard LLMs are trained with **outcome reward models** (ORMs) — reward is only given at the final answer. **Process Reward Models** supervise the **quality of each reasoning step**:

- Each intermediate step is scored for correctness and logical validity.
- Incorrect steps receive negative signal immediately rather than only at the end.

PRMs improve reasoning quality, especially for multi-step math where a single error early in a chain invalidates the final answer. They require **step-level labels** — much more expensive to collect than answer-level labels.

## Best-of-N and Search

Without full RL training, reasoning quality can be improved at inference time using **search strategies**:

- **Best-of-N**: Generate $N$ independent reasoning traces; select the one whose answer is most common (self-consistency) or highest-scored by a reward model.
- **Beam search over thoughts**: Expand the most promising partial reasoning paths rather than a single left-to-right trace.
- **Monte Carlo Tree Search (MCTS)**: Used experimentally for LLM reasoning — explore the space of reasoning paths with rollouts.

**Self-consistency** (Wang et al., 2022) — sampling multiple CoT paths and taking the majority vote answer — is a simple and effective technique that remains widely used.

## Thinking Tokens and Budgets

Reasoning models trade **tokens for accuracy**. Users can often configure a "thinking budget":

- Claude 3.7 Sonnet's **extended thinking** mode lets users set a maximum thinking token budget.
- OpenAI o3's "effort" level (low/medium/high) controls how much inference compute to allocate.
- Higher budgets → lower throughput and higher cost → better accuracy on hard tasks.

For routine tasks (summarization, simple Q&A), a high thinking budget adds latency without benefit. For complex tasks (math competitions, multi-step planning), it is critical.

## When Reasoning Models Excel

| Task Type | Standard LLM | Reasoning Model |
| --- | --- | --- |
| Creative writing | Excellent | Similar |
| Summarization | Excellent | Unnecessary overhead |
| Simple Q&A | Excellent | Similar |
| Multi-step math | Struggles | Excellent |
| Competition coding | Good | Excellent |
| Scientific reasoning | Good | Excellent |
| Complex planning | Moderate | Strong |
| Formal verification | Poor | Improved |

## Limitations

- **Cost and latency** — Hundreds of thinking tokens cost significantly more than a direct answer.
- **Overthinking** — Models sometimes generate unnecessarily long reasoning for simple tasks.
- **Verbosity** — Thinking traces can be meandering; not every token represents genuine useful deliberation.
- **Limited to "closed" problems** — Reasoning models improve on tasks with verifiable answers. Open-ended creative or social tasks show less benefit.
- **Hallucination in reasoning** — The thinking chain itself can contain plausible-sounding but incorrect steps.

## Leading Reasoning Models (2025)

| Model | Developer | Open Weights |
| --- | --- | --- |
| o3 / o4-mini | OpenAI | No |
| Claude 3.7 Sonnet | Anthropic | No |
| Gemini 2.5 Pro | Google | No |
| DeepSeek-R1 | DeepSeek | Yes |
| QwQ-32B | Alibaba (Qwen) | Yes |
| Phi-4-reasoning | Microsoft | Yes |

Reasoning models represent the most significant paradigm shift in LLM capability since instruction tuning — moving from models that retrieve knowledge to models that actively think through problems.
