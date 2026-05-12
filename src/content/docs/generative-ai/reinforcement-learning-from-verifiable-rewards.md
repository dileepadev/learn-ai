---
title: Reinforcement Learning from Verifiable Rewards
description: Explore Reinforcement Learning from Verifiable Rewards (RLVR) — the training paradigm behind DeepSeek-R1 and similar reasoning models — covering verifiable reward signals, the GRPO algorithm, math and code verification, comparison with RLHF, and the scaling properties of process-verifiable feedback.
---

**Reinforcement Learning from Verifiable Rewards (RLVR)** is a training paradigm that uses binary or scalar reward signals derived from objective, automatically verifiable outcomes — such as mathematical correctness or code test passage — to improve language model reasoning through reinforcement learning. Unlike RLHF, which relies on a trained reward model that approximates human preferences, RLVR grounds the reward signal in ground-truth verification, eliminating a class of reward modeling errors and enabling higher-quality reasoning improvement.

RLVR gained wide attention in early 2025 with the release of DeepSeek-R1, which demonstrated that a 7B-parameter model trained primarily with verifiable rewards on mathematics and coding problems could match or exceed significantly larger models on reasoning benchmarks — without human-labeled chain-of-thought data.

## Motivation: Why Verifiable Rewards

### The Reward Model Bottleneck in RLHF

Standard RLHF (Reinforcement Learning from Human Feedback) trains a **reward model** (RM) on human preference annotations, then uses the RM as a surrogate for human judgment during RL training. This creates two sources of error:

- **Reward model inaccuracy**: the RM is a learned approximation. On complex reasoning tasks involving multi-step mathematics or intricate code logic, the RM may assign high rewards to plausible-sounding but incorrect outputs.
- **Reward hacking**: as RL training optimizes against the RM, the policy learns to exploit RM weaknesses, producing outputs that score well according to the RM but fail on the underlying task.

For tasks where the correct answer is verifiable — whether a proof step is valid, whether code passes unit tests, whether an arithmetic result is correct — a **verifier** can provide a perfectly accurate, non-hackable reward signal without the need for a learned RM.

### Verifiable vs. Non-Verifiable Domains

| Domain | Verifiable? | Verification Method |
| --- | --- | --- |
| Mathematical problem solving | Yes | Compare final answer to ground truth |
| Code generation | Yes | Execute against test cases |
| Logical puzzle solving | Yes | Formal satisfiability check |
| Creative writing | No | Requires human or RM judgment |
| Summarization quality | No | Requires human evaluation |
| Open-ended question answering | Partially | Depends on task format |

RLVR is ideally suited to the first three categories and can be extended to any domain where outcome correctness is decidable.

## The GRPO Algorithm

DeepSeek-R1 uses **Group Relative Policy Optimization (GRPO)**, a variant of PPO that avoids the need for a separate value (critic) model by estimating advantages from a group of policy-generated responses.

### Standard PPO in LLM Training

PPO updates the policy $\pi_\theta$ to maximize the expected reward $r$ while staying close to a reference policy $\pi_{\text{ref}}$:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(\rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$ is the probability ratio and $\hat{A}_t$ is the advantage estimate from the critic. For LLMs, the critic must be another large model, doubling GPU memory requirements.

### GRPO: Group-Based Advantage Estimation

GRPO eliminates the critic by sampling $G$ responses $\{o_1, \ldots, o_G\}$ from the current policy for each question $q$ and computing advantages from within-group reward statistics:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

The GRPO objective is:

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}_{q \sim \mathcal{D},\ \{o_i\} \sim \pi_\theta(\cdot|q)}\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i\right) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

Key properties:

- **No value model**: advantage estimates come from the reward distribution of sampled outputs, not a learned critic.
- **Memory efficiency**: halves GPU memory relative to PPO by removing the critic model.
- **Unbiased under verifiable rewards**: when rewards are binary (correct/incorrect), the within-group normalization cleanly separates successful from unsuccessful reasoning traces.

## Reward Signal Design

### Mathematics Rewards

For math problems with deterministic answers, the reward is:

$$r(o, a^*) = \begin{cases} 1 & \text{if } \text{parse\_answer}(o) = a^* \\ 0 & \text{otherwise} \end{cases}$$

where $a^*$ is the ground-truth answer and `parse_answer` extracts the final numerical or symbolic answer from the model's chain-of-thought output. Datasets used include MATH, GSM8K, AIME, and AMC competition problems.

**Format rewards** are also commonly applied alongside correctness rewards:

- +0.1 for correctly using a designated answer delimiter (e.g., `\boxed{...}`)
- −0.1 for excessively long or repetitive outputs

These lightweight format rewards encourage structured outputs without requiring human annotation.

### Code Execution Rewards

For code generation tasks, outputs are executed in a sandboxed environment against unit test cases:

$$r(o) = \frac{\text{tests passed}}{\text{total tests}}$$

This provides a dense reward signal (fraction of tests passed) rather than binary, giving the RL algorithm more gradient signal for partial credit. Sandboxing is essential to prevent code execution security issues.

### Process-Level Rewards vs. Outcome-Level Rewards

RLVR typically operates at the **outcome level** — the reward is assigned to the entire response based on whether the final answer is correct. **Process Reward Models (PRMs)** assign intermediate rewards at each reasoning step, which can accelerate learning when steps are individually verifiable. However, step-level verification is harder to automate and requires either formal systems (proof checkers) or learned verifiers that reintroduce reward model error.

## DeepSeek-R1: Training Pipeline

DeepSeek-R1's training proceeds in multiple stages:

### Stage 1: Cold-Start Fine-Tuning

A small set (~thousands) of high-quality long chain-of-thought (CoT) examples are used for supervised fine-tuning to initialize the policy. This avoids the instability of applying RLVR to a raw pretrained model with no CoT behavior.

### Stage 2: RLVR Training (DeepSeek-R1-Zero → DeepSeek-R1)

GRPO is applied with verifiable math and code rewards. The model learns to:

- Generate long, structured reasoning chains before producing a final answer.
- Self-correct within a single response when intermediate steps lead to contradictions.
- Apply reflection — explicitly reviewing earlier steps and revising them.

These behaviors emerge **without explicit supervision** of the reasoning process. The model discovers that longer, more careful reasoning chains tend to produce correct final answers, which GRPO reinforces.

### Stage 3: Distillation to Smaller Models

Reasoning traces generated by the large RLVR-trained model are distilled into smaller models (1.5B–14B parameters) via supervised fine-tuning. This produces compact models with strong reasoning abilities at much lower inference cost.

## Emergent Reasoning Behaviors

RLVR training on DeepSeek-R1-Zero revealed several reasoning behaviors that emerge without explicit instruction:

- **Extended thinking**: average response length increases from ~200 tokens at initialization to ~2000+ tokens after RLVR training, as the model learns that more reasoning leads to higher rewards.
- **Backtracking and self-correction**: the model generates phrases like "Wait, let me reconsider..." and revises intermediate steps, a behavior not present in the SFT model.
- **Exploration of alternative approaches**: when one approach fails, the model tries a different method within the same response.
- **Language mixing**: early RLVR training without cold-start CoT initialization produces code-switching (mixing languages within a response), which is suppressed by the cold-start stage.

## Comparison with RLHF and Other Approaches

| Aspect | RLVR | RLHF | SFT on CoT |
| --- | --- | --- | --- |
| Reward source | Ground-truth verifier | Learned reward model | Teacher forcing |
| Reward accuracy | Perfect (within task) | Approximate | N/A |
| Reward hacking risk | None | High | N/A |
| Applicable tasks | Verifiable domains only | Broad | Broad |
| Human labeling cost | Low (answer only) | High (pairwise preferences) | High (CoT traces) |
| Reasoning emergence | Yes (empirically) | Limited | Depends on data |

## Scaling Properties

RLVR shows favorable scaling behavior along two axes:

- **Inference compute scaling**: RLVR-trained models improve significantly with longer inference budgets. When given the option to "think longer" (more tokens at test time), accuracy on AIME and MATH500 increases monotonically — a property not observed in base models or standard SFT models.
- **Training data scaling**: performance improves consistently as more verifiable training problems are added, without the plateau effects seen in SFT on fixed datasets.

These properties suggest that RLVR is particularly well-suited for reasoning domains where problem difficulty and quantity can be scaled independently of human labeling effort.

## Limitations and Open Questions

- **Domain restriction**: RLVR requires verifiable ground truth. Most NLP tasks (translation, summarization, dialogue) do not have easily verifiable correct answers.
- **Test case quality for code**: code rewards depend on the quality and coverage of unit tests. Sparse test suites allow reward hacking by generating code that passes tests without solving the underlying problem.
- **Reward sparsity**: binary outcome rewards provide no gradient signal for near-correct answers. Early RLVR training can be slow to get off the ground if the policy rarely produces correct outputs.
- **Format sensitivity**: small changes in answer format (e.g., `3/4` vs. `0.75`) can cause incorrect reward assignment, requiring careful answer normalization.

## Summary

Reinforcement Learning from Verifiable Rewards replaces the learned reward model in RLHF with an objective verifier, eliminating reward modeling errors and reward hacking on tasks with decidable correct answers. Combined with GRPO — which removes the need for a separate critic model — RLVR provides a memory-efficient, highly accurate training signal for reasoning tasks. DeepSeek-R1 demonstrated that RLVR induces emergent reasoning behaviors (extended thinking, self-correction, reflection) without explicit process supervision, enabling small models to reach frontier-level performance on mathematical reasoning. RLVR represents a significant advance for verifiable domains and is likely to influence reasoning model training across the industry.
