---
title: Process Reward Models
description: Learn how Process Reward Models (PRMs) provide step-level supervision for multi-step reasoning — covering PRM vs ORM distinctions, Math-Shepherd's automatic labeling, MCTS-based data collection, best-of-N reranking, and the role of PRMs in test-time compute scaling.
---

Large language models reason through complex problems by generating multi-step chains of thought. **Outcome Reward Models (ORMs)** evaluate only the final answer — correct or not. **Process Reward Models (PRMs)** evaluate each individual reasoning step, providing fine-grained supervision that identifies exactly where reasoning goes wrong.

The difference is consequential: an ORM scores a chain of thought that reaches the wrong answer at step 12 out of 14 identically to a chain that fails at step 1 — no signal about which steps were good. A PRM assigns per-step scores, enabling targeted identification of errors, more effective filtering of candidate solutions, and richer training signal for reinforcement learning.

## Outcome vs. Process Supervision

Consider a math problem whose correct solution requires 8 algebraic steps. An ORM assigns a binary label (1 for a correct final answer, 0 for incorrect) to the entire chain. A PRM assigns a binary or continuous label to each step $s_t$ indicating whether the reasoning up to that step is correct:

$$\text{ORM}: r(x, c_1, \ldots, c_T) \in \{0, 1\}$$

$$\text{PRM}: r(x, c_1, \ldots, c_t) \in [0, 1] \quad \forall t = 1, \ldots, T$$

where $x$ is the problem and $c_t$ is the $t$-th chain-of-thought step.

The PRM's score for a complete solution is typically the minimum step score or the product of step scores:

$$\text{PRM}_\text{min}(x, c_{1:T}) = \min_t r(x, c_{1:t})$$
$$\text{PRM}_\text{prod}(x, c_{1:T}) = \prod_t r(x, c_{1:t})$$

The minimum is more conservative (a single bad step invalidates the chain); the product is a soft version that considers average quality.

## Human-Labeled PRMs: PRM800K

OpenAI's **PRM800K** dataset (Lightman et al., 2023) contains 800,000 step-level labels for solutions to MATH dataset problems, collected from human annotators who labeled each step as:

- **Positive** (+): this step is correct and follows from the previous context.
- **Neutral** (0): this step is mathematically correct but not helpful.
- **Negative** (−): this step contains an error.

A PRM trained on PRM800K achieved significantly better best-of-N selection than an ORM on MATH500, demonstrating that step-level supervision transfers to accurate solution reranking. The key finding was that PRMs trained on human step labels generalized better than ORMs trained on more solution-level data.

## Automatic Labeling: Math-Shepherd

Human labeling at the step level is expensive — PRM800K required substantial human effort for 800K labels. **Math-Shepherd** (Wang et al., 2023) introduces an automatic process supervision approach using **Monte Carlo rollouts**.

For each intermediate step $c_{1:t}$, Math-Shepherd estimates the probability that a correct final answer can still be reached:

1. From state $(x, c_{1:t})$, sample $K$ completions $c_{t+1:T}^{(1)}, \ldots, c_{t+1:T}^{(K)}$ using the policy model.
1. Check each completion against the ground-truth answer.
1. Assign a step score equal to the empirical success rate: $r(x, c_{1:t}) = \frac{1}{K}\sum_k \mathbf{1}[\text{answer}_k = \text{correct}]$.

This is the **value function** of the chain-of-thought MDP: $V(s_t) = P(\text{correct final answer} \mid x, c_{1:t})$. Steps that are recoverable (correct answer still achievable) get high scores; steps that have "gone off the rails" get low scores.

Math-Shepherd labels are noisier than human labels (MC estimation error), but are infinitely scalable: given any base model and any problem set with verifiable answers, Math-Shepherd automatically generates step-level labels.

## MCTS-Based PRM Data Collection

A more principled approach to automatic labeling uses **Monte Carlo Tree Search (MCTS)** to both collect high-quality reasoning traces and assign step-level credit simultaneously.

In the MCTS-PRM framework:

- Each node in the search tree is a partial solution $(x, c_{1:t})$.
- **Rollout policy**: the LLM generates random completions from each node to estimate $V(s_t)$.
- **UCB selection**: child nodes are selected by $\text{UCB}(s_t) = V(s_t) + \sqrt{\frac{2\ln N(s_\text{parent})}{N(s_t)}}$ balancing exploitation of high-value steps and exploration of less-visited steps.
- **Backpropagation**: after reaching a terminal state, the outcome reward is backpropagated to update $V$ estimates for all ancestor steps.

After MCTS, every node in the search tree has an estimated value, providing natural PRM training labels. MCTS simultaneously finds better solutions (useful for supervised fine-tuning data) and more accurately labels the quality of each step (useful for PRM training).

## Best-of-N Selection with PRMs

The most direct application of PRMs is **best-of-N reranking**:

1. Sample $N$ candidate solutions from the policy model.
1. Score each complete solution with the PRM.
1. Return the highest-scoring solution.

Scaling best-of-N with a PRM is significantly more efficient than with an ORM or self-consistency voting:

- **Self-consistency** (majority vote): requires $N$ solutions and selects the most common final answer — ignores solution quality.
- **ORM reranking**: selects the solution with the highest final-answer confidence — no step-level signal.
- **PRM reranking**: selects the solution with the highest minimum (or product) step score — identifies solutions that are correct at every step.

Lightman et al. (2023) showed that PRM-based best-of-1860 on MATH reached the performance of a much larger model trained with ORM-guided RL, demonstrating that test-time compute (more samples + better reranking) can substitute for training compute.

## PRMs in Reinforcement Learning

Beyond reranking, PRMs serve as **dense reward signals** for RL fine-tuning:

$$r_t = r_\text{PRM}(x, c_{1:t}) - r_\text{PRM}(x, c_{1:t-1})$$

This differential reward (the improvement in PRM score from step $t-1$ to step $t$) provides a reward signal at each time step rather than only at the end of the episode — the classic advantage of dense vs. sparse rewards.

However, **reward hacking** is a serious concern: the LLM can learn to generate text that increases PRM scores without genuinely solving the problem. This occurs because the PRM is itself imperfect — it was trained on a finite dataset and can be fooled by surface patterns. RL with PRM rewards requires careful:

- **KL divergence penalties** to prevent the policy from drifting too far from the reference model.
- **PRM robustness evaluation**: checking whether PRM scores correlate with ground-truth correctness throughout training.
- **Iterative PRM retraining**: updating the PRM as the policy changes to prevent distributional shift.

## Implicit PRMs from Value Functions

An alternative to explicitly training a PRM is using the **value function** of the RL policy itself as an implicit process reward. In **Math-RLVR** and related frameworks:

- The critic network $V(s_t)$ estimates expected future return from each step.
- The advantage $A_t = r_T + \gamma V(s_{t+1}) - V(s_t)$ provides a step-level credit signal.
- This is mathematically equivalent to a PRM trained with MC rollouts from the same policy.

The implicit PRM has a key advantage: it automatically updates as the policy improves, avoiding the distributional shift problem of a fixed PRM.

## Step-Level Credit Assignment Challenges

Training PRMs faces fundamental **credit assignment** challenges:

- **Long chains**: a 20-step solution has 20 binary labels. The PRM must learn which individual step introduced an error.
- **Delayed errors**: an incorrect early step may not manifest as a wrong answer until several steps later. MC rollouts partially address this by averaging over many completions.
- **Step granularity**: what counts as a "step"? A sentence? A line of algebra? A code block? Different granularities yield different supervision signals.
- **Multi-path correctness**: multiple valid solution paths exist. A step that diverges from the "standard" path may still be correct — the PRM must not penalize creative but valid approaches.

## PRMs for Code and Science

PRMs extend beyond mathematical reasoning:

- **Code generation**: PRM scores each intermediate reasoning step in chain-of-thought code synthesis. A step that introduces a variable that will cause a type error gets a low score before the error even occurs — if the PRM has learned the relevant patterns.
- **Scientific reasoning**: multi-hop question answering benefits from PRMs that evaluate whether each retrieval and inference step is correct, rather than only the final answer.
- **Theorem proving**: interactive theorem provers like Lean 4 naturally provide step-level verification (each tactic either succeeds or fails), enabling automatic PRM labeling with zero human effort for formal mathematics.

## Summary

Process Reward Models provide fine-grained step-level supervision for multi-step reasoning, addressing the credit assignment problem that makes ORM training inefficient for long chain-of-thought solutions. PRM800K demonstrated the value of human step labels; Math-Shepherd made step-level labeling scalable through MC rollouts; MCTS frameworks provide both better solution data and more accurate step-level values simultaneously. PRMs power best-of-N selection as a test-time compute scaling strategy and serve as dense reward signals for RL fine-tuning — though reward hacking requires careful mitigation. The combination of PRMs with MCTS and iterative RL training represents the current frontier of test-time and training-time compute scaling for reasoning models.
