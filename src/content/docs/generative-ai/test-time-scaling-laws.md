---
title: Test-Time Scaling Laws
description: How scaling compute at inference time — through chain-of-thought, search, verifiers, and multi-step reasoning — yields predictable performance gains governed by scaling laws analogous to those discovered during pretraining. Covers compute-optimal inference, process reward models, Monte Carlo tree search for LLMs, and practical tradeoffs.
---

**Test-time scaling** refers to the systematic improvement in language model performance obtained by investing additional compute *during inference*, rather than during training. The discovery that many reasoning tasks benefit from extended thinking — and that this benefit follows predictable power laws — has shifted the field's intuition: a smaller model given more inference compute can outperform a much larger model on challenging reasoning benchmarks.

## The Pretraining Scaling Law Analogy

Kaplan et al. (2020) established that pretraining loss decreases as a smooth power law in compute, dataset size, and parameter count. This insight — that simply scaling along any axis yields reliable returns — transformed how the field allocates resources.

**Test-time scaling laws** extend this logic to inference. Snell et al. (2024) and subsequent work showed that for reasoning tasks:

$$\text{Performance} \propto C_{\text{inference}}^{\alpha}$$

where $C_{\text{inference}}$ is inference compute (measured in FLOPs or tokens generated) and $\alpha$ depends on the task and scaling strategy. This means:

- Doubling inference compute yields a predictable, consistent performance gain.
- The slope $\alpha$ varies: verification-based search scales more steeply than sequential chain-of-thought alone.
- There is a **compute-optimal frontier**: for a fixed inference budget, some model size and generation strategy combination is optimal.

## Sequential vs. Parallel Inference Compute

Two broad strategies for spending inference compute:

### Sequential Scaling (Longer Thinking)

The model generates a longer chain of thought before producing a final answer. This is implicit in models like OpenAI o1 and DeepSeek-R1, where "thinking tokens" are generated before the visible response. Sequential scaling benefits:

- Tasks where the reasoning path matters (mathematics, code debugging, multi-step planning)
- Problems where early reasoning steps condition later ones (backchaining, hypothesis refinement)

**Limitations**: Sequential generation is fundamentally sequential — each token depends on all previous ones. Wall-clock time scales linearly with token count. The model's inherent reasoning capability bounds how much extended thinking helps.

### Parallel Scaling (Best-of-N / Search)

Generate $N$ candidate answers independently and select the best using a **verifier** or **reward model**. Performance improves because:

$$P(\text{at least one correct among } N) = 1 - (1 - p)^N$$

where $p$ is the per-sample success probability. As $N$ increases, the probability of finding a correct answer approaches 1 — even when per-sample accuracy is modest.

**Key insight**: Parallel scaling is embarrassingly parallel — all $N$ generations can run simultaneously on independent hardware. For latency-tolerant applications, this is highly efficient.

## Verifiers and Process Reward Models

The bottleneck for parallel scaling is the **verifier** — the component that selects the best candidate from $N$ generated solutions. Verifier quality is critical: a poor verifier wastes the benefit of generating diverse candidates.

### Outcome Reward Models (ORMs)

An ORM scores a complete solution $(x, y)$ as correct or incorrect. ORMs are easy to train where ground truth is available (math problems with known answers) but:

- Provide no signal about *which step* in a multi-step solution was wrong.
- Reward hacking: models learn to produce solutions that fool the ORM without being genuinely correct.

### Process Reward Models (PRMs)

A PRM assigns a reward score to **each step** in a chain-of-thought reasoning trace. Introduced in Lightman et al. (2023) ("Let's Verify Step by Step"), PRMs:

- Enable fine-grained credit assignment.
- Guide tree search by scoring partial solutions, enabling pruning of unpromising branches.
- Are harder to train: require step-level labels, which demand expensive human annotation or LLM-based automatic labeling.

Training data for PRMs can be generated via **process supervision** (human annotators marking each step as correct/incorrect) or **Monte Carlo estimation** — sampling many completions from each step and computing the empirical success rate as a proxy for step-level correctness.

## Tree Search Methods for LLMs

Treating LLM generation as a tree search problem — where each node is a reasoning state and each edge is a generated token or reasoning step — enables systematic exploration of the solution space.

### Beam Search with Verifier Scoring

The simplest tree search: maintain a beam of $k$ partial solutions, expand each, score expansions with a PRM, and keep the top $k$. More principled than greedy decoding but still depth-first in character.

### Monte Carlo Tree Search (MCTS) for LLMs

**MCTS** balances exploration (trying novel reasoning branches) and exploitation (extending promising ones) via the UCB1 formula:

$$\text{UCT}(s, a) = Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

Applied to LLM reasoning:

- **State**: The current reasoning prefix.
- **Action**: Generating the next reasoning step.
- **Value function**: PRM score or Monte Carlo rollout success rate.
- **Policy**: The LLM itself, used to propose candidate next steps.

MCTS finds correct solutions on competition mathematics problems where beam search fails, because it can backtrack from dead-end reasoning branches and try alternative approaches. **AlphaCode 2** and similar systems use MCTS-style search over code generation candidates.

### Divide and Conquer / Hierarchical Search

Complex problems are decomposed into subproblems, each solved by an LLM sub-agent. The decomposition tree is itself generated by the LLM and scored by a high-level verifier. This enables tackling problems whose solution length would exceed the model's effective context window for monolithic generation.

## Compute-Optimal Inference Frontier

Given a fixed inference compute budget $C$, how should one allocate it? Key tradeoffs:

| Strategy | Parameters | Compute per token | #Tokens | Parallelism |
| --- | --- | --- | --- | --- |
| Large model, greedy | High | High | Few | None |
| Small model, best-of-N | Low | Low | Many | Full |
| Small model, MCTS | Low | Low | Many (branching) | Partial |
| Medium model, long CoT | Medium | Medium | Long chain | None |

**Empirical finding** (Snell et al., 2024): For tasks amenable to verification, smaller models with more inference compute can match or exceed larger models with less compute — but only if the verifier quality is high. A weak verifier (ORM with 70% accuracy) limits the benefit of parallel scaling.

The **compute-optimal inference curve** — analogous to the Chinchilla optimal training curve — shows that for each capability level, there is an optimal model size and inference compute allocation.

## Thinking Tokens and Latent Reasoning

Models like **o1** and **DeepSeek-R1** are trained end-to-end to produce "thinking tokens" — extended intermediate reasoning that may include false starts, self-corrections, and exploratory paths before committing to a final answer. This is distinct from externally-imposed chain-of-thought:

- **Internalized CoT**: The model learns when and how much to think through RLHF/GRPO training, rather than following a fixed prompt template.
- **Budget forcing**: At inference time, operators can constrain thinking token counts to trade accuracy for latency.
- **Emergent behaviors**: Models trained with extended thinking budgets develop strategies like backtracking, re-reading the problem, and checking intermediate computations — behaviors that emerge from training reward rather than being explicitly taught.

## Self-Consistency and Universal Self-Consistency

**Self-consistency** (Wang et al., 2022) generates $N$ chain-of-thought solutions and takes a majority vote over final answers. It requires no trained verifier — the model itself is the "verifier" through diversity of its outputs. Self-consistency provides reliable gains on math and multi-step reasoning and is easy to implement.

**Universal self-consistency** extends this to open-ended generation where answers are not directly comparable (e.g., free-form text summaries). An LLM judge selects the most self-consistent response.

## Scaling Laws Parameters: What Determines $\alpha$?

The slope $\alpha$ in the test-time scaling law depends on:

- **Task verifiability**: Fully verifiable tasks (competition math, code with test cases) scale steeply; subjective tasks plateau quickly.
- **Model capability**: A model that cannot produce correct solutions even with unlimited tries has $\alpha \approx 0$.
- **Verifier quality**: Higher-quality verifiers (PRMs vs. ORMs) yield steeper scaling.
- **Diversity of generations**: If the model always generates the same incorrect solution, more samples do not help. Temperature tuning and diverse prompting strategies are needed to ensure variation.

## Practical Implications

**For practitioners**:

- Before scaling model size, consider scaling inference compute — especially for reasoning-intensive tasks.
- Best-of-N with a simple ORM is often a high-return, low-implementation-cost first step.
- PRM training is expensive but unlocks significantly steeper scaling — worth the investment for high-stakes applications.
- Monitor the efficiency frontier: cloud inference costs scale linearly with tokens; gains may not.

**For researchers**:

- Studying test-time scaling laws for different domains (code, math, scientific reasoning, planning) remains an open area.
- Understanding the relationship between pretraining compute allocation and test-time scaling efficiency is an active frontier.
- Developing better automatic process supervision (removing the need for expensive human step labels) is a key bottleneck.

Test-time scaling has transformed what it means to deploy an AI system. The boundary between model capability and inference strategy has blurred — a well-designed inference-time compute allocation can dramatically expand the effective frontier of what a given model can achieve.
