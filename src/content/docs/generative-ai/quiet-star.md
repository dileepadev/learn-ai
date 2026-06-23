---
title: "Quiet-STaR: Teaching Language Models to Think Before Speaking"
description: Explore Quiet-STaR, a generalization of the Self-Taught Reasoner (STaR) that trains LLMs to generate hidden rationales (thoughts) at every token step to improve downstream reasoning.
---

Standard Autoregressive Language Models predict the next token sequentially: $P(x_{t} | x_{<t})$. While highly efficient, this forces the model to generate words without planning or looking ahead, which degrades performance on complex reasoning tasks.

**Quiet-STaR** (Self-Taught Reasoner on Quiet thoughts) is a method that trains language models to generate **hidden rationales (thoughts)** before predicting the next token. Unlike standard STaR, which requires structured dataset prompts with explicit step-by-step reasoning steps, Quiet-STaR operates on arbitrary, unstructured text by generating thoughts "quietly" in the background at every token step.

---

## The Core Concept: Hidden Rationales

In human speech, we often pause to think before answering a hard question. Quiet-STaR models this behavior mathematically. 

For each token $x_i$ in a sequence, the model:
1. Generates a "thought" sequence $T_i = (t_1, t_2, \dots, t_M)$ of length $M$.
2. Predicts the next actual token $x_{i+1}$ conditioned on both the history $x_{\le i}$ and the generated thought $T_i$:

$$P(x_{i+1} | x_{\le i}, T_i)$$

3. Uses a reinforcement learning algorithm to reward thoughts that make the prediction of the actual next token $x_{i+1}$ more likely.

```
Input Tokens: "The climate of the region is arid, [THOUGHT: arid means dry, no rain] so..."
```
During training, the thought is inserted before the word "so" to help the model predict "so" and subsequent tokens. During inference, these thoughts can be generated in a hidden scratchpad (hence "quiet") and omitted from the final user-facing text.

---

## The Quiet-STaR Architecture

Quiet-STaR introduces three main additions to standard training pipelines:

### 1. Parallel Thought Generation
Using a special attention mask and formatting tokens (e.g., `<thought>` and `</thought>`), the model generates $M$ thought tokens at every single token position in the input sequence. This is done efficiently in parallel using customized caching.

### 2. The Mixing Head
Because some tokens do not benefit from thinking (e.g., predicting punctuation or common transitions), the model features a learnable **mixing head**. This head outputs a weight $w \in [0, 1]$ to linearly combine the prediction logit calculated *without* the thought and the prediction logit calculated *with* the thought:

$$P_{\text{final}}(x_{i+1}) = (1 - w) \cdot P(x_{i+1} | x_{\le i}) + w \cdot P(x_{i+1} | x_{\le i}, T_i)$$

### 3. Policy Gradient Reinforcement Learning
To train the model to think constructively, Quiet-STaR uses REINFORCE. The reward for a thought $T_i$ is defined as the reduction in loss (improvement in perplexity) on the subsequent tokens:

$$R = -\log P_{\text{final}}(x_{i+1:i+k} | T_i) - (-\log P_{\text{base}}(x_{i+1:i+k}))$$

To stabilize training, the reward is contrasted against a baseline (using the average loss across multiple rolled-out thoughts). Thoughts that help the model predict the future text receive positive gradients, while distracting thoughts are penalized.

---

## Key Achievements of Quiet-STaR

- **Zero-Shot Reasoning:** Quiet-STaR significantly improves zero-shot performance on difficult reasoning benchmarks (like GSM8K and CommonsenseQA) without requiring fine-tuning on reasoning-specific datasets.
- **Unsupervised Learning:** It can learn how to think from raw, unlabeled internet text, as the reward signal is derived entirely from predicting the next token in the text.
- **Dynamic Reasoning Budget:** The length of the thoughts $M$ can be scaled up or down at inference time depending on the difficulty of the prompt and the available compute budget.

---

## Why Quiet-STaR Matters for the Future

Quiet-STaR represents a shift towards **test-time compute scaling**. Instead of making models larger, we can let them spend more compute during inference—thinking through multiple reasoning paths in a hidden scratchpad before delivering the optimal output.
