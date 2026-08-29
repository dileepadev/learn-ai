---
title: Circuits and Induction Heads in Language Models
description: Explore mechanistic interpretability in transformers, two-layer attention circuits, induction heads that drive in-context learning, and computational subgraphs.
---

Why do large language models possess such extraordinary **in-context learning (ICL)** capabilities? When shown a few examples in a prompt, how does a model dynamically recognize and continue arbitrary patterns, code formatting, or novel factual mappings without updating its weights?

In 2021–2022, researchers at Anthropic (Elhage et al., Olsson et al.) uncovered a breakthrough answer through the lens of **Mechanistic Interpretability**. By treating the transformer as an electrical circuit that can be reverse-engineered, they discovered that in-context learning is primarily driven by a specific sub-circuit of collaborating attention heads known as **Induction Heads**.

---

## The Transformer Circuits Framework

Traditional deep learning analysis treats neural networks as black boxes. The **Circuits Framework** decomposes transformers into human-understandable computational graphs by analyzing:

1. **The Residual Stream as a Shared Bus:** Each attention layer and MLP block reads information from the residual stream and adds its output vectors back into the stream via vector addition ($x_{l+1} = x_l + \text{Attn}(x_l) + \text{MLP}(x_l)$).
2. **Head Composition:** Attention heads do not act in isolation. An attention head in Layer $L_2$ can read the output written into the residual stream by an attention head in Layer $L_1$.

```
Residual Stream ─────────────────────────────────────────────────────────────► Logits
     ▲                           ▲                           ▲
     │ Reads/Writes              │ Reads/Writes              │ Reads/Writes
┌────┴───────────────┐     ┌─────┴──────────────┐     ┌──────┴──────────────┐
│  Layer 0 Attention │     │  Layer 1 Attention │     │  Layer 2 Attention  │
│  (Previous Token)  │────►│  (Induction Head)  │────►│  (Classifier/Logit) │
└────────────────────┘     └────────────────────┘     └─────────────────────┘
```

---

## What is an Induction Head?

An **Induction Head** is an attention head in a two-layer (or deeper) circuit that performs sequence completion by looking back into the context for historical occurrences of the current token and copying the token that followed it.

### The Canonical Induction Pattern: $[A][B] \dots [A] \to [B]$

Imagine an LLM processing text containing a repeated sequence:

$$\dots [A] \; [B] \dots \dots [A]$$

When the model reaches the second token $[A]$, the induction head attends to token $[B]$ (the token that came immediately after the earlier $[A]$) and promotes $[B]$ in the output vocabulary logits.

```
Context Stream:   ... [Harry] [Potter] ... [Harry] ──► [ Predicts "Potter" ]
                        ▲        ▲           ▲
                        │        │           │
Step 1 (Layer 0):       └── "Next Token" ────┘ (Previous-Token Head writes 'Harry' into 'Potter')
Step 2 (Layer 1):       Induction Head at second [Harry] attends to [Potter] via Key-Query match!
```

---

## How the Two-Head Circuit Works Mechanistically

An induction head cannot exist in a single-layer attention-only model. It requires a **two-head composition** across at least two consecutive layers:

### 1. Previous-Token Head (Layer 0)
- The attention head in Layer 0 attends to token $t-1$ relative to position $t$.
- At position of token $[B]$, this head reads the vector for token $[A]$ and writes that information into the residual stream at token $[B]$'s position.
- As a result, token $[B]$ now carries the contextual information: *"I was preceded by token $[A]$"*.

### 2. Induction Head (Layer 1)
- The induction head at the second occurrence of $[A]$ forms its **Query vector** from current token $[A]$: $q = W_Q x_{[A]}$.
- Its **Key matrix** $W_K$ is specifically tuned to recognize the *"I was preceded by $[A]$"* feature that Layer 0 stored at position $[B]$: $k = W_K x_{[B]}$.
- Because the Query for $[A]$ matches the Key for $[B]$, the attention weight spikes sharply on token $[B]$:

$$\text{AttentionScore} = \frac{(W_Q x_{[A]})^\top (W_K x_{[B]})}{\sqrt{d_k}} \gg 0$$

- The **Value matrix** $W_V$ and **Output projection** $W_O$ copy the identity of token $[B]$ into the residual stream, directly increasing the logit probability that the next generated token will be $[B]$.

---

## The "Induction Bump" Phase Change in Training

During pretraining of language models, Olsson et al. discovered a striking phenomenon:
1. Early in training, models lack induction heads and rely entirely on unigram and bigram frequency statistics.
2. At a specific point during pretraining (typically between $2.5\times 10^8$ and $10^9$ tokens), induction heads form suddenly and simultaneously across multiple layers.
3. This formation coincides with a sharp, discontinuous drop in training loss—the **"Induction Bump"**—and marks the precise moment when the model acquires **in-context learning capabilities**.

```
Loss
 ▲
 │   Without Induction Circuits (Flat Learning)
 │   \
 │    \   "Induction Bump" (Phase Change)
 │     \   Suddenly Forms Induction Heads!
 │      └───► Sharp Loss Drop & In-Context Learning Emerges
 └──────────────────────────────────────────────────────────► Training Tokens
```

---

## Detecting Induction Heads via Activation Patching

How do interpretability researchers prove that a specific head is acting as an induction head? Through **Causal Activation Patching**:

1. **Clean Run:** Run the model on a repeated token sequence (e.g., random sequences like `X Y Z ... X Y Z`) and record the output logit for the correct completion.
2. **Corrupted Run:** Run the model on a non-repeating sequence (`X Y Z ... A B C`).
3. **Intervention:** During the corrupted run, pause execution at Layer $L$, head $H$, and overwrite its activation tensor with the clean run's activation.
4. If restoring that single head's activation restores the model's ability to complete the pattern, that head is confirmed to be part of the causal induction circuit.

---

## Broader Implications

- **Translation and Few-Shot Learning:** Induction heads generalize beyond literal token copying. In multilingual models, induction heads map foreign words to English translations (e.g., `[chien][dog] ... [chat] -> [cat]`).
- **Prompt Injection Vulnerability:** Many prompt injection and jailbreak attacks exploit induction head circuits by tricking the model into copying malicious instructions disguised as preceding tokens.
- **Circuit Pruning & Model Compression:** Understanding essential induction circuits allows researchers to prune redundant heads without destroying in-context reasoning.
