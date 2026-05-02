---
title: Emergent Abilities in Large Language Models
description: Explore emergent abilities in large language models — capabilities that appear unpredictably at scale, including few-shot learning, chain-of-thought reasoning, and in-context arithmetic. Understand the scientific debates around measurement artifacts, phase transitions, and what emergence tells us about scaling laws and future AI development.
---

**Emergent abilities** in large language models are capabilities that are absent or near-random in smaller models but appear — sometimes abruptly — in larger ones. The term was popularized by Wei et al. (Google Brain, 2022) who documented over 100 tasks where model performance was effectively zero up to a certain scale threshold, then jumped sharply as models grew from billions to tens of billions of parameters.

Emergence in LLMs is simultaneously one of the most fascinating and most contested phenomena in AI research. It challenges the intuition that bigger-is-smoother-is-better, raises profound questions about predictability and safety, and touches fundamental issues of how we measure AI capability.

## What "Emergent" Means

An ability is considered **emergent** if it satisfies two conditions:

1. **Unpredictability from smaller scale**: Performance at smaller scale does not allow predicting when the ability will appear.
2. **Qualitative change**: The ability appears to be a new capability, not merely a quantitative improvement in an existing one.

This is borrowed from physics and complex systems: water's ability to wet surfaces emerges from collective molecular behavior, not from individual water molecules. Similarly, the hypothesis is that certain language capabilities emerge from the interaction of millions of learned associations, rather than being continuously learned parameter by parameter.

## Documented Emergent Behaviors

### Few-Shot Learning

The ability to perform a task from a handful of in-context examples — without any gradient updates — was largely absent in GPT-2 (1.5B parameters) and appeared dramatically in GPT-3 (175B). At smaller scales, adding examples to the prompt often had negligible or even negative effects.

### Chain-of-Thought Reasoning

Standard prompting on multi-step arithmetic and commonsense reasoning benchmarks showed near-flat scaling across 10M–10B parameters. Chain-of-thought prompting, however, only became effective above approximately 100B parameters — smaller models produced plausible-looking but incorrect intermediate reasoning steps.

```
Small model (7B) with CoT prompting:
"Q: If there are 3 cars with 4 wheels each, and 2 trucks with 8 wheels each,
   how many wheels are there in total?"
"A: Let me think step by step.
   Cars have 4 wheels. Trucks have more wheels.
   Total: 3 + 2 = 5 vehicles, so 5 × 4 = 20 wheels."  ← WRONG

Large model (100B+) with CoT prompting:
"A: Let me think step by step.
   Cars: 3 × 4 = 12 wheels.
   Trucks: 2 × 8 = 16 wheels.
   Total: 12 + 16 = 28 wheels."  ← CORRECT
```

### Other Documented Examples

- **Word unscrambling**: Below ~10B parameters, near-random. Above ~100B, models solve scrambled words reliably.
- **Multi-digit arithmetic**: Addition/multiplication of 4–5 digit numbers shows near-zero performance below certain scales, then jumps.
- **Analogical reasoning**: e.g., "A is to B as C is to ___" (IQ test-style) shows emergent behavior around 50–100B parameters.
- **Code execution simulation**: Mentally tracing the output of short Python programs improves dramatically at scale.
- **Calibration on known facts**: Knowing what you don't know (expressing appropriate uncertainty) shows emergent characteristics at large scales.

## The Measurement Controversy

A landmark rebuttal by Schaeffer, Miranda & Koyejo (Stanford, 2023) argued that **emergence is largely a measurement artifact**, not a fundamental property of the models:

**The core argument**: Many "emergent" abilities are only discontinuous because researchers measure them with **discontinuous metrics** (exact match: 0 or 1). If you instead measure a continuous proxy (e.g., log-probability of the correct token, rather than whether the final decoded string is correct), the improvement is smooth and predictable from smaller scales.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_emergence_artifact(
    param_counts: np.ndarray,
    alpha: float = 0.3
) -> dict:
    """
    Simulate how metric choice creates apparent emergence.
    
    Underlying capability grows smoothly: P(correct token) ∝ N^alpha
    But binary exact-match only triggers when cumulative probability
    exceeds a threshold — creating a discontinuous-looking jump.
    """
    # Underlying smooth capability (e.g., log-probability of correct sequence)
    smooth_capability = param_counts ** alpha / (param_counts[-1] ** alpha)
    
    # Binary exact-match: 1 only if *all* tokens in sequence are correct
    # For a 10-token sequence, P(exact match) = P(single token)^10
    sequence_length = 10
    exact_match = smooth_capability ** sequence_length
    
    # The jump appears because P(single token) must reach ~0.9 before
    # P(all 10 correct) becomes non-negligible
    threshold_idx = np.argmax(exact_match > 0.01)
    
    return {
        "param_counts": param_counts,
        "smooth_capability": smooth_capability,
        "exact_match": exact_match,
        "apparent_emergence_at": param_counts[threshold_idx]
    }

param_counts = np.logspace(8, 11, 100)  # 1B to 100B
result = simulate_emergence_artifact(param_counts, alpha=0.3)

# The smooth capability line shows no emergence
# The exact-match line shows a dramatic jump — but it's an artifact of the metric
```

This doesn't fully explain all emergent phenomena — some capabilities seem to require more nuanced explanations — but it significantly narrows the scope of genuine phase transitions.

## Scaling Laws and Predictability

**Chinchilla scaling laws** (Hoffmann et al., DeepMind, 2022) showed that loss decreases smoothly and predictably with compute. But loss is a continuous aggregate metric. Individual task performance can behave differently:

- **Continuous tasks** (translation quality, summarization ROUGE): Smooth improvement with scale.
- **Threshold tasks** (exact-match arithmetic, multi-step logic): Discontinuous improvement as model capability crosses a task-specific threshold.

The key insight: loss predicts average behavior, but individual tasks can have high variance around that average, making them appear to "emerge" from certain model sizes.

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

Where $N$ is model parameters, $D$ is training tokens, and $L_\infty$ is the irreducible loss. This equation fits smoothly — but whether a specific task is solvable given loss $L$ depends on the task's difficulty distribution.

## Grokking: A Related Phenomenon

**Grokking** (Power et al., OpenAI, 2022) describes a related emergent phenomenon in smaller models: after a model appears to have overfit (training loss near zero, validation loss near chance), continued training suddenly causes a dramatic jump in generalization performance — sometimes thousands of steps later.

```python
import torch
import torch.nn as nn
import numpy as np

def train_modular_arithmetic(p: int = 97, hidden: int = 128,
                              n_steps: int = 50000) -> list[dict]:
    """
    Train a tiny transformer to learn modular addition: (a + b) mod p.
    Grokking predicts: model first memorizes, then (much later) generalizes.
    """
    # Generate all a+b mod p pairs
    a = torch.arange(p).repeat(p)
    b = torch.arange(p).repeat_interleave(p)
    c = (a + b) % p

    # 50% train / 50% test split
    idx = torch.randperm(len(a))
    n_train = len(a) // 2
    train_mask = idx[:n_train]
    test_mask = idx[n_train:]

    # Simple MLP
    model = nn.Sequential(
        nn.Embedding(p * 2, hidden),  # a and b are embedded separately
        nn.Flatten(),
        nn.Linear(hidden * 2, hidden), nn.ReLU(),
        nn.Linear(hidden, p)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    history = []

    for step in range(n_steps):
        # Train step
        model.train()
        inputs = torch.stack([a[train_mask], b[train_mask] + p], dim=1)
        logits = model(inputs)
        loss = criterion(logits, c[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                test_inputs = torch.stack([a[test_mask], b[test_mask] + p], dim=1)
                test_logits = model(test_inputs)
                test_acc = (test_logits.argmax(1) == c[test_mask]).float().mean().item()
                history.append({"step": step, "test_acc": test_acc})
                if step % 5000 == 0:
                    print(f"Step {step}: test_acc = {test_acc:.3f}")
    return history

# Expect: test_acc ≈ 0.0 for thousands of steps, then sudden jump to ~1.0
```

## Implications for AI Safety and Forecasting

Emergence has several important implications:

**Safety**: If dangerous capabilities can appear suddenly at scale, evals run on smaller models may provide a false sense of safety. A model that cannot break cryptographic algorithms at 30B parameters might acquire that capability at 300B — with no warning from the smooth loss curve.

**Forecasting**: Emergence complicates capability prediction. Organizations that rely on extrapolating from smaller-scale evals may be surprised when qualitatively new behaviors appear in large production models.

**Interpretability**: Emergent capabilities may correspond to qualitative changes in internal representations — circuits that form suddenly once supporting sub-circuits are sufficiently developed. Mechanistic interpretability research (identifying specific circuits responsible for specific behaviors) is one approach to understanding this.

## Current Scientific Consensus

The debate remains active. The current best understanding is:

- Some apparent emergence is **metric artifact** (discontinuous metrics applied to smooth underlying capability improvements).
- Some emergence reflects **task difficulty thresholds** — tasks that require a certain minimum capability level to solve at all.
- Some emergence may reflect genuine **phase transitions** in learned representations, analogous to physical phase transitions.
- **Grokking** is well-documented and not easily explained by metric artifacts, suggesting genuine non-linear learning dynamics exist.

Regardless of mechanism, the empirical pattern is real: certain capabilities of large language models are difficult to predict from smaller-scale experiments alone — a fact that motivates both careful evaluation methodology and ongoing research into the foundations of large-scale learning.
