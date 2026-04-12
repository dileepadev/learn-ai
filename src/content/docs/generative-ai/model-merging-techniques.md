---
title: AI Model Merging Techniques
description: Discover how model merging combines multiple fine-tuned models into a single powerful model without additional training, using techniques like SLERP, TIES, DARE, and Task Arithmetic.
---

Model merging is the practice of combining the weights of two or more separately fine-tuned models into a single model — without any additional training data or GPU compute. It has exploded in popularity as a cheap, effective way to create capable models that blend multiple skill sets.

## Why Merge Models?

Fine-tuning a large language model for every specialized task is expensive. Model merging offers:

- **Zero training cost:** No forward or backward passes required
- **Skill combination:** A math-tuned model and a coding-tuned model can be merged to get both capabilities
- **Open-source power:** Community-merged models on Hugging Face often outperform individually fine-tuned models
- **Privacy:** Merge without sharing proprietary training data

## Task Arithmetic: The Foundation

Task Arithmetic (Ilharco et al., 2022) introduced the concept of **task vectors** — the difference in weights between a fine-tuned model and its base:

$$\tau_\text{task} = \theta_\text{finetuned} - \theta_\text{base}$$

These vectors can be **added, subtracted, and scaled** arithmetically:

$$\theta_\text{merged} = \theta_\text{base} + \lambda_1 \tau_1 + \lambda_2 \tau_2 + \cdots$$

This allows intuitive operations like:

- **Adding a skill:** $\theta_\text{base} + \tau_\text{coding}$
- **Removing a behavior:** $\theta_\text{base} - \tau_\text{unsafe}$
- **Blending two specialists:** $\theta_\text{base} + 0.5\tau_\text{math} + 0.5\tau_\text{reasoning}$

## SLERP: Spherical Linear Interpolation

SLERP treats model weights as points on a high-dimensional sphere and interpolates along the **geodesic path** between two models:

$$\text{SLERP}(\theta_A, \theta_B, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\theta_A + \frac{\sin(t\Omega)}{\sin\Omega}\theta_B$$

where $\Omega = \arccos\left(\frac{\theta_A \cdot \theta_B}{|\theta_A||\theta_B|}\right)$ is the angle between the weight vectors.

**Advantages over linear interpolation:**

- Preserves the magnitude of weight vectors
- Avoids the "weight collapse" that can occur when averaging models trained in different directions
- Produces smoother transitions between model capabilities

SLERP is best for merging exactly **two models**; for three or more, TIES or DARE are preferred.

## TIES: Trim, Elect Sign, Disjoint Merge

TIES (Yadav et al., 2023) addresses a key failure mode of naive weight averaging: **sign conflicts**. When different fine-tuned models push a weight in opposite directions, averaging cancels both contributions.

TIES merges task vectors in three steps:

1. **Trim:** Keep only the top-$k$% of parameters by magnitude, zeroing the rest (reduce noise)
2. **Elect Sign:** For each parameter, take the majority sign across all task vectors
3. **Disjoint Merge:** Average only the task vectors that agree with the elected sign

$$\theta_\text{merged} = \theta_\text{base} + \lambda \cdot \text{TIES-merge}(\tau_1, \tau_2, \ldots)$$

TIES consistently outperforms simple task arithmetic on multi-task merges.

## DARE: Drop and Rescale

DARE (Yu et al., 2023) applies a **random dropout** to task vector parameters before merging:

1. **Drop:** Randomly zero out a fraction $p$ of the delta weights
2. **Rescale:** Multiply remaining weights by $\frac{1}{1-p}$ to maintain the expected magnitude

This reduces interference between models and allows merging many fine-tunes simultaneously. DARE can be combined with TIES (DARE-TIES) for even better results.

## Model Breadcrumbs

Model Breadcrumbs (Davari et al., 2023) applies sparsification more aggressively — zeroing out not only small-magnitude deltas but also **large outlier deltas**, keeping only the "goldilocks" middle range. This produces sparse task vectors that interfere minimally with other merged models.

## Frankenmerging: Layer-Wise Stacking

Beyond weight interpolation, **frankenmerging** (also called layer stacking or depth merging) creates new models by **reassembling layers from different models**:

```
Merged Model:
  Layers 0–12  →  from Model A
  Layers 13–24 →  from Model B
  Layers 25–32 →  from Model A
```

This exploits the observation that LLM layers are somewhat modular. Popular community merges on Hugging Face often use this technique to create models with more total layers than any individual source model.

## Practical Tools

| Tool | Description |
|---|---|
| `mergekit` | De-facto standard CLI/library for model merging (supports SLERP, TIES, DARE, Task Arithmetic, frankenmerge) |
| Hugging Face Hub | Community hub for sharing merged models |
| `LM Eval Harness` | Benchmark tool to evaluate merged models |

## Evaluation and Pitfalls

- **Evaluate on your target tasks** — merging can both gain and lose capabilities unpredictably
- **Check for catastrophic interference** — very different fine-tunes can hurt base model performance
- **Scaling factor $\lambda$ matters** — too high causes instability; too low loses the merged skill
- **Base model alignment** — merging works best when all models share the same base (e.g., all Llama-3 variants)

## Real-World Examples

- **WizardMath + WizardCoder:** Merged to get a model strong at both math reasoning and code
- **OpenHermes + neural-chat:** Community merges combining instruct-following and chat styles
- **Leaderboard toppers:** Many top positions on the Open LLM Leaderboard are merged models, often using mergekit

## Further Reading

- Ilharco et al. (2022), *Editing Models with Task Arithmetic*
- Yadav et al. (2023), *TIES-Merging: Resolving Interference When Merging Models*
- Yu et al. (2023), *DARE: Language Model Merging by Dropping and Rescaling*
- mergekit GitHub repository by arcee-ai
