---
title: "TIES-Merging & DARE for Model Merging"
description: Explore advanced model merging techniques like TIES-Merging and DARE that combine multiple fine-tuned models without parameter conflicts.
---

Model Merging is a paradigm that allows developers to combine multiple task-specific Large Language Models (e.g., a math model, a coding model, and a chat model) into a single, multi-task model without performing additional training.

However, naive merging (like parameter averaging) leads to severe performance degradation due to parameter interference and sign conflicts. Advanced model merging algorithms like **TIES-Merging** and **DARE (Drop And REscale)** resolve these conflicts by filtering out redundant weights and aligning parameter sign directions.

---

## The Challenges of Naive Merging

When fine-tuning a base model $W_{\text{base}}$ on two different tasks to create $W_A$ and $W_B$, the parameter updates are:

$$\Delta W_A = W_A - W_{\text{base}}, \quad \Delta W_B = W_B - W_{\text{base}}$$

A simple average merge:

$$W_{\text{merged}} = W_{\text{base}} + \frac{\Delta W_A + \Delta W_B}{2}$$

This fails due to two primary issues:
1. **Redundant Parameter Updates:** Many parameter changes during fine-tuning are noise. Averaging noise from different models degrades performance.
2. **Sign Conflicts:** A parameter might increase in $W_A$ ($\Delta W_{A, i} > 0$) but decrease in $W_B$ ($\Delta W_{B, i} < 0$). Averaging them cancels out both updates.

---

## TIES-Merging (Trimming, Electing, and Merging)

TIES-Merging resolves parameter conflicts in three distinct phases:

```
Task Deltas (ΔWa, ΔWb) ---> 1. Trim (Keep Top-K%) ---> 2. Elect (Resolve Signs) ---> 3. Merge (Average Non-Zero)
```

### 1. Trim (Sparsification)
For each task-specific model, TIES-Merging keeps only the most significant parameter changes (e.g., the top-20% updates based on absolute magnitude) and sets the remaining 80% to zero. This removes low-magnitude noise.

### 2. Elect Sign (Sign Resolution)
To resolve sign conflicts, TIES-Merging determines a unified consensus sign for each parameter position. It aggregates the sign direction across all task updates and elects the majority sign direction:

$$\text{Consensus Sign}_i = \text{Sign}\left( \sum_{\tau \in \{A, B, \dots\}} \Delta W_{\tau, i} \right)$$

Updates that do not match the consensus sign are set to zero, preventing them from canceling each other out.

### 3. Merge
Finally, TIES-Merging averages the remaining non-zero parameter updates that align with the consensus sign, scales them, and adds them back to the base weights.

---

## DARE (Drop and Rescale)

**DARE (Drop And REscale)** is a merging technique that sparsifies task-specific delta parameters up to 99% before merging.

### How DARE Works
1. **Random Drop:** Instead of keeping the top-K% updates based on magnitude, DARE randomly drops delta parameters with a probability $p$ (often $p \ge 0.90$ or $0.99$):
   
   $$\tilde{\Delta W}_i = \begin{cases} 
   0 & \text{with probability } p \\
   \Delta W_i & \text{with probability } 1-p 
   \end{cases}$$

2. **Rescaling:** To ensure the expected value of the updates remains unchanged, DARE rescales the surviving parameters by a factor of $1/(1-p)$:
   
   $$\Delta W_i^{\text{DARE}} = \tilde{\Delta W}_i \cdot \frac{1}{1-p}$$

3. **Merge:** The sparsified, rescaled updates from different models are averaged and merged.

Because DARE reduces the number of active parameter changes to less than 5% of the total network weights, it eliminates conflicts, allowing developers to merge dozens of models successfully.

---

## Comparison: TIES-Merging vs. DARE

| Feature | TIES-Merging | DARE (Drop and Rescale) |
|---|---|---|
| **Selection Type** | Magnitude-based pruning (top-K) | Random dropping (dropout style) |
| **Rescaling** | None (standard average scaling) | Explicitly rescaled by $1/(1-p)$ |
| **Sign Resolution** | Strict majority sign voting | Implicitly mitigated via extreme sparsity |
| **Merging Capacity** | Best for 2-5 models | Can merge dozens of models simultaneously |
| **Sparsity Level** | Moderate ($80\%$ pruned) | Extreme ($90\% - 99\%$ pruned) |

---

## Code Concept: Simulating DARE Sparsification

Below is a PyTorch-like implementation of DARE parameter pruning.

```python
import torch

def dare_sparsify(delta_weight, drop_rate=0.90):
    """
    Applies DARE sparsification to a delta weight tensor.
    """
    # 1. Generate random dropout mask
    # True indicates parameter is kept
    mask = torch.rand_like(delta_weight) > drop_rate
    
    # 2. Apply mask
    sparsified_delta = delta_weight * mask
    
    # 3. Rescale remaining parameters
    rescale_factor = 1.0 / (1.0 - drop_rate)
    dare_delta = sparsified_delta * rescale_factor
    
    return dare_delta

# Example usage for merging two weight matrices:
# W_base, W_A, W_B are model weight matrices
# delta_A = W_A - W_base
# delta_B = W_B - W_base
#
# dare_A = dare_sparsify(delta_A, drop_rate=0.9)
# dare_B = dare_sparsify(delta_B, drop_rate=0.9)
#
# W_merged = W_base + (dare_A + dare_B) / 2
```
