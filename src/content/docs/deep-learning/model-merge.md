---
title: "Model Merging: Combining LLMs Without Training"
description: "Learn how to merge multiple language models into a single model without additional training — from simple weight averaging to advanced techniques like mergekit and task arithmetic."
---

Model merging is a surprising technique: you can combine the capabilities of multiple fine-tuned models by simply averaging their weights. No additional training required. This has become an essential technique in the open-source LLM community.

## The Basic Insight

If model A is good at math and model B is good at coding, and both are based on the same base model, can we create a model that's good at both by averaging their weights?

```
merged = A.weights * 0.5 + B.weights * 0.5
```

Surprisingly, yes — with some caveats. This is the foundation of model merging.

## Simple Weight Averaging

The simplest merging strategy: average the weights of two models:

```python
import torch

def merge_simple(model_a_path, model_b_path, output_path, alpha=0.5):
    model_a = load_model(model_a_path)
    model_b = load_model(model_b_path)
    
    merged_state_dict = {}
    for name in model_a.state_dict():
        merged_state_dict[name] = (
            alpha * model_a.state_dict()[name].float() +
            (1 - alpha) * model_b.state_dict()[name].float()
        )
    
    merged_model = type(model_a)(config)
    merged_model.load_state_dict(merged_state_dict)
    merged_model.save_pretrained(output_path)
```

The `alpha` parameter controls the balance between the two models.

## Task Arithmetic

Task arithmetic (Ilharco et al., 2023) improves merging by focusing on the *difference* between fine-tuned and base models:

```python
def merge_task_arithmetic(base, task_a, task_b, alpha=0.5):
    """
    Merge by adding weighted task vectors to the base model.
    
    task_vector = fine_tuned_weights - base_weights
    merged = base + alpha * task_a + (1-alpha) * task_b
    """
    merged = {}
    for name in base:
        task_a_vec = task_a[name] - base[name]
        task_b_vec = task_b[name] - base[name]
        merged[name] = base[name] + alpha * task_a_vec + (1 - alpha) * task_b_vec
    return merged
```

This works better than direct weight averaging because it:
- Cancels out noise in the fine-tuning process.
- Focuses on the actual learned changes.
- Allows merging of models with different base architectures.

## The MergeKit Library

MergeKit is the standard tool for model merging:

```python
# mergekit.yaml
models:
  - model: meta-llama/Llama-2-7b-math
    parameters:
      alpha: 0.6
  - model: meta-llama/Llama-2-7b-code
    parameters:
      alpha: 0.4
merge_method: linear
dtype: bfloat16
```

```bash
# Run the merge
mergekit merge config.yaml output --allow-crimes
```

The `--allow-crimes` flag enables merging models that haven't been validated together (more on this below).

### Available Merge Methods in MergeKit

| Method | Description | Best For |
|--------|-------------|----------|
| `linear` | Simple weighted average | Models with similar capabilities |
| `task_arithmetic` | Add task vectors to base | Preserving base model knowledge |
| `dare_ties` | Drop redundant weights | Reducing interference |
| `zipit` | Merge layer by layer | Models with different architectures |
| `slerp` | Spherical linear interpolation | Smooth interpolation in weight space |

## DARE: Drop and REcombine

DARE improves merging by dropping redundant weights and recombining:

```python
def merge_dare_ties(base, model_a, model_b, alpha=0.5, epsilon=0.1):
    # Compute task vectors
    vec_a = model_a - base
    vec_b = model_b - base
    
    # Drop similar weights (epsilon)
    mask_a = torch.rand_like(vec_a) > epsilon
    mask_b = torch.rand_like(vec_b) > epsilon
    
    # Recombine
    merged = base + alpha * (vec_a * mask_a) + (1-alpha) * (vec_b * mask_b)
    return merged
```

This reduces interference between models by pruning weights that are similar in both models.

## ZipIt: Merging Different Architectures

ZipIt allows merging models with different layer structures:

```yaml
models:
  - model: mistral-7b-v0.1
  - model: mistral-7b-instruct
merge_method: zipit
```

ZipIt matches and merges layers based on their similarity, enabling merging of:
- Base model + instruction-tuned version.
- Model + LoRA adapter merged back into weights.
- Models with different layer orderings.

## Common Merging Patterns

### Merging for Capability Combination

```yaml
# Create a model good at both math and code
models:
  - model: meta-llama/Llama-2-7b-math
  - model: meta-llama/Llama-2-7b-code
merge_method: dare_ties
parameters:
  alpha: 0.5
```

### Merging Base + Chat Model

```yaml
# Preserve knowledge while adding chat capability
models:
  - model: meta-llama/Llama-2-7b-base
  - model: meta-llama/Llama-2-7b-chat
merge_method: task_arithmetic
parameters:
  alpha: 0.8  # Favor base knowledge
```

### Merging Multiple Expert Models

```yaml
# Create a generalist model
models:
  - model: math_expert_7b
  - model: code_expert_7b
  - model: science_expert_7b
  - model: creative_writing_7b
merge_method: linear
parameters:
  weights: [0.25, 0.25, 0.25, 0.25]
```

## Evaluation and Validation

After merging, evaluate the combined model:

```python
def evaluate_merged_model(model_path):
    model = load_model(model_path)
    
    evaluations = {
        "math": eval_math(model),
        "code": eval_code(model),
        "instruction": eval_instruction(model),
        "commonsense": eval_commonsense(model),
    }
    
    return evaluations
```

### The "Merge Crimes" Problem

Merging arbitrary models often produces mediocre results — called "merge crimes." Models may:
- Lose capabilities from both parents.
- Develop unexpected failure modes.
- Exhibit incoherent outputs.

Successful merges typically share:
- Same base architecture.
- Similar training data distribution.
- Compatible fine-tuning objectives.

## Practical Tips for Successful Merging

1. **Use models from the same family**: LLaMA 7B + LLaMA 7B works better than LLaMA 7B + Mistral 7B.

2. **Merge gradually**: Merge two models, evaluate, then merge the result with another. This reduces interference.

3. **Try different alphas**: The optimal balance varies by model pair. Search over alpha values.

4. **Use task arithmetic for unrelated tasks**: When combining very different capabilities, task arithmetic preserves more of each.

5. **DARE for many models**: When merging 3+ models, DARE-TIES reduces interference.

6. **Evaluate at scale**: Test on multiple benchmarks — the merged model may excel at some while failing at others.

## Automating Model Selection

Use grid search to find optimal merge configurations:

```python
from itertools import product

search_space = {
    "alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
    "method": ["linear", "task_arithmetic", "dare_ties"],
}

for config in product(*search_space.values()):
    merged = merge_with_config(models, **config)
    score = evaluate(merged, benchmark)
    save_result(config, score)
```

Model merging has democratized access to capable LLMs. The open-source community has produced thousands of merged models by combining fine-tunes — creating models that would have required enormous compute to train from scratch.