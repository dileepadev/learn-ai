---
title: Self-Instruct - Bootstrapping Instruction Data from a Model
description: Learn how Self-Instruct uses a language model to generate its own instruction-tuning data from a small seed set, and the filtering that keeps it useful.
---

Self-Instruct is a method for creating instruction-following datasets with little human annotation. A pretrained model is prompted to generate new tasks and responses from a small seed set, and the results are filtered and used to fine-tune the same or another model.

## The Pipeline

1. **Seed tasks**: start with a few hundred human-written instructions (the original work used 175)
2. **Generate instructions**: prompt the model with a handful of sampled tasks and ask for new ones
3. **Classify task type**: determine whether the task has a limited set of outputs (classification) or is open-ended
4. **Generate instances**: produce inputs and outputs, using an output-first approach for classification to avoid label bias
5. **Filter**: remove low-quality or near-duplicate items
6. **Fine-tune**: train on the filtered set, and optionally add it back to the task pool

## Filtering Heuristics

```python
def keep(new_instruction, pool, scorer):
    if any(rouge_l(new_instruction, old) > 0.7 for old in pool):
        return False                      # near duplicate
    if contains_banned_keywords(new_instruction):
        return False                      # e.g. image, graph, file
    if len(new_instruction.split()) < 3:
        return False
    return True
```

Common filters include similarity thresholds, length checks, and removal of tasks that depend on unavailable modalities.

## Impact

Self-Instruct showed a base model could close much of the gap to instruction-tuned models using its own generations, and inspired datasets such as Alpaca, which used a stronger model to produce 52K examples.

## Modern Variants

- **Evol-Instruct** makes instructions progressively harder or more varied
- **Persona-based generation** diversifies topics and styles
- **Rejection sampling with reward models or verifiers** keeps only high-quality answers

## Risks

Generated data inherits the generator's mistakes and biases, can lack diversity, and may violate terms of service when produced by proprietary models. Training repeatedly on model-generated data risks quality collapse, so mix in human-written data and measure on held-out benchmarks.
