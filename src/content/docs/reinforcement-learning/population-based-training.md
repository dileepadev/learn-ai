---
title: Population Based Training - Jointly Optimizing Weights and Hyperparameters
description: Learn how Population Based Training evolves hyperparameter schedules during training using exploit and explore steps, and where it is used in RL and large-scale training.
---

Population Based Training (PBT) is an optimization method that trains a population of models in parallel and, while they train, copies weights from better performers and perturbs hyperparameters. It produces hyperparameter *schedules*, not a single fixed configuration.

## Motivation

Grid and random search treat each configuration as an independent run and discard compute from poor runs. Many hyperparameters, especially in reinforcement learning, are best changed over time (a learning rate that decays, an entropy bonus that shrinks). PBT discovers such schedules automatically and reuses partial training progress.

## The Algorithm

Run N workers asynchronously. Each worker repeats:

```text
train for a fixed interval (steps)
evaluate performance
if performing poorly relative to the population:
    exploit:  copy weights and hyperparameters from a better member
    explore:  perturb the copied hyperparameters
continue training
```

- **Exploit** strategies: truncation selection (replace bottom 20% with copies of the top 20%) or binary tournament
- **Explore** strategies: multiply values by random factors such as 0.8 or 1.2, or resample from the prior

## Illustrative Code

```python
import copy, random

def pbt_step(population, frac=0.2):
    ranked = sorted(population, key=lambda m: m.score, reverse=True)
    cut = max(1, int(len(ranked) * frac))
    top, bottom = ranked[:cut], ranked[-cut:]
    for worker in bottom:
        donor = random.choice(top)
        worker.weights = copy.deepcopy(donor.weights)
        worker.hparams = {k: v * random.choice([0.8, 1.2]) for k, v in donor.hparams.items()}
```

## Benefits

- Efficient use of compute compared with independent random search
- Adapts hyperparameters to the training phase
- Naturally parallel and robust to unstable runs
- Applied successfully in deep RL (including multi-agent game agents), GANs, and translation models

## Practical Considerations

| Concern | Guidance |
|---------|----------|
| Population size | Often 16 to 64; too small reduces diversity |
| Interval length | Long enough for evaluation signal, short enough to adapt |
| Noisy metrics | Use smoothed or averaged evaluations |
| Greediness | Pure exploit can collapse diversity; keep exploration |
| Reproducibility | Final model comes from a lineage, so log the whole schedule |

Libraries such as Ray Tune implement PBT schedulers, including a variant (PB2) that picks new hyperparameters using Bayesian optimization instead of random perturbation.
