---
title: Multi-Fidelity Optimization
description: Explore multi-fidelity optimization — a family of methods that accelerate hyperparameter search and neural architecture search by strategically allocating compute across approximations of varying fidelity — covering Hyperband, BOHB, multi-fidelity Bayesian optimization, and their practical implementation.
---

Hyperparameter optimization (HPO) and neural architecture search (NAS) require evaluating many candidate configurations. Each evaluation means training a model — potentially for hundreds of GPU-hours. **Multi-fidelity optimization** attacks this computational bottleneck by exploiting cheaper, lower-fidelity approximations (shorter training, smaller datasets, reduced model size) to quickly screen candidates, reserving full-fidelity evaluation for the most promising ones.

## The Multi-Fidelity Principle

A fidelity $z \in [z_{\min}, z_{\max}]$ parameterizes the cost and accuracy of an evaluation:

- **Training budget**: evaluate on $z$ epochs out of a full $z_{\max}$-epoch training run
- **Dataset fraction**: train on $z\%$ of the full dataset
- **Model size**: use $z\%$ of the full model width or depth
- **Resolution**: train on $z$-pixel images instead of full resolution

Low-fidelity evaluations are cheap but noisy: a configuration that ranks highly at $z_{\min}$ may not maintain its ranking at $z_{\max}$. Multi-fidelity methods assume **ordinal correlation** — good configurations at low fidelity tend to be good at high fidelity — which empirically holds for most hyperparameters.

## Successive Halving

**Successive Halving (SH)** is the foundational algorithm. Given $n$ configurations and total budget $B$:

1. Allocate equal budget $B/n$ to all $n$ configurations
1. Evaluate all configurations at this budget
1. Keep the top $1/\eta$ fraction (typically $\eta = 3$ or $\eta = 4$)
1. Double the budget and repeat with surviving configurations

After $\log_\eta n$ rounds, one configuration has consumed the majority of the budget. Mathematically, total compute is:

$$\text{Total budget} = \sum_{k=0}^{\log_\eta n} \frac{n}{\eta^k} \cdot \frac{B}{n} \eta^k = B \log_\eta n$$

SH uses $O(\log n)$ times the budget of a single full evaluation — much cheaper than random search with $n$ full runs.

### The Dilemma: $n$ vs. Budget Per Config

SH requires fixing $n$ upfront. More configurations ($n$ large) explores more of the hyperparameter space but provides less budget per configuration — risking discarding good configurations that need more training to reveal their potential. Multi-fidelity methods resolve this tension.

## Hyperband

**Hyperband** (Li et al., 2017) eliminates the need to fix $n$ by running multiple brackets of Successive Halving with different $n$ / budget tradeoffs simultaneously.

### Algorithm

Given maximum budget per configuration $R$ and halving rate $\eta$:

$$s_{\max} = \lfloor \log_\eta R \rfloor$$

For each bracket $s \in \{s_{\max}, s_{\max}-1, \ldots, 0\}$:

- $n_s = \lceil \frac{s_{\max}+1}{s+1} \eta^s \rceil$ — number of configurations
- $r_s = R \eta^{-s}$ — initial budget per configuration

Run Successive Halving within each bracket. Bracket $s = s_{\max}$ uses many cheap evaluations; bracket $s = 0$ runs one configuration at full budget.

### Example (R = 81, η = 3)

| Bracket | Configs | Initial budget | Rounds |
| --- | --- | --- | --- |
| $s=4$ | 81 | 1 epoch | 5 rounds: 81→27→9→3→1 |
| $s=3$ | 27 | 3 epochs | 4 rounds: 27→9→3→1 |
| $s=2$ | 9 | 9 epochs | 3 rounds: 9→3→1 |
| $s=1$ | 3 | 27 epochs | 2 rounds: 3→1 |
| $s=0$ | 1 | 81 epochs | 1 round |

Total budget: $5 \times 81 = 405$ epochs — much less than $81 \times 81 = 6561$ for full evaluation of all 81 configurations.

Hyperband is purely bandit-based: configurations are sampled randomly and eliminated based on performance, with no model of the objective function.

## BOHB: Bayesian Optimization and Hyperband

**BOHB** (Bayesian Optimization and Hyperband, Falkner et al., 2018) combines the exploration efficiency of Hyperband's multi-fidelity elimination with the model-guided search of Bayesian optimization.

### Key Innovation

Replace Hyperband's random configuration sampling with **Tree-structured Parzen Estimators (TPE)** — a Bayesian model that fits probability densities over good and bad configurations:

$$\text{EI}(\lambda) \propto \frac{\ell(\lambda)}{g(\lambda)}$$

where $\ell(\lambda)$ is the density of configurations in the top $\gamma\%$ of observed losses and $g(\lambda)$ is the density of the rest. Maximizing EI selects configurations likely to be in the good region.

### How BOHB Combines Both

- Configuration selection: use TPE (with a separate model per fidelity level)
- Configuration elimination: use Hyperband's successive halving
- Cross-fidelity transfer: the Bayesian model at high fidelity is informed by observations at all lower fidelities

The result: BOHB finds good configurations faster than either pure Hyperband (benefits from model guidance) or pure Bayesian optimization (benefits from cheap low-fidelity screening).

## Multi-Fidelity Bayesian Optimization

Bayesian optimization builds a surrogate model of the objective and uses acquisition functions to decide the next evaluation point. Multi-fidelity extensions add fidelity as an additional input dimension.

### Multi-Fidelity Gaussian Processes

A GP surrogate over the joint space $(x, z)$ of hyperparameters $x$ and fidelity $z$ models the correlation between evaluations at different fidelities. **Multi-task GPs** (e.g., MTGP-BO) learn a covariance structure that transfers information from cheap low-fidelity observations to predict high-fidelity outcomes.

### MF-MES (Multi-Fidelity Max-value Entropy Search)

Acquisition functions for multi-fidelity BO select both the configuration $x^*$ and fidelity $z^*$ jointly:

$$(\hat{x}, \hat{z}) = \arg\max_{x,z} \frac{\text{MES}(x, z)}{\text{cost}(z)}$$

by maximizing the information gain about the optimal high-fidelity value per unit cost. This naturally allocates budget to cheap low-fidelity evaluations early and switches to expensive high-fidelity as the search converges.

## ASHA: Asynchronous Successive Halving

Hyperband and SH are synchronous — a bracket must wait for all configurations at the current fidelity before promoting survivors. **ASHA (Asynchronous Successive Halving)** removes synchronization barriers:

- Promote a configuration as soon as it is competitive with currently promoted configurations at that fidelity
- No waiting for slower workers
- Trivially parallelizable across many GPUs or machines

ASHA achieves similar performance to synchronous SH while enabling distributed hyperparameter search with near-linear scaling across workers.

## Practical Implementation

### Ray Tune

```python
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB, ASHAScheduler
from ray.tune.search.bohb import TuneBOHB

# BOHB configuration
algo = TuneBOHB()
scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=100,  # max epochs
    reduction_factor=3,
)

search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128, 256]),
    "dropout": tune.uniform(0.0, 0.5),
    "hidden_size": tune.choice([64, 128, 256, 512]),
}

tuner = tune.Tuner(
    trainable=train_function,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=100,
    ),
)
results = tuner.fit()
```

### Optuna with Hyperband

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(),
    pruner=HyperbandPruner(
        min_resource=1,
        max_resource=100,
        reduction_factor=3,
    ),
)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    for epoch in range(100):
        val_loss = train_epoch(lr, batch_size, epoch)

        # Report intermediate value for pruning
        trial.report(val_loss, epoch)

        # Early stopping if pruner decides to stop
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss

study.optimize(objective, n_trials=100)
```

## Fidelity Types and When to Use Them

| Fidelity Type | Works Well For | Pitfall |
| --- | --- | --- |
| Training epochs | Most deep learning tasks | Learning rate schedules may be epoch-sensitive |
| Dataset fraction | Large datasets, data-dominated tasks | Small datasets may not preserve class balance |
| Model size (width/depth) | NAS, architecture search | Small models may not predict large model behavior |
| Image resolution | Vision tasks | Some architectures (ViT patch size) depend on resolution |
| Number of trees (RF/GBM) | Ensemble methods | Tree-count correlation is strong — works well |

## When Multi-Fidelity Helps Most

Multi-fidelity optimization provides the largest gains when:

1. **Fidelity correlation is high**: low-fidelity rankings agree with high-fidelity rankings most of the time
1. **Cost ratio is large**: full training is orders of magnitude more expensive than low-fidelity evaluation
1. **Search space is large**: many configurations to explore, making exhaustive evaluation infeasible
1. **Early learning signals are informative**: validation loss after 10 epochs is predictive of final performance

It provides less benefit when:

- Low-fidelity evaluations are poor proxies (e.g., very different regularization dynamics at small vs. full scale)
- Training is already fast (no need to prune)
- The search space is small and random search is sufficient

## Comparison of Methods

| Method | Guided? | Async? | Fidelity model? | Best for |
| --- | --- | --- | --- | --- |
| Random search | No | Yes | No | Baseline |
| Successive Halving | No | No | No | Simple budget allocation |
| Hyperband | No | No | No | Robust, no prior needed |
| ASHA | No | Yes | No | Distributed, large-scale |
| BOHB | Yes (TPE) | No | Per-fidelity | Most practical tasks |
| MF-BO (GP) | Yes (GP) | Partial | Joint | Small search spaces |

## Summary

Multi-fidelity optimization is the dominant approach for scalable hyperparameter and architecture search in deep learning:

- **Successive Halving**: eliminates poor configurations early through geometric budget doubling
- **Hyperband**: hedges across multiple SH brackets, tolerating uncertainty about optimal budget allocation
- **BOHB**: adds Bayesian model-guided sampling to Hyperband's elimination, combining exploration and exploitation
- **ASHA**: async variant enabling efficient distributed search at scale

By spending 90% of compute on the top 10% of configurations, multi-fidelity methods achieve the same or better results as exhaustive search using a fraction of the total budget — making large-scale HPO practical for teams without unlimited GPU resources.
