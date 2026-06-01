---
title: "Hyperparameter Optimization for LLMs"
description: "Learn systematic approaches to tuning LLM hyperparameters — from learning rates and batch sizes to context length and generation parameters — with practical techniques and frameworks."
---

Hyperparameter tuning is often treated as an afterthought, but the difference between a well-tuned and default-configured LLM can be dramatic. This guide covers the key hyperparameters for training and inference, and how to optimize them systematically.

## The Hyperparameter Landscape

LLM hyperparameters fall into several categories:

1. **Training hyperparameters**: Learning rate, batch size, warmup, weight decay.
2. **Model architecture**: Layer count, attention heads, hidden dimensions.
3. **Inference hyperparameters**: Temperature, top-p, max tokens, presence penalty.
4. **Training tricks**: LoRA rank, gradient accumulation, checkpoint frequency.

## Training Hyperparameters

### Learning Rate
The single most important training hyperparameter.

| Model Size | Typical Learning Rate | Notes |
|------------|---------------------|-------|
| 7B (full fine-tune) | 1e-5 to 3e-5 | Higher than full pretraining |
| 7B (LoRA) | 1e-4 to 3e-4 | LoRA needs higher LR |
| 70B (full fine-tune) | 5e-6 to 1e-5 | Larger models need smaller LR |
| 70B (QLoRA) | 2e-5 to 5e-5 | QLoRA needs careful tuning |

### Learning Rate Schedulers
- **Linear warmup + decay**: The most common approach. Warmup for 2–5% of steps, then decay linearly or with cosine.
- **Cosine decay**: Often works better for long training runs. Smooth decay prevents overshooting late in training.
- **Inverse square root decay**: Good for tasks where training steps vary widely.

### Batch Size
- **Effective batch size**: batch_size × gradient_accumulation_steps.
- **Larger is better** for training stability but increases memory requirements.
- **Power of 2 rule**: Batch sizes of 16, 32, 64, 128, 256 often work well.
- **Scaling law**: Validation loss often follows a log-linear relationship with batch size.

### Weight Decay
- **Default**: 0.01 to 0.1 for AdamW.
- **Higher for larger models**: 0.1 prevents overfitting in large models.
- **Lower for LoRA**: LoRA has fewer parameters, so less regularization is needed.

## Inference Hyperparameters

### Temperature
Controls the randomness of token selection.

- **temperature = 0**: Greedy decoding, deterministic output.
- **temperature = 0.3 to 0.7**: Good for most tasks, balances creativity and consistency.
- **temperature > 1.0**: More creative/distinct but can be incoherent.

```python
# Temperature selection by task
configs = {
    "factual_qa": {"temperature": 0.0},
    "creative_writing": {"temperature": 0.8},
    "code_generation": {"temperature": 0.2},
    "reasoning": {"temperature": 0.3},
}
```

### Top-k and Top-p (Nucleus Sampling)
Restrict the token distribution before sampling:

- **top_k**: Only consider the top K most likely tokens.
- **top_p**: Only consider tokens whose cumulative probability exceeds p.

```python
# Combining temperature with top-p
configs = {
    "balanced": {"temperature": 0.7, "top_p": 0.9},
    "focused": {"temperature": 0.3, "top_p": 0.7},
    "creative": {"temperature": 0.9, "top_p": 0.98},
}
```

### Presence and Frequency Penalties
Reduce repetition in generated text:

- **presence_penalty**: Reduces probability of tokens that have appeared anywhere.
- **frequency_penalty**: Reduces probability proportionally to how often a token appeared.

```python
# Reducing repetition
configs = {
    "no_repeat": {"presence_penalty": 0.5, "frequency_penalty": 0.5},
    "standard": {"presence_penalty": 0.1, "frequency_penalty": 0.1},
}
```

## Hyperparameter Tuning Strategies

### Grid Search
Try all combinations of hyperparameters. Prohibitive for large search spaces.

### Random Search
Sample hyperparameters randomly. More efficient than grid search for the same budget.

```python
from scipy import stats

search_space = {
    "learning_rate": stats.loguniform(1e-6, 1e-3),
    "batch_size": [16, 32, 64, 128],
    "warmup_ratio": stats.uniform(0.01, 0.1),
    "weight_decay": stats.loguniform(1e-4, 1e-1),
}

best_loss = float("inf")
for _ in range(100):
    config = {k: dist.rvs() for k, dist in search_space.items()}
    loss = train_and_evaluate(config)
    if loss < best_loss:
        best_loss = loss
        best_config = config
```

### Bayesian Optimization
Build a surrogate model of the objective function and efficiently explore promising regions. Libraries like Optuna and Ray Tune implement this.

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    temperature = trial.suggest_float("temperature", 0.0, 1.0)
    
    model = train_model(learning_rate=lr, batch_size=batch_size)
    return evaluate(model, temperature=temperature)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

### Successive Halving
For expensive training runs, quickly eliminate bad configurations:

1. Train all configurations for a small number of steps.
2. Keep the top performers.
3. Increase the number of steps and repeat.

## Hyperparameter Transferability

Some hyperparameters transfer across models and tasks:

- **Learning rate scheduler**: Cosine decay with warmup works well universally.
- **Batch size**: Powers of 2 work across most models.
- **Inference settings**: Temperature and top-p choices are largely task-dependent, not model-dependent.

Others require retuning:
- **Learning rate absolute values**: Vary by model size and architecture.
- **LoRA rank**: Depends on the task and model.
- **Generation penalties**: Task-dependent.

## Automated LLM Tuning

New research automates LLM hyperparameter tuning:

- **Optuna**: General-purpose optimization framework.
- **Ray Tune**: Distributed hyperparameter tuning with LLM support.
- **Weights & Biases Sweeps**: Cloud-based parallel tuning.
- **DeepSpeed-Chat**: RLHF hyperparameter tuning with auto-tuning.

## Practical Recommendations

1. **Start with known good defaults**: Don't start from scratch; use published configurations.
2. **Tune learning rate first**: It's usually the most impactful.
3. **Use early stopping**: Don't waste time on configurations that aren't converging.
4. **Track everything**: Use experiment tracking (MLflow, W&B) to compare runs.
5. **Validate on realistic data**: Tuning on synthetic data doesn't guarantee production quality.

Hyperparameter optimization is an art grounded in science. The principles are well-understood, but the specific values for your model and task require systematic experimentation.