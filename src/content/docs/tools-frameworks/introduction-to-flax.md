---
title: Introduction to Flax - Neural Networks on JAX
description: Learn how Flax builds on JAX to provide a flexible neural network library, covering the NNX module API, explicit state handling, and training with Optax.
---

Flax is a neural network library built on JAX. JAX provides composable function transformations (`jit`, `grad`, `vmap`, `pmap`) over NumPy-style code; Flax adds the layers, parameter management, and training utilities needed to build models.

## Modules with Flax NNX

NNX is Flax's modern API, which uses regular Python objects with mutable state while still working with JAX transformations:

```python
from flax import nnx
import jax.numpy as jnp

class MLP(nnx.Module):
    def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.l2 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))

model = MLP(784, 256, 10, rngs=nnx.Rngs(0))
```

## Training Step with Optax

Optax supplies optimizers and loss functions:

```python
import optax

optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss
```

## Why Use JAX and Flax

- **XLA compilation** produces fast code on GPUs and TPUs
- **Explicit randomness** through PRNG keys makes experiments reproducible
- **Easy parallelism**: sharding APIs scale models across devices
- **Functional transformations** compose cleanly (for example, per-example gradients with `vmap(grad(...))`)

## Considerations

JAX requires pure-function thinking and static shapes under `jit`; debugging compiled code is harder than eager PyTorch. The payoff is performance and strong support for large-scale TPU training.
