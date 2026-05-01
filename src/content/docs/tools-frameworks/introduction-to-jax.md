---
title: Introduction to JAX
description: Learn JAX — Google's high-performance numerical computing library that combines NumPy-compatible arrays with composable function transformations including JIT compilation, automatic differentiation, vectorization, and multi-device parallelism, enabling research-grade ML at production scale.
---

**JAX** (Just After eXecution) is a Python library developed by Google Research that provides NumPy-compatible arrays with powerful function transformations for high-performance machine learning and scientific computing. Where NumPy is sequential and CPU-only, JAX adds four composable transformations — `jit`, `grad`, `vmap`, and `pmap` — that together enable fast, differentiable, parallelized computation on CPUs, GPUs, and TPUs with minimal code changes.

JAX has become the framework of choice for cutting-edge ML research at Google DeepMind, where it powers AlphaFold 2, Gemini training, and numerous research publications. Its functional programming model, explicit randomness handling, and composable transforms make it particularly well-suited for rapid research experimentation.

## Core Concepts: Pure Functions and Immutable Arrays

JAX is built around **pure functions** — functions with no side effects that always produce the same output for the same input. This constraint is what makes JAX's transformations composable and safe.

Unlike NumPy, JAX arrays are **immutable**. You cannot modify array elements in-place:

```python
import jax
import jax.numpy as jnp
import numpy as np

# JAX arrays: similar to NumPy but on accelerators
x = jnp.array([1.0, 2.0, 3.0, 4.0])
print(x.shape, x.dtype)      # (4,) float32
print(type(x))                # jaxlib.xla_extension.ArrayImpl

# Automatic device placement (GPU/TPU if available)
print(x.device())             # TFRT_CPU_0 or CudaDevice(id=0)

# NumPy-compatible operations
y = jnp.sqrt(x)
z = jnp.sin(x) + jnp.cos(x)

# In-place modification is NOT allowed:
# x[0] = 5.0  ← raises TypeError

# Instead, use functional update:
x_updated = x.at[0].set(5.0)       # creates a new array
x_added = x.at[1].add(10.0)        # x[1] + 10.0 in new array
x_mul = x.at[2:].multiply(0.5)     # slice update

# Convert between NumPy and JAX
np_array = np.array(x)             # copy to CPU NumPy
jax_from_np = jnp.asarray(np_array)
```

## JIT Compilation with `jax.jit`

`jax.jit` compiles a Python function to XLA (Accelerated Linear Algebra), a low-level compiler that optimizes and fuses operations for the target hardware:

```python
import jax
import jax.numpy as jnp
import time

def slow_fn(x, w, b):
    """A simple linear layer with ReLU activation — Python-level."""
    return jnp.maximum(0, x @ w + b)

# JIT-compiled version
fast_fn = jax.jit(slow_fn)

# Initialize inputs
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, shape=(1000, 512))
w = jax.random.normal(key, shape=(512, 256))
b = jnp.zeros(256)

# First call: traces + compiles (slow)
out = fast_fn(x, w, b)
out.block_until_ready()   # JAX is async by default; force synchronization

# Subsequent calls: runs compiled code (fast)
start = time.perf_counter()
for _ in range(100):
    out = fast_fn(x, w, b)
    out.block_until_ready()
print(f"100 calls in {time.perf_counter() - start:.3f}s")

# Can also use as a decorator
@jax.jit
def relu_layer(x, w, b):
    return jnp.maximum(0, x @ w + b)
```

**How JIT works**: On the first call, JAX traces the function with abstract values (ShapedArrays), recording all operations into a computation graph. This graph is compiled to native code by XLA. Subsequent calls with the same shapes skip tracing entirely and run the compiled code directly. This makes JAX as fast as hand-optimized CUDA kernels for many workloads.

## Automatic Differentiation with `jax.grad`

`jax.grad` computes the gradient of a scalar-valued function with respect to its first argument (by default):

```python
import jax
import jax.numpy as jnp

def mse_loss(params, x, y_true):
    """Mean squared error for a linear model y = x @ w + b."""
    w, b = params
    y_pred = x @ w + b
    return jnp.mean((y_pred - y_true) ** 2)

# grad differentiates w.r.t. the first argument (params)
grad_loss = jax.grad(mse_loss)

# value_and_grad computes both value and gradient in one pass (more efficient)
loss_and_grad = jax.value_and_grad(mse_loss)

# Initialize
key = jax.random.PRNGKey(42)
w = jax.random.normal(key, (10, 1))
b = jnp.zeros(1)
params = (w, b)

x = jax.random.normal(key, (100, 10))
y_true = jax.random.normal(key, (100, 1))

# Compute gradients — params is a PyTree (tuple of arrays)
loss, grads = loss_and_grad(params, x, y_true)
grad_w, grad_b = grads
print(f"Loss: {loss:.4f}, grad_w shape: {grad_w.shape}")

# Simple gradient descent step
lr = 0.01
params = (w - lr * grad_w, b - lr * grad_b)
```

### Higher-Order Derivatives

```python
def f(x):
    return jnp.sin(x) ** 2

df = jax.grad(f)       # first derivative
ddf = jax.grad(df)     # second derivative
dddf = jax.grad(ddf)   # third derivative

x = jnp.array(1.0)
print(f"f={f(x):.4f}, f'={df(x):.4f}, f''={ddf(x):.4f}")

# Jacobian and Hessian
def g(x):
    return jnp.array([x[0]**2 + x[1], x[0] * x[1]])

J = jax.jacobian(g)(jnp.array([2.0, 3.0]))     # (2, 2) Jacobian
H = jax.hessian(lambda x: jnp.sum(g(x)))(jnp.array([2.0, 3.0]))  # (2, 2) Hessian
```

## Vectorization with `jax.vmap`

`jax.vmap` (vectorized map) automatically batches a function that operates on a single example to operate on a batch — without rewriting the function:

```python
import jax
import jax.numpy as jnp

def predict_single(w, b, x_single):
    """Forward pass for a single example."""
    return jnp.tanh(w @ x_single + b)

# Vectorize over examples (axis 0 of x_batch, not w or b)
predict_batch = jax.vmap(
    predict_single,
    in_axes=(None, None, 0)   # w and b are shared; x is batched over axis 0
)

w = jnp.ones((8, 16))
b = jnp.zeros(8)
x_batch = jnp.ones((32, 16))   # 32 examples, 16 features

out = predict_batch(w, b, x_batch)   # (32, 8) — no explicit batch dimension in the function!
print(out.shape)   # (32, 8)

# vmap composes with jit
fast_predict_batch = jax.jit(jax.vmap(predict_single, in_axes=(None, None, 0)))
```

`vmap` is especially powerful for **per-example gradient computation** (useful for differential privacy, influence functions, and meta-learning):

```python
# Compute per-example gradients efficiently
def loss_single(params, x_single, y_single):
    w, b = params
    pred = w @ x_single + b
    return jnp.mean((pred - y_single) ** 2)

# Map grad over individual examples — no loops needed
per_example_grads = jax.vmap(
    jax.grad(loss_single),
    in_axes=(None, 0, 0)
)(params, x_batch, y_batch)
```

## Multi-Device Parallelism with `jax.pmap`

`jax.pmap` (parallel map) shards computation across multiple devices (GPUs or TPUs), with each device processing a different slice of the batch:

```python
import jax
from jax import pmap
import jax.numpy as jnp

# Number of devices available
n_devices = jax.device_count()
print(f"Devices: {jax.devices()}")

@pmap
def parallel_dot(x, w):
    """Each device computes x @ w for its shard of x."""
    return x @ w

# Input must have a leading axis of size n_devices
x_sharded = jnp.ones((n_devices, 128, 512))  # (devices, batch_per_device, features)
w = jnp.ones((512, 256))

# Replicate w across devices
w_replicated = jnp.stack([w] * n_devices)

out = parallel_dot(x_sharded, w_replicated)   # (n_devices, 128, 256)

# For training: use jax.lax.pmean to average gradients across devices
@pmap
def train_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Average gradients across all devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    return loss, grads
```

## Explicit Randomness: PRNG Keys

JAX uses **explicit PRNG keys** rather than hidden global state. This makes randomness reproducible and compatible with JAX's functional model:

```python
import jax
import jax.numpy as jnp

# Create a root key — all randomness flows from here
key = jax.random.PRNGKey(seed=42)

# NEVER reuse a key — split to get independent subkeys
key, subkey1, subkey2 = jax.random.split(key, num=3)

# Use subkeys for different operations
w1 = jax.random.normal(subkey1, shape=(256, 256))
w2 = jax.random.normal(subkey2, shape=(256, 256))

# Inside loops/functions: always split
def init_weights(key, layer_sizes):
    params = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, subkey = jax.random.split(key)
        # He initialization
        scale = jnp.sqrt(2.0 / fan_in)
        w = jax.random.normal(subkey, (fan_in, fan_out)) * scale
        params.append(w)
    return params

params = init_weights(key, [784, 256, 128, 10])
```

## Composing Transforms: A Training Loop

The transforms compose naturally — `jit(vmap(grad(...)))` is a common pattern:

```python
import jax
import jax.numpy as jnp
import optax   # JAX-compatible optimizer library

def mlp_forward(params, x):
    """Multi-layer perceptron forward pass."""
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return x @ w + b

def cross_entropy_loss(params, x, y):
    logits = mlp_forward(params, x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

# JIT-compile the entire training step
@jax.jit
def train_step(params, opt_state, x_batch, y_batch):
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Initialize optimizer (optax provides Adam, AdamW, SGD, etc.)
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
opt_state = optimizer.init(params)

for epoch in range(100):
    for x_batch, y_batch in dataloader:
        params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
```

## JAX Ecosystem

| Library | Purpose |
|---|---|
| **Flax** | Neural network layers, training utilities (Google DeepMind) |
| **Haiku** | Module system with Sonnet-like API (DeepMind) |
| **Equinox** | PyTorch-like module system with pure functional core |
| **Optax** | Gradient processing and optimizers |
| **Orbax** | Checkpointing and model serialization |
| **Chex** | Testing and debugging utilities for JAX code |
| **Distrax** | Probability distributions and bijectors |
| **RLax** | Reinforcement learning building blocks |

## JAX vs. PyTorch

| Feature | JAX | PyTorch |
|---|---|---|
| Execution model | Functional, explicit state | Object-oriented, mutable state |
| JIT compilation | XLA (ahead-of-time trace) | TorchDynamo (eager + compile) |
| Auto-diff | `grad`, `jacobian`, `hessian` | `autograd`, `torch.compile` |
| Vectorization | `vmap` (explicit) | Broadcasting (implicit) |
| Multi-device | `pmap` / `shard_map` | `DistributedDataParallel` |
| Ecosystem | Research-focused (DeepMind) | Broad (industry + research) |
| Learning curve | Steeper (functional paradigm) | Gentler (Pythonic) |

JAX's functional purity and composable transforms make it exceptionally powerful for research — especially work that requires custom gradient computations, per-example gradients, or novel parallelism strategies. For production deployment of standard architectures, PyTorch's larger ecosystem and tooling often provides a faster path to shipping.
