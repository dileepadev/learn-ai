---
title: Introduction to Keras - High-Level Deep Learning API
description: Learn how Keras 3 provides a unified high-level API for building and training neural networks that can run on JAX, TensorFlow, or PyTorch backends.
---

Keras is a high-level deep learning API focused on developer productivity. Keras 3 is multi-backend: the same model code can run on JAX, TensorFlow, or PyTorch, selected through configuration.

## Three Ways to Build Models

**Sequential** for simple stacks:

```python
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax"),
])
```

**Functional API** for branching and multiple inputs/outputs:

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
```

**Subclassing** for full custom logic by overriding `call`.

## Compile, Fit, Evaluate

```python
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(x_train, y_train, validation_split=0.1, epochs=10, batch_size=64)
model.evaluate(x_test, y_test)
```

## Callbacks

Callbacks hook into training for common needs: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, and `TensorBoard` logging.

## Choosing a Backend

```bash
export KERAS_BACKEND=jax   # or tensorflow, torch
```

## KerasHub

KerasHub offers pretrained architectures for vision and language tasks, with fine-tuning support, so you rarely start from random weights.

## When It Fits

Keras is a strong choice for fast prototyping and standard architectures. For highly custom training loops or research code, you may prefer working directly in PyTorch or JAX.
