---
title: Introduction to PyTorch Lightning - Structured Training Code
description: Learn how PyTorch Lightning separates research code from engineering boilerplate, handling devices, distributed training, and logging through the LightningModule and Trainer.
---

PyTorch Lightning is a lightweight wrapper that organizes PyTorch code. It removes repetitive engineering code (device placement, training loops, distributed setup, checkpointing) while leaving the model logic in plain PyTorch.

## LightningModule

You define what the model does; Lightning decides how to run it:

```python
import torch
import lightning as L
import torch.nn.functional as F

class Classifier(L.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", F.cross_entropy(self.model(x), y))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
```

## Trainer

```python
trainer = L.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices="auto",
    precision="16-mixed",
)
trainer.fit(Classifier(net), train_loader, val_loader)
```

Switching to multiple GPUs, mixed precision, or gradient accumulation is a configuration change rather than a code rewrite.

## Useful Features

- **Callbacks**: early stopping, checkpointing, learning-rate monitoring
- **Loggers**: TensorBoard, Weights & Biases, MLflow
- **Strategies**: DDP, FSDP, DeepSpeed for scaling
- **LightningDataModule**: encapsulates data preparation and loaders

## Trade-offs

Lightning improves reproducibility and structure, but its abstractions add a learning curve for unusual training schemes such as multiple optimizers or custom loop control.
