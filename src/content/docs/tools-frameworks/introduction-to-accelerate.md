---
title: Introduction to Accelerate
description: Learn HuggingFace Accelerate — the library that makes PyTorch training code device-agnostic and distributed with minimal code changes — covering single-GPU, multi-GPU data parallel, FSDP, DeepSpeed integration, mixed precision, gradient accumulation, and profiling for efficient LLM training.
---

Training a large model in PyTorch requires boilerplate for device placement, distributed data parallelism, mixed precision, and gradient scaling that quickly overwhelms the actual training logic. **Accelerate** (HuggingFace) solves this by providing a thin abstraction layer that makes the same training loop run identically on a laptop CPU, a single GPU, multiple GPUs, and TPUs — with minimal code changes and full compatibility with native PyTorch.

## The Problem Accelerate Solves

A standard PyTorch training loop requires:

- `model.to(device)` — manual device placement
- `torch.nn.DataParallel` or `DistributedDataParallel` — distributed training setup
- `torch.cuda.amp.autocast()` and `GradScaler` — mixed precision
- Custom gradient accumulation logic
- DeepSpeed or FSDP configuration files

Accelerate replaces all of this with a single `Accelerator` object:

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",        # Automatic mixed precision
    gradient_accumulation_steps=4, # Gradient accumulation
)
model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)
```

## Installation

```bash
pip install accelerate
accelerate config   # Interactive configuration wizard
```

## Basic Training Loop

The minimal Accelerate training loop adds four lines to standard PyTorch:

```python
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Four changes from standard PyTorch:
accelerator = Accelerator()                                     # 1. Create Accelerator

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

model, optimizer, train_loader, scheduler = accelerator.prepare( # 2. Prepare everything
    model, optimizer, train_loader, scheduler
)

for epoch in range(10):
    for batch in train_loader:
        with accelerator.accumulate(model):                     # 3. Handle gradient accumulation
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)                          # 4. Use accelerator.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

This loop works unchanged across CPU, single GPU, multi-GPU DDP, and TPU.

## Launch Scripts

Accelerate provides a unified launcher:

```bash
# Single GPU
accelerate launch train.py

# Multi-GPU (all available)
accelerate launch --multi_gpu train.py

# 4 GPUs on 2 nodes
accelerate launch --num_processes=8 --num_machines=2 --machine_rank=0 train.py

# DeepSpeed ZeRO-3
accelerate launch --config_file deepspeed_config.yaml train.py
```

## Mixed Precision

Accelerate handles mixed precision automatically — no manual `autocast` or `GradScaler`:

```python
# bfloat16 (recommended for Ampere+ GPUs): no dynamic loss scaling needed
accelerator = Accelerator(mixed_precision="bf16")

# float16 (for older GPUs): GradScaler applied automatically
accelerator = Accelerator(mixed_precision="fp16")

# Check precision inside training loop
if accelerator.mixed_precision == "bf16":
    print("Training with bfloat16")
```

## FSDP: Fully Sharded Data Parallel

For models too large to fit on a single GPU, FSDP shards model parameters, gradients, and optimizer states across all GPUs — enabling LLaMA-3-70B training on 8× A100 80GB GPUs:

```python
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

Or via configuration file (`accelerate config`), selecting FSDP:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1  # FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
mixed_precision: bf16
num_processes: 8
```

## DeepSpeed Integration

Accelerate integrates DeepSpeed ZeRO stages without requiring changes to the training loop:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"},
    "overlap_comm": true,
    "allgather_partitions": true,
    "reduce_scatter": true
  },
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

```bash
accelerate launch --config_file ds_zero3.json train.py
```

DeepSpeed ZeRO-3 with CPU offload enables training 175B+ parameter models on 8 GPUs by offloading optimizer states and parameters to CPU RAM.

## Gradient Accumulation

Accelerate's `accumulate` context manager handles gradient accumulation correctly — syncing gradients only on the final step:

```python
accelerator = Accelerator(gradient_accumulation_steps=8)

for batch in train_loader:
    with accelerator.accumulate(model):
        loss = model(**batch).loss
        accelerator.backward(loss)
        # Gradient sync and optimizer step happen automatically
        # only every 8 steps
        if accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

Effective batch size = `per_device_batch_size × num_gpus × gradient_accumulation_steps`.

## Checkpointing

Accelerate provides distributed-safe checkpointing:

```python
# Save (handles FSDP/DeepSpeed sharding automatically)
accelerator.save_state("./checkpoint-1000")

# Load
accelerator.load_state("./checkpoint-1000")

# Wait for all processes before saving
accelerator.wait_for_everyone()
# Unwrap the model before saving (removes DDP/FSDP wrappers)
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    "./final-model",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)
```

## Utilities and Logging

```python
# Only log/print from main process
if accelerator.is_main_process:
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Gather predictions from all GPUs
all_preds = accelerator.gather_for_metrics(preds)
all_labels = accelerator.gather_for_metrics(labels)

# Use built-in logging to multiple backends
from accelerate.logging import get_logger
logger = get_logger(__name__)
logger.info(f"Loss: {loss:.4f}", main_process_only=True)

# Track memory usage
accelerator.print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Notebook Launcher

For Jupyter notebooks, Accelerate provides a launcher that spawns distributed processes:

```python
from accelerate import notebook_launcher


def training_function():
    accelerator = Accelerator()
    # ... full training loop ...


notebook_launcher(training_function, num_processes=4)
```

## Full LLM Fine-Tuning Example

```python
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=len(train_loader) * 3
)

model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)

for epoch in range(3):
    model.train()
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 100 == 0 and accelerator.is_main_process:
            print(f"Step {step}: loss={loss.item():.4f}")

    accelerator.wait_for_everyone()
    accelerator.save_state(f"./checkpoint-epoch-{epoch}")
```

## Summary

Accelerate makes distributed LLM training accessible without sacrificing flexibility:

- **Four lines of change** convert a standard PyTorch loop to run across CPU, GPU, multi-GPU, and TPU
- **Mixed precision** (bf16/fp16) is handled automatically with no manual `autocast` or `GradScaler`
- **FSDP and DeepSpeed** integration enables training models that don't fit in a single GPU's memory
- **Gradient accumulation** with `accumulate()` correctly handles gradient synchronization in distributed settings
- **Distributed-safe checkpointing** handles FSDP/DeepSpeed model sharding transparently
- Accelerate is the backbone of HuggingFace's `Trainer` and is fully compatible with PEFT, TRL, and transformers
