---
title: Introduction to DeepSpeed
description: Learn how Microsoft DeepSpeed enables training of trillion-parameter models through ZeRO optimizer stages, pipeline parallelism, tensor parallelism, and CPU/NVMe offloading — making large-scale deep learning accessible on commodity hardware.
---

**DeepSpeed** is an open-source deep learning optimization library developed by Microsoft that enables training of extremely large language models on GPU clusters that would otherwise be computationally infeasible. Originally released in 2020, DeepSpeed has become one of the foundational tools for large-scale model training — used in the development of models including GPT-NeoX 20B, Bloom 176B, and many others in the open-source community.

The core innovation of DeepSpeed is the **ZeRO (Zero Redundancy Optimizer)** family of algorithms, which eliminates the memory redundancy inherent in standard data-parallel training. Alongside ZeRO, DeepSpeed provides a comprehensive suite of techniques: mixed-precision training, gradient accumulation, pipeline parallelism, tensor parallelism, sparse attention, and CPU/NVMe memory offloading — all accessible through a unified configuration system.

## Why Large Model Training Is Hard

To understand why DeepSpeed matters, consider the memory demands of training a large model. A model with $N$ parameters in FP16 requires:

- $2N$ bytes for **model parameters**.
- $2N$ bytes for **gradients**.
- $12N$ bytes for **optimizer states** (Adam maintains fp32 parameters, momentum, and variance — 3 fp32 values per parameter = 12 bytes).

Total: approximately $16N$ bytes — or 16 GB for a 1B parameter model. A 7B parameter model requires ~112 GB of GPU memory for training in standard data-parallel mode, far exceeding the 80 GB capacity of an A100.

**Standard data parallelism** replicates the full model on each GPU and splits the batch — every GPU stores all parameters, gradients, and optimizer states. At 7B parameters, this means 112 GB per GPU, limiting training to GPUs with sufficient memory.

## ZeRO: Zero Redundancy Optimizer

ZeRO eliminates memory redundancy by **partitioning** the model state across GPUs instead of replicating it. ZeRO has three stages of increasing memory savings:

### ZeRO Stage 1: Optimizer State Partitioning

Each GPU stores only $\frac{1}{N_{GPU}}$ of the optimizer states (Adam momentum and variance). Parameters and gradients remain replicated.

- **Memory reduction**: ~4× for optimizer states (from 12N to 3N bytes across the cluster).
- **Communication overhead**: Minimal — standard all-reduce for gradients plus a reduce-scatter for optimizer state updates.

### ZeRO Stage 2: Gradient Partitioning

Each GPU stores only its partition of optimizer states AND its partition of gradients. Parameters remain replicated.

- **Memory reduction**: ~8× for optimizer states and gradients combined.
- **Communication overhead**: Replace all-reduce with reduce-scatter (equivalent communication volume).

### ZeRO Stage 3: Parameter Partitioning

Each GPU stores only its partition of **all** model state: parameters, gradients, and optimizer states.

- **Memory reduction**: Proportional to the number of GPUs — with 64 GPUs, memory per GPU is reduced by 64×.
- **Communication overhead**: Parameters must be gathered from all GPUs during forward and backward passes. Communication volume is 1.5× standard data parallelism.

ZeRO Stage 3 enables training models that are too large to fit on a single GPU — the memory per GPU scales with model size divided by the number of GPUs.

$$\text{Memory per GPU (ZeRO-3)} \approx \frac{16N}{N_{GPU}} \text{ bytes}$$

For a 70B parameter model with 8 GPUs, ZeRO-3 reduces per-GPU memory to approximately 140 GB / 8 = 17.5 GB — within range of an A100.

### ZeRO-Infinity: CPU and NVMe Offloading

**ZeRO-Infinity** extends ZeRO Stage 3 by offloading optimizer states, gradients, or even parameters to CPU memory or NVMe storage:

- **CPU offload**: Leverage the large DRAM capacity of modern servers (1–2 TB) to hold optimizer states and parameters that don't fit on GPU.
- **NVMe offload**: For extreme-scale models, NVMe SSDs (with 8+ TB capacity) serve as a third tier of the memory hierarchy.

ZeRO-Infinity enables training 100B+ parameter models on a single GPU server with sufficient CPU RAM and NVMe storage, at the cost of significantly reduced training throughput due to PCIe bandwidth limitations between CPU and GPU.

## Configuration with DeepSpeed

DeepSpeed is configured through a JSON configuration file that specifies the training strategy, precision, ZeRO stage, and all optimization settings.

### Basic Configuration (ZeRO Stage 2)

```json
{
  "train_batch_size": 256,
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 8,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16
  },

  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "contiguous_gradients": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-4,
      "warmup_num_steps": 2000,
      "total_num_steps": 100000
    }
  }
}
```

### ZeRO Stage 3 with CPU Offload

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

### Python Integration

```python
import deepspeed
import torch

model = MyLargeModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Initialize DeepSpeed engine
model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    training_data=train_dataset,
    config="ds_config.json"
)

# Training loop — identical to standard PyTorch
for batch in train_dataloader:
    inputs, labels = batch
    outputs = model_engine(inputs)
    loss = criterion(outputs, labels)

    model_engine.backward(loss)
    model_engine.step()
```

The `model_engine` wraps the model with DeepSpeed's parallelism and memory management, providing a backward-compatible interface — the training loop code remains essentially unchanged.

## Pipeline Parallelism

**Pipeline parallelism** partitions model layers across GPUs, with each GPU responsible for a contiguous set of layers:

```
GPU 0: Layers 0-11  (Embedding + first 12 transformer blocks)
GPU 1: Layers 12-23 (Middle 12 transformer blocks)
GPU 2: Layers 24-35 (Last 12 transformer blocks)
GPU 3: Layers 36-47 (Final transformer blocks + LM head)
```

Training proceeds in a pipeline: micro-batches flow forward through GPUs in sequence, then gradients flow backward. DeepSpeed uses **1F1B (one-forward-one-backward) scheduling** to keep all pipeline stages active simultaneously, minimizing idle ("bubble") time.

Pipeline parallelism is particularly effective for models where each layer is too large to fit on a single GPU, or where combining with data parallelism would still exceed per-GPU memory.

## Tensor Parallelism

**Tensor parallelism** partitions individual layers across GPUs, with each GPU computing a portion of each matrix multiplication. In a transformer's attention head, for example:

- Each GPU computes attention for a subset of heads.
- Results are aggregated via all-reduce.

Tensor parallelism has higher communication overhead than pipeline parallelism (requiring all-reduces inside each forward/backward pass) but provides finer-grained memory partitioning.

DeepSpeed's integration with **Megatron-LM** enables tensor parallelism alongside ZeRO and pipeline parallelism — the "3D parallelism" used to train models at the scale of GPT-NeoX 20B and Bloom 176B.

## Mixed Precision Training

DeepSpeed's FP16 (and BF16) mixed precision training:

- Stores parameters in FP32 for optimizer updates (full precision).
- Performs forward and backward passes in FP16 (half precision) for speed and memory efficiency.
- Uses **dynamic loss scaling** to prevent gradient underflow — automatically scaling up the loss before backward pass and scaling down gradients, detecting and skipping updates that produce overflow.

BF16 (brain float 16) is preferred on Ampere and later NVIDIA GPUs and on TPUs — it has the same dynamic range as FP32 (avoiding the underflow problem) with the memory savings of FP16, eliminating the need for loss scaling.

## DeepSpeed Inference

**DeepSpeed-Inference** optimizes inference (not just training) for transformer models:

- **Kernel fusion**: Fusing multiple operations (attention, layer norm, activation) into single GPU kernels, reducing memory bandwidth overhead.
- **INT8 quantization**: Deploying models in 8-bit integer precision for 2× memory reduction with minimal accuracy loss.
- **Transformer engine integration**: Leveraging NVIDIA's Transformer Engine for FP8 precision on H100 GPUs.
- **Tensor parallelism for inference**: Distributing large models across multiple GPUs for inference, enabling models that don't fit on a single GPU.

The DeepSpeed-Inference kernel for transformer attention achieves 3–4× throughput improvement over standard PyTorch for common model sizes.

## DeepSpeed with Hugging Face

The Hugging Face `accelerate` library and `transformers` `TrainingArguments` both support DeepSpeed as a backend, making it accessible without deep DeepSpeed expertise:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    deepspeed="ds_config.json",  # Single line enables DeepSpeed
    per_device_train_batch_size=4,
    num_train_epochs=3,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

## Choosing ZeRO Stages

Selecting the appropriate ZeRO stage involves balancing memory savings against communication overhead:

| Stage | Memory Saved | Communication Overhead | Best For |
| --- | --- | --- | --- |
| ZeRO-1 | ~4× optimizer states | Minimal | Models that fit with optimizer partitioning |
| ZeRO-2 | ~8× optimizer + grads | Moderate | Most fine-tuning scenarios |
| ZeRO-3 | Full partition (scales with GPUs) | High | Models that don't fit on single GPU |
| ZeRO-Infinity | Virtually unlimited | Very high | 100B+ models, limited GPU memory |

In practice:

- **ZeRO-2** is the most common choice for fine-tuning medium-large models (7B–30B) on multi-GPU setups.
- **ZeRO-3** is necessary for training models that are too large to fit even with ZeRO-2.
- **ZeRO-Infinity with CPU offload** is the option of last resort — it enables training large models on smaller hardware, but at significant throughput cost.

DeepSpeed has democratized large model training, making it feasible to train and fine-tune models that previously required hyperscaler infrastructure on university clusters, startup-scale GPU budgets, and consumer-accessible cloud instances.
