---
title: Distributed Training for Large Language Models
description: How frontier LLMs are trained across thousands of GPUs — covering data parallelism, tensor parallelism, pipeline parallelism, the ZeRO optimizer, FSDP, DeepSpeed, and Megatron-LM strategies for scaling training beyond a single device.
---

**Distributed training** is the practice of spreading the computation of training a neural network across multiple accelerators (GPUs or TPUs). For large language models with hundreds of billions of parameters, distributed training is not optional — the model itself cannot fit on a single device, and even smaller models benefit from parallelism to reach the compute scale needed for good performance. Understanding how this works is essential for anyone building or deploying frontier AI systems.

## Why a Single GPU is Not Enough

A 70B-parameter model in 16-bit (bfloat16) precision requires approximately 140 GB of memory just to store the weights. During training, gradient and optimizer state storage (with Adam) adds roughly 4× the model size — totaling ~560 GB. The largest consumer GPUs have 80 GB VRAM. Even enterprise accelerators max out at 192 GB (H200). Training 70B+ models requires aggregating memory across many devices.

Beyond memory, the compute required to train frontier models is staggering — GPT-4 class models require estimated exaFLOPs of training compute, achievable only by running thousands of accelerators in parallel for months.

## The Three Axes of Parallelism

Distributed LLM training uses combinations of three distinct parallelism strategies, each targeting a different bottleneck.

### 1. Data Parallelism (DP)

**Data parallelism** is the simplest form. Each device holds a complete copy of the model. The training batch is split across devices; each device computes gradients on its shard of the batch. After the forward and backward pass, gradients are **all-reduced** (summed across devices) to keep all replicas synchronized.

**All-reduce communication**: The bandwidth bottleneck in data parallelism. With $N$ devices and model size $M$, each step requires transmitting $2M(N-1)/N \approx 2M$ bytes (ring all-reduce).

**Limitation**: Each device must hold the full model. For 70B+ parameter models, even with mixed precision, this quickly exceeds single-device memory.

### 2. Tensor Parallelism (TP)

**Tensor parallelism** (also called **intra-layer model parallelism**) splits individual weight matrices across devices. For a linear layer $Y = XA$, the weight matrix $A$ can be partitioned column-wise across $N$ devices:

$$Y = X[A_1 | A_2 | \ldots | A_N] = [XA_1 | XA_2 | \ldots | XA_N]$$

Each device computes its shard of the output independently, then results are concatenated or reduced. The attention mechanism and MLP blocks in transformers decompose naturally this way.

**Megatron-LM** (NVIDIA) pioneered and formalized tensor parallelism for transformers, demonstrating efficient scaling to thousands of GPUs.

**Communication cost**: Each tensor-parallel layer requires an **all-reduce** (for row-parallel) or **all-gather** (for column-parallel) communication. These are frequent — every forward and backward pass through each layer — so TP requires high-bandwidth interconnects (NVLink within a node).

**Practical range**: Tensor parallelism typically scales to the number of GPUs within a single node (8 GPUs), not across nodes, due to bandwidth requirements.

### 3. Pipeline Parallelism (PP)

**Pipeline parallelism** assigns different transformer layers to different devices. Device 1 handles layers 1–8; device 2 handles layers 9–16; and so on. Data flows sequentially through the pipeline.

**Naive pipeline**: A major inefficiency — while device 2 processes the output of device 1, device 1 sits idle waiting for the next batch. **GPU utilization can drop to 1/N** (pipeline bubble).

**Micro-batching**: The solution is to split each training batch into **micro-batches**. As soon as device 1 finishes processing micro-batch 1, it starts on micro-batch 2 while device 2 processes micro-batch 1 — significantly reducing the bubble fraction.

**GPipe and PipeDream** introduced these micro-batching and asynchronous pipeline variants. Megatron-LM's interleaved schedule further reduces the bubble to approximately $\frac{1}{N \cdot M}$ where $N$ is pipeline stages and $M$ is micro-batches.

**Communication cost**: Only **activation tensors** (not weights) are passed between pipeline stages — much lower bandwidth requirement than TP. PP can scale across nodes.

## 3D Parallelism

In practice, all three strategies are combined into **3D parallelism**:

- **TP** within a node (8 GPUs, NVLink interconnect).
- **PP** across nodes (pipeline stages spanning multiple machines).
- **DP** across pipeline replicas (multiple copies of the full TP+PP model).

A 1024-GPU training job might use TP=8, PP=8, DP=16 — simultaneously splitting layers, sharding matrices, and duplicating across data parallel replicas.

## The ZeRO Optimizer

Even with model parallelism, optimizer state remains a major memory consumer. **ZeRO** (Zero Redundancy Optimizer), introduced by Microsoft DeepSpeed, eliminates redundant copies of model states across data-parallel ranks.

### ZeRO Stages

**ZeRO Stage 1**: Partition optimizer states across DP ranks. Each rank stores only $1/N$ of the Adam moment tensors.

**ZeRO Stage 2**: Additionally partition gradients across DP ranks.

**ZeRO Stage 3**: Additionally partition model parameters across DP ranks. All ranks together hold one copy of the model.

| Stage | Memory Reduction | Communication Overhead |
| --- | --- | --- |
| Stage 1 (optimizer states) | 4× | Minimal |
| Stage 2 (+ gradients) | 8× | Low |
| Stage 3 (+ parameters) | 64× | Higher (all-gather params at each layer) |

ZeRO Stage 3 enables training models much larger than a single device's memory, essentially providing a form of model parallelism through data-parallel communication. The trade-off is increased communication volume compared to standard DP.

### ZeRO-Infinity

**ZeRO-Infinity** extends ZeRO Stage 3 to offload parameters, gradients, and optimizer states to CPU RAM and NVMe SSDs, enabling training of models larger than aggregate GPU memory — at the cost of I/O bandwidth.

## Fully Sharded Data Parallel (FSDP)

**FSDP** (Fully Sharded Data Parallel) is PyTorch's native implementation of ZeRO Stage 3 semantics. It shards model parameters, gradients, and optimizer states across DP ranks. During the forward and backward pass, each layer's full parameters are temporarily reconstructed via **all-gather** before the computation, then re-sharded afterward.

FSDP is the standard training strategy for LLaMA-family models and most open-source LLM training:

- Native PyTorch integration — no separate library required.
- Composable with `torch.compile` and gradient checkpointing.
- `fsdp_config` in HuggingFace Trainer supports FSDP out of the box.

## DeepSpeed

**DeepSpeed** (Microsoft) is a comprehensive distributed training library implementing ZeRO, along with:

- **DeepSpeed-MoE**: Efficient training of Mixture-of-Experts models.
- **Curriculum learning**: Dynamic data ordering for faster convergence.
- **Activation checkpointing**: Recompute activations during backward pass instead of storing them (trades compute for memory).
- **Communication compression**: Gradient compression for bandwidth-limited deployments.
- **DeepSpeed-Inference**: Optimized distributed inference with ZeRO-Inference.

## Megatron-LM

**Megatron-LM** (NVIDIA) is the training framework used for GPT-3, Megatron-Turing NLG, and many other frontier models. It provides:

- Highly optimized tensor and pipeline parallelism implementations.
- Fused CUDA kernels for transformer layers.
- Sequence parallelism: Distributing the sequence-length dimension across TP ranks for layer norm and dropout operations.
- Integration with cuDNN, NCCL (NVIDIA's collective communication library), and InfiniBand networking.

Most frontier models (GPT-4, LLaMA, Falcon) are trained using combinations of Megatron-LM, DeepSpeed, or proprietary versions of these approaches.

## Gradient Checkpointing (Activation Recomputation)

A memory technique orthogonal to parallelism: instead of storing all intermediate activations for the backward pass, **recompute** them during backpropagation. This reduces memory usage by roughly the square root of the number of layers at the cost of ~33% additional compute.

Almost all large-scale LLM training uses activation recomputation in conjunction with parallelism strategies.

## Communication Infrastructure

Efficient distributed training demands specialized networking:

- **NVLink/NVSwitch**: NVIDIA's high-bandwidth GPU-to-GPU interconnect. 900 GB/s bidirectional bandwidth on H100 NVLink. Essential for tensor parallelism.
- **InfiniBand**: High-bandwidth, low-latency network between nodes. HDR/NDR InfiniBand at 200–400 Gbps. Used for data parallelism gradient all-reduce and pipeline communication.
- **RoCE (RDMA over Converged Ethernet)**: Lower-cost alternative to InfiniBand, used in cloud clusters.

The ratio of compute-to-communication bandwidth determines the practical scaling efficiency and the optimal parallelism strategy for a given cluster configuration.

## Practical Efficiency: MFU

**Model FLOP Utilization (MFU)** measures what fraction of theoretical peak FLOP/s a training run achieves. A well-optimized training run at scale achieves 40–60% MFU on H100 clusters — implying that 40–60% of the GPU's theoretical compute is doing useful model computation, with the rest spent on communication, memory transfers, and overhead.

Improving MFU is one of the primary engineering goals in LLM infrastructure teams. The difference between 30% and 60% MFU at the scale of a frontier training run represents hundreds of thousands of dollars and weeks of training time.

## Summary

Distributed training for LLMs combines multiple complementary strategies:

- **Data parallelism** scales batch size across replicas.
- **Tensor parallelism** splits individual layers across devices within a node.
- **Pipeline parallelism** assigns layer ranges to pipeline stages across nodes.
- **ZeRO / FSDP** eliminates redundant memory copies without changing the parallelism topology.
- **Activation recomputation** trades compute for memory.

Mastering the interplay of these techniques — and the communication costs that accompany each — is what separates efficient large-scale training from wasted GPU-hours.
