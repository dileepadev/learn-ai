---
title: Tensor Parallelism vs. Pipeline Parallelism for Large Model Training
description: Learn how tensor parallelism and pipeline parallelism split a single large model across multiple devices, and how they combine in practice.
---

When a model is too large to fit on a single GPU even with a batch size of one, training requires splitting the model itself across multiple devices — a fundamentally different problem from data parallelism, which replicates the whole model and splits only the training data.

## Tensor Parallelism

Tensor parallelism splits individual layers' computations across devices — for example, splitting a large matrix multiplication in a transformer's feedforward or attention layer so each device computes only a portion of the output, then communicating partial results between devices as needed to complete each layer's full computation:

```text
Layer weight matrix W split column-wise across 4 GPUs
each GPU computes: partial_output = input @ W_shard
combine partial outputs across GPUs (all-reduce) -> full layer output
```

This requires frequent, low-latency communication between devices within every single layer's forward and backward pass, which is why tensor parallelism is typically only practical across GPUs connected by very high-bandwidth interconnects (like NVLink within a single server), since the communication overhead would dominate training time over slower network connections between separate machines.

## Pipeline Parallelism

Pipeline parallelism instead splits the model by layer, assigning different contiguous groups of layers to different devices, so a single training example's forward pass flows through devices sequentially like a pipeline:

```text
GPU 0: layers 1-8   -> GPU 1: layers 9-16   -> GPU 2: layers 17-24   -> GPU 3: layers 25-32
```

Naive pipeline parallelism leaves most devices idle most of the time, since each device can only work on its layers once the previous device has finished and passed data forward — micro-batching (splitting each training batch into smaller chunks and pipelining them through the stages, similar in spirit to instruction pipelining in CPU architecture) keeps more devices busy simultaneously by overlapping different micro-batches at different pipeline stages, substantially reducing this idle time or "pipeline bubble."

## Combining Both with Data Parallelism

Training the largest models combines all three parallelism strategies simultaneously: tensor parallelism within a server (exploiting fast intra-server interconnects for layer-splitting communication), pipeline parallelism across groups of servers (tolerating higher inter-server communication latency for the less frequent, layer-boundary communication pipeline parallelism requires), and data parallelism replicating this entire tensor-plus-pipeline-parallel setup across multiple such groups to increase total training throughput.

## Practical Guidance

Reach for tensor parallelism first when a single layer's weights don't fit in one GPU's memory, and you have high-bandwidth interconnects available between devices. Reach for pipeline parallelism when the model fits reasonably within available high-bandwidth groupings but the whole model doesn't fit on one device, and you're willing to manage pipeline bubble overhead through careful micro-batching. For most practical large-model training, use an existing framework (like DeepSpeed or Megatron-LM) that implements and tunes these combined strategies, rather than implementing the communication patterns manually.
