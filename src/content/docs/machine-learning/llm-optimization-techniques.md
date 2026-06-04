---
title: "LLM Optimization Techniques: Memory, Latency, and Cost"
description: "A comprehensive guide to optimizing large language models — from quantization and pruning to efficient fine-tuning and inference strategies for production deployment."
---

Deploying LLMs efficiently requires optimization across multiple dimensions: memory (fitting models on hardware), latency (responding quickly), and cost (keeping infrastructure affordable). This guide covers the complete landscape of LLM optimization.

## Optimization Overview

```python
class LLMOptimizationSpectrum:
    """Different optimizations at different levels."""
    
    # Data types
    fp32 → bf16 → fp16 → int8 → int4
    
    # Architecture
    full_attention → sparse_attention → linear_attention
    
    # Training
    full_fine_tune → LoRA → QLoRA → prompt_tuning
    
    # Inference
    naive_serving → batching → paged_attention → speculative_decoding
    
    # Deployment
    single_gpu → tensor_parallelism → pipeline_parallelism → distributed
```

## Quantization

### Post-Training Quantization (PTQ)

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization with GPTQ
from optimum.gptq import GPTQQuantizer

quantizer = GPTQQuantizer(
    bits=4,
    dataset="c4",
    block_size=128,
    damp_percent=0.01,
)

# Quantize model
quantized_model, quantization_config = quantizer.quantize_model(
    "meta-llama/Llama-2-7b",
    save_dir="./llama-2-7b-gptq"
)
```

### Activation-Aware Weight Quantization (AWQ)

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    safetensors_path="./llama-2-7b-awq/model.safetensors"
)

# AWQ finds optimal quantization scaling based on activation magnitude
quantize_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
}

model.quantize(tokenizer, quantize_config)
```

### GGUF Quantization

```python
# Using llama.cpp for GGUF conversion
import subprocess

# Convert HF model to GGUF
subprocess.run([
    "python", "convert-hf-to-gguf.py",
    "./llama-2-7b",
    "--outtype", "q4_0",  # or q4_1, q5_0, q5_1, q8_0
    "--outfile", "./llama-2-7b-q4_0.gguf"
])

# Load and use with llama.cpp
import llama_cpp

model = llama_cpp.Llama.from_pretrained(
    "./llama-2-7b-q4_0.gguf",
    n_ctx=4096,
    n_gpu_layers=35,  # Offload to GPU
)
```

### bitsandbytes for Training

```python
from transformers import BitsAndBytesConfig

# 4-bit training configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # Saves more memory
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=quantization_config,
    device_map="auto",
)
```

## Model Pruning

### Magnitude Pruning

```python
def magnitude_prune(model, sparsity=0.3):
    """Remove weights with smallest magnitude."""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            threshold = np.percentile(
                param.abs().cpu().numpy(),
                sparsity * 100
            )
            mask = param.abs() > threshold
            param.data = param.data * mask.float()
            param.grad = None  # No gradient for pruned weights
    return model
```

### Structured Pruning (Layer Removal)

```python
def remove_layers(model, layer_indices):
    """Remove specific transformer layers."""
    remaining_layers = []
    for i, layer in enumerate(model.transformer.h):
        if i not in layer_indices:
            remaining_layers.append(layer)
    model.transformer.h = nn.ModuleList(remaining_layers)
    return model

# Example: Remove half the layers
model = remove_layers(model, [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35])
```

### Pruning with Distillation

```python
class PruningWithDistillation:
    def __init__(self, teacher, student, temperature=2.0):
        self.teacher = teacher
        self.student = student
    
    def train(self, student_layers, teacher_layers):
        """Train student to match teacher outputs."""
        # Student makes predictions
        student_out = self.student(...)
        
        # Teacher makes predictions (frozen)
        with torch.no_grad():
            teacher_out = self.teacher(...)
        
        # Distillation loss
        loss = F.kl_div(
            F.log_softmax(student_out / temperature),
            F.softmax(teacher_out / temperature),
            reduction='batchmean'
        ) * (temperature ** 2)
```

## Efficient Fine-Tuning

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rank of adaptation
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 6.28M || all params: 7.12B || trainable%: 0.09%
```

### QLoRA with Custom Config

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# Memory-efficient QLoRA setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for training
model = prepare_model_for_int8_training(model)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

### DoRA: Weight-Decomposed LoRA

```python
from peft import DoRAConfig

dora_config = DoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    init_type="lora",
)

model = get_peft_model(base_model, dora_config)
```

## Efficient Inference

### KV Cache Optimization

```python
class OptimizedKVCache:
    """Optimized KV cache with caching and eviction."""
    
    def __init__(self, max_cache_size=1000):
        self.cache = {}  # request_id -> KV cache
        self.lru = LRUCache(max_cache_size)
    
    def get(self, request_id):
        """Retrieve cached KV or compute."""
        if request_id in self.cache:
            self.lru.access(request_id)
            return self.cache[request_id]
        
        # Compute and cache
        kv = compute_kv_cache(request_id)
        self.cache[request_id] = kv
        self.lru.add(request_id)
        return kv
    
    def evict_if_needed(self):
        """Evict LRU items when cache is full."""
        while len(self.cache) > self.max_cache_size:
            evicted_id = self.lru.evict()
            del self.cache[evicted_id]
```

### Batch Size Optimization

```python
class DynamicBatcher:
    """Batcher that optimizes for throughput."""
    
    def __init__(self, max_batch_size=32, max_wait_time=0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
    
    def add(self, request):
        self.pending_requests.append(request)
    
    def step(self):
        # Wait for enough requests or timeout
        while len(self.pending_requests) < self.max_batch_size:
            if time_elapsed > self.max_wait_time:
                break
            wait()
        
        # Form batch
        batch_size = len(self.pending_requests)
        batch = self.pending_requests[:batch_size]
        self.pending_requests = self.pending_requests[batch_size:]
        
        # Run inference
        return run_batch(batch)
```

### Speculative Decoding

```python
class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, gamma=4):
        self.draft = draft_model      # Small, fast model
        self.target = target_model    # Large, accurate model
        self.gamma = gamma           # Draft tokens per step
    
    def generate(self, prompt, max_tokens=100):
        """Generate with draft-then-verify approach."""
        # Draft phase
        draft_tokens = []
        draft_probs = []
        
        for _ in range(self.gamma):
            token, prob = self.draft.predict_next(prompt + draft_tokens)
            draft_tokens.append(token)
            draft_probs.append(prob)
        
        # Verify phase: batch process all draft tokens
        target_probs = self.target.verify(prompt, draft_tokens)
        
        # Accept/reject each token
        accepted = []
        for i, (draft_tok, draft_p, target_p) in enumerate(
            zip(draft_tokens, draft_probs, target_probs)
        ):
            # Acceptance probability
            if target_p[draft_tok] / draft_p[draft_tok] >= random.random():
                accepted.append(draft_tok)
            else:
                # Rejection: use target's token and stop drafting
                break
        
        return accepted + [target_tokens[0]]  # Include target's first token
```

## Memory-Efficient Training

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def forward(self, x):
        # Checkpoint every other layer
        for i, layer in enumerate(self.layers):
            if i % 2 == 1:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x
```

### Gradient Accumulation

```python
# Simulate larger batch sizes with accumulation
effective_batch_size = 64
micro_batch_size = 4
gradient_accumulation_steps = effective_batch_size // micro_batch_size

optimizer.zero_grad()
for i in range(gradient_accumulation_steps):
    # Forward pass (micro batch)
    loss = model(micro_batch)
    
    # Backward pass (accumulate gradients)
    loss.backward()
    
    # Update after accumulation
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch.input)
        loss = criterion(outputs, batch.target)
    
    # Scale loss and backward
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
```

## Distributed Training

### DeepSpeed ZeRO

```python
import deepspeed

# ZeRO config for optimizer state partitioning
deepspeed_config = {
    "zero_optimization": {
        "stage": 1,  # 1: optimizer states, 2: gradients, 3: parameters
        "offload_optimizer": {
            "device": "cpu",  # Offload to CPU memory
        },
    },
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
}

# Initialize with DeepSpeed
model, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=deepspeed_config,
)
```

### Tensor Parallelism

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model with tensor parallelism
model = TensorParallelism(model, mp_group=dist.get_world_size())

# DDP wrapper
model = DDP(model, device_ids=[local_rank])

# Forward pass (automatic sharding)
outputs = model(inputs)
```

## Inference Serving Optimization

### vLLM Configuration

```python
from vllm import LLM, SamplingParams

# Optimized vLLM serving
llm = LLM(
    model="meta-llama/Llama-2-7b",
    
    # Quantization
    quantization="awq",
    
    # Parallelism
    tensor_parallel_size=2,
    
    # Memory optimization
    max_model_len=4096,
    enable_prefix_caching=True,
    
    # Batching
    max_num_batched_tokens=8192,
    max_num_seqs=64,
)

# Conservative sampling for consistent outputs
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    logprobs=0,  # Disable for speed
)
```

### TensorRT-LLM

```python
# TensorRT-LLM for maximum performance
from tensorrt_llm import Builder
from tensorrt_llm.runtime import ModelRunner

# Build optimized engine
builder = Builder()
engine = builder.build_engine(
    "meta-llama/Llama-2-7b",
    max_batch_size=32,
    max_input_len=2048,
    max_output_len=512,
    use_gpt_attention=True,
)

# Run with TensorRT
runner = ModelRunner(engine)
outputs = runner.generate(
    ["Explain quantum mechanics"],
    max_new_tokens=100,
)
```

## Cost Optimization Strategies

### Model Selection Matrix

| Model Size | Use Case | Cost/Tokens |
|------------|----------|-------------|
| 3-7B | Simple tasks, quick responses | $0.0001 |
| 13-34B | General purpose | $0.0003 |
| 70B+ | Complex reasoning, high quality | $0.001 |

### Caching Strategy

```python
class SemanticCache:
    def __init__(self, embedding_model, vector_store, similarity_threshold=0.95):
        self.embedder = embedding_model
        self.store = vector_store
        self.threshold = similarity_threshold
    
    def get_or_generate(self, query, generator):
        # Check cache
        query_embedding = self.embedder.encode(query)
        cached = self.store.search(query_embedding, top_k=1)
        
        if cached and cached[0].score > self.threshold:
            return cached[0].response  # Return cached
        
        # Generate new response
        response = generator(query)
        
        # Cache response
        self.store.add(query_embedding, response)
        
        return response
```

### Request Routing

```python
class SmartRouter:
    def __init__(self, small_model, large_model):
        self.small = small_model
        self.large = large_model
    
    def classify_query(self, query):
        """Determine if simple or complex query."""
        # Heuristic: query length, keyword detection
        if len(query.split()) < 10 and "explain" not in query.lower():
            return "simple"
        return "complex"
    
    def route(self, query):
        query_type = self.classify_query(query)
        
        if query_type == "simple":
            return self.small.generate(query)
        return self.large.generate(query)
```

LLM optimization is multidimensional. The right technique depends on your constraints: quantization for memory, batching for throughput, distillation for cost, and speculative decoding for latency. A well-optimized system combines multiple approaches for maximum efficiency.