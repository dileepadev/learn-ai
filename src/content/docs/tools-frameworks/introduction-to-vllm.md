---
title: "Introduction to vLLM: High-Throughput LLM Inference"
description: "Learn how vLLM achieves 10x higher throughput for LLM serving using PagedAttention, continuous batching, and efficient memory management."
---

vLLM is an open-source inference engine that has revolutionized LLM serving. By reimagining how KV caches are managed, vLLM achieves dramatically higher throughput than naive implementations. This guide covers the architecture and practical usage.

## Why vLLM Matters

Traditional LLM serving suffers from memory fragmentation:

```python
# Naive KV cache management
class NaiveKVCache:
    def __init__(self, max_batch_size=32, max_seq_len=2048, hidden_size=4096):
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        
        # Allocate full memory for each request upfront
        # Wastes memory when requests are short
        self.cache = torch.zeros(max_batch_size, max_seq_len, hidden_size)
    
    def allocate(self, request_id, seq_len):
        # Assign fixed slot in pre-allocated memory
        self.allocations[request_id] = (0, seq_len)
    
    def free(self, request_id):
        # Just mark as free - memory not reused
        del self.allocations[request_id]
```

vLLM's innovations:
- **PagedAttention**: Memory-efficient KV cache like virtual memory.
- **Continuous batching**: Dynamic batch sizes for variable-length sequences.
- **High-throughput serving**: 10x better than HuggingFace Transformers.

## PagedAttention

```python
class PagedKVCache:
    def __init__(self, num_layers, num_heads, head_dim, block_size=16):
        self.block_size = block_size
        self.num_layers = num_layers
        
        # Logical blocks (what the model sees)
        self.logical_blocks = {}
        
        # Physical blocks (actual GPU memory)
        self.free_physical_blocks = set()
        self.physical_blocks = {}  # block_num -> tensor
        
        # Initialize free blocks
        for i in range(num_physical_blocks):
            self.free_physical_blocks.add(i)
    
    def allocate(self, request_id, num_tokens):
        """Allocate logical blocks for a request."""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        blocks = []
        
        for _ in range(num_blocks):
            if not self.free_physical_blocks:
                # Need to evict or allocate more
                raise OutOfMemoryError
            
            physical_block = self.free_physical_blocks.pop()
            blocks.append(physical_block)
            self.physical_blocks[physical_block] = {}
        
        self.logical_blocks[request_id] = blocks
        return blocks
    
    def copy(self, src_request, dst_request, src_offset, dst_offset, num_blocks):
        """Copy blocks between requests (for prefix caching)."""
        src_blocks = self.logical_blocks[src_request]
        dst_blocks = self.logical_blocks[dst_request]
        
        for i in range(num_blocks):
            src_physical = src_blocks[src_offset + i]
            dst_physical = dst_blocks[dst_offset + i]
            
            # Copy tensor data
            self.physical_blocks[dst_physical] = self.physical_blocks[src_physical].clone()
```

## Continuous Batching

```python
class ContinuousBatcher:
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.active_requests = []
        self.pending_requests = queue.Queue()
    
    def add_request(self, request):
        """Add new request to queue."""
        if len(self.active_requests) < self.max_batch_size:
            self.active_requests.append(request)
        else:
            self.pending_requests.put(request)
    
    def step(self):
        """Run one inference step."""
        # Prepare batch
        batch = self._prepare_batch()
        
        # Run inference
        outputs = run_model(batch)
        
        # Process completions
        completed = []
        remaining = []
        
        for i, request in enumerate(batch.requests):
            output = outputs[i]
            
            if output.finished:
                completed.append(request)
            else:
                request.append_token(output.new_token)
                remaining.append(request)
        
        # Replace completed requests with pending
        self.active_requests = remaining
        
        while self.active_requests and self.pending_requests:
            if len(self.active_requests) >= self.max_batch_size:
                break
            self.active_requests.append(self.pending_requests.get())
        
        return completed, remaining
```

## Using vLLM

### Basic Usage

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,  # Use multiple GPUs
    dtype="half",            # Use fp16
    enforce_eager=True,      # Don't use CUDA graphs for debugging
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop=["User:", "\n###"],
)

# Generate completions
prompts = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate Fibonacci numbers.",
    "What are the benefits of exercise?",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("---")
```

### OpenAI-Compatible API

```python
# vLLM provides OpenAI-compatible endpoints
import openai

# Configure client to use vLLM server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # Not required for local vLLM
)

# Use exactly like OpenAI API
response = client.chat.completions.create(
    model="llama-2-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Multi-LoRA Serving

```python
from vllm import LLM, SamplingParams
from peft import PeftModel

# Load base model with LoRA support
llm = LLM(
    model="meta-llama/Llama-2-7b",
    enable_lora=True,     # Enable LoRA support
    max_loras=4,          # Maximum loaded LoRAs
    max_lora_rank=16,     # Maximum LoRA rank
)

# Serve multiple adapters
llm.set_adapters(["adapter_1", "adapter_2", "adapter_3"])

# Generate with specific adapter
outputs = llm.generate(
    ["Hello, I need help with coding."],
    SamplingParams(temperature=0.7),
    lora_request="adapter_1"  # Use specific adapter
)
```

## Performance Optimization

### Tensor Parallelism

```python
# For multi-GPU serving
llm = LLM(
    model="meta-llama/Llama-2-70b",
    tensor_parallel_size=8,  # Spread across 8 GPUs
    pipeline_parallel_size=1,
)

# vLLM handles all the communication internally
# No code changes needed beyond specifying tensor_parallel_size
```

### Quantization

```python
# Use quantized models for better throughput
llm = LLM(
    model="TheBloke/Llama-2-7b-Chat-GPTQ",
    quantization="gptq",  # Or "awq", "squeezellm"
)

# Benefits:
# - 2-4x memory reduction
# - Higher batch sizes
# - Lower latency for memory-bound workloads
```

### Prefix Caching

```python
# Enable prefix caching for repeated contexts
llm = LLM(
    model="meta-llama/Llama-2-7b",
    enable_prefix_caching=True,  # Cache KV for repeated prefixes
)

# Example: System prompt caching
# First request processes full prompt
# Subsequent requests reuse cached KV for system prompt
```

## Monitoring and Debugging

### Logits Processing

```python
# Access logits for analysis
outputs = llm.generate(
    ["Hello, my name is"],
    SamplingParams(
        temperature=0.7,
        logprobs=5,  # Return log probabilities
        top_logprobs=5,
    ),
)

for output in outputs[0].outputs[0].logprobs:
    print(f"Token: {token}, Logprob: {logprob}")
```

### Performance Metrics

```python
# vLLM exposes Prometheus metrics at /metrics
import requests

response = requests.get("http://localhost:8000/metrics")
print(response.text)

# Key metrics:
# - vllm:request_throughput_tok_per_second
# - vllm:generation_token_count
# - vllm:num_requests_waiting
# - vllm:gpu_cache_usage
```

## Comparing vLLM to Alternatives

| Feature | vLLM | TensorRT-LLM | HuggingFace |
|---------|------|--------------|-------------|
| **Ease of use** | High | Medium | High |
| **Throughput** | Very High | Very High | Medium |
| **Latency** | Low | Lowest | Medium |
| **Model support** | Wide | NVIDIA-optimized | All models |
| **Customization** | Medium | High | Very High |
| **Hardware** | All GPUs | NVIDIA | All GPUs |

## Production Deployment

### Kubernetes

```yaml
# vLLM deployment for Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-7b-chat-hf"
        - name: TP_SIZE
          value: "1"
```

### Health Checks

```python
# vLLM provides health endpoints
import requests

# Liveness check
requests.get("http://localhost:8000/health")

# Readiness check (includes model loaded status)
requests.get("http://localhost:8000/health")
```

vLLM has become the standard for high-throughput LLM serving. Its combination of PagedAttention, continuous batching, and memory-efficientKV cache management enables serving models that would be impossible with naive implementations.