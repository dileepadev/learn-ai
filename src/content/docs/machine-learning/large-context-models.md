---
title: "Large Context Models: Extending LLM Context Windows"
description: "Learn how modern LLMs extend context windows to 100K+ tokens — covering techniques like position interpolation, YaRN, LongLoRA, and efficient long-context inference."
---

The ability to process long contexts — hundreds of thousands of tokens — is one of the most important capabilities for building sophisticated LLM applications. This guide covers how context windows are extended and optimized.

## Why Context Length Matters

Long context enables:
- **Full document analysis**: Process entire PDFs, books, or codebases.
- **Extended conversations**: Maintain coherent multi-session chats.
- **Complex reasoning**: Work with large codebases, legal contracts, or research papers.
- **In-context learning**: Show many examples without running out of space.

Most models ship with 4K–8K token contexts. Extending to 128K+ requires modifications.

## Position Interpolation

The original approach to context extension:

```python
def interpolate_positions(positions, old_max, new_max):
    """Scale positions to fit new context window.
    
    If positions are [0, 1, 2, ..., old_max-1],
    rescale to [0, 1, 2, ..., new_max-1] * (old_max / new_max)
    """
    scale = old_max / new_max
    return positions * scale

# Original: positions 0 to 2048 for 2K context
# Interpolated: positions 0 to 8192 for 8K context
```

**How it works**:
1. The model was trained with positions 0 to N-1.
2. To extend to 2N, multiply all positions by N/(2N) = 0.5.
3. The model sees positions 0, 0.5, 1, ..., N-0.5.
4. Fine-tune on the extended context to adapt.

**Key insight**: The model never sees actual positions beyond N, so it can't extrapolate. Interpolation keeps it in-distribution.

## YaRN: Yet another RoPE extensioN

YaRN improves on simple interpolation by applying different scaling to different frequency components:

```python
class YaRNPositionalEncoding:
    def __init__(self, base_model, ctx_len=32768, factor=2.0):
        self.base_model = base_model
        self.factor = factor
        
        # Split dimensions into low and high frequency
        self.base = 10000
        self.high_freq_factor = 4  # Scale high freq less
        self.low_freq_factor = 1   # Keep low freq as-is
        
        # Calculate scale for each dimension
        self.scale = self._compute_scales(ctx_len)
    
    def _compute_scales(self, ctx_len):
        """Compute position scale for each frequency."""
        scales = []
        for i in range(dim_size):
            base_freq = self.base ** (2 * i / dim_size)
            if base_freq > threshold:
                scales.append(ctx_len / (ctx_len * self.high_freq_factor))
            else:
                scales.append(1.0)  # No scaling for low freq
        return scales
    
    def forward(self, positions):
        scaled_positions = []
        for i, pos in enumerate(positions):
            scaled = pos * self.scale[i]
            scaled_positions.append(scaled)
        return self.base_model(pos, *scaled_positions)
```

**Why frequency matters**:
- Low frequency (slowly varying): Encodes coarse position (paragraphs, sections).
- High frequency (fast varying): Encodes fine position (tokens within a word).

High frequency needs finer resolution; low frequency can be interpolated more aggressively.

## LongLoRA: Efficient Long-Context Fine-Tuning

LongLoRA combines three techniques for efficient context extension:

### 1. Sparse Attention Shift

```python
class LongLoRAAttention:
    def __init__(self, attention, shift_size=7):
        self.attention = attention
        self.shift_size = shift_size
    
    def forward(self, x):
        # Shift the input so each token sees different neighbors
        x = shift_tokens_right(x, self.shift_size)
        
        # Apply regular (sparse) attention
        output = self.attention(x)
        
        # Shift back
        output = shift_tokens_left(output, self.shift_size)
        return output
```

The shift enables "sliding window" attention at inference time while maintaining long-context learning.

### 2. Context Expansion with LoRA

```python
from peft import LoraConfig, get_peft_model

# Use LoRA on embedding and normalization layers
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["embed_tokens", "norm"],
)

model = get_peft_model(model, lora_config)
```

### 3. Grouped-Query Attention (GQA)

Use more efficient attention patterns to reduce memory:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, num_heads, num_kv_groups):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = hidden_size // num_heads
        
        # Fewer KV heads than query heads
        self.q_proj = nn.Linear(hidden, num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden, num_kv_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden, num_kv_groups * self.head_dim)
```

## Efficient Long-Context Inference

### Ring Attention

Process context in chunks across multiple devices:

```python
def ring_attention_forward(query, key, value, block_size=4096):
    """Ring attention: each device processes a block."""
    num_devices = get_world_size()
    
    # Split QKV across devices
    local_q = split_along_dim(query, block_size)
    local_k = split_along_dim(key, block_size)
    local_v = split_along_dim(value, block_size)
    
    # Compute local attention
    local_output = scaled_dot_product_attention(local_q, local_k, local_v)
    
    # Ring communication: pass KV to next device
    for i in range(num_devices):
        next_k = send_to_device(local_k, (i + 1) % num_devices)
        next_v = send_to_device(local_v, (i + 1) % num_devices)
        
        # Cross-block attention
        cross_output = scaled_dot_product_attention(
            local_q, next_k, next_v
        )
        local_output = merge_outputs(local_output, cross_output)
    
    return local_output
```

### KV Cache Optimization

```python
# Paged KV cache (vLLM-style)
class PagedKVCache:
    def __init__(self, num_layers, num_heads, head_dim, block_size=16):
        self.tables = {}  # page_number -> tensor
        self.free_pages = set()
        self.block_size = block_size
    
    def append(self, layer, key_layer, value_layer):
        """Append new tokens to cache."""
        for l in range(num_layers):
            num_new_tokens = key_layer.shape[1]
            
            # Allocate new pages if needed
            while num_new_tokens > 0:
                page = self.allocate_page()
                # Copy tokens to page
                num_new_tokens -= self.block_size
    
    def get(self, layer, positions):
        """Retrieve KV values for specific positions."""
        pages = [self.get_page(pos) for pos in positions]
        return cat_along_dim(pages, dim=1)
```

### Prefix Caching for Repeated Contexts

```python
class PrefixKVCache:
    def __init__(self, cache: PagedKVCache):
        self.cache = cache
        self.hash_to_pagenum = {}  # KV hash -> page numbers
        self.lru = LRUCache()
    
    def lookup(self, prompt):
        """Check if prompt prefix exists in cache."""
        prompt_hash = hash(prompt[:self.prefix_length])
        
        if prompt_hash in self.hash_to_pagenum:
            pages = self.hash_to_pagenum[prompt_hash]
            self.lru.access(prompt_hash)
            return pages
        return None
    
    def store(self, prompt, kv_tensor):
        """Store KV cache for future reuse."""
        if len(prompt) >= self.prefix_length:
            prompt_hash = hash(prompt[:self.prefix_length])
            pages = self.cache.copy_to_new_pages(kv_tensor)
            self.hash_to_pagenum[prompt_hash] = pages
```

## Context Length Benchmarks

| Model | Context | Extension Method | Quality |
|-------|---------|------------------|---------|
| LLaMA-2 | 4K → 32K | Interpolation | Good |
| LLaMA-2 | 4K → 100K | YaRN | Very Good |
| Mistral | 32K (native) | Sliding window | Excellent |
| LongLoRA | 8K → 100K | Shift + LoRA | Excellent |
| Claude 2.1 | 200K (native) | Native | Excellent |

## Evaluating Long-Context Models

### Needle in a Haystack

Test if the model can find information buried in long context:

```python
def needle_in_haystack(model, context_length=100000, num_needles=5):
    """Place needles at various depths and test retrieval."""
    haystack = generate_random_text(context_length)
    
    needles = []
    for depth in [0.0, 0.25, 0.5, 0.75, 1.0]:
        needle = f"SPECIAL_TOKEN_{random_string()}"
        position = int(depth * context_length)
        haystack = insert_at(haystack, position, needle)
        needles.append((needle, position))
    
    # Query for each needle
    results = []
    for needle, position in needles:
        response = model.generate(f"Find the special token in this text: {haystack}")
        results.append({
            "needle": needle,
            "true_position": position,
            "retrieved_position": extract_position(response),
        })
    
    return results
```

### Key Tasks for Long-Context Evaluation

| Task | Description | What It Tests |
|------|-------------|---------------|
| **Needle retrieval** | Find a fact in long context | Basic retrieval |
| **Multi-needle** | Find multiple facts | Complex retrieval |
| **KV pair matching** | Match keys to values | Associative memory |
| **BookQA** | Answer questions about books | Full context understanding |
| **Code completion** | Complete code from file | Long-range dependencies |

## Practical Considerations

### Input Length Limits

```python
def truncate_to_context_limit(prompt, max_tokens, context_limit):
    """Truncate prompt while preserving critical information."""
    available = context_limit - max_tokens  # Reserve for output
    
    if len(prompt) <= available:
        return prompt
    
    # Keep system prompt
    if has_system_prompt(prompt):
        system = extract_system_prompt(prompt)
        available -= len(system)
    
    # Keep recent messages (for chat)
    recent = truncate_from_end(prompt, available)
    
    return combine_prompt(system, recent)
```

### Streaming and Chunked Processing

For contexts that exceed limits:

```python
def process_long_document(document, chunk_size=8000, overlap=500):
    """Process document in chunks with overlap."""
    chunks = []
    for i in range(0, len(document), chunk_size - overlap):
        chunk = document[i:i + chunk_size]
        # Process each chunk
        result = process_chunk(chunk)
        chunks.append(result)
    
    # Aggregate results
    return aggregate_results(chunks)
```

Long context is essential for sophisticated LLM applications. The techniques here — position interpolation, YaRN, LongLoRA, and efficient inference — enable processing contexts of 100K+ tokens reliably.