---
title: Introduction to SGLang
description: Discover SGLang, a high-performance framework for fast execution and structured generation of LLM programs featuring RadixAttention, speculative execution, and multi-node serving.
---

SGLang (Structured Generation Language) is an open-source, high-performance execution engine and programming interface designed for Large Language Models (LLMs). Developed by researchers at LMSYS (UC Berkeley, Stanford, and UCSD), SGLang dramatically accelerates LLM inference, particularly for complex agentic workflows, multi-turn conversations, and structured JSON output generation.

## Why SGLang?

Standard LLM inference engines like vLLM or Hugging Face TGI excel at single-prompt throughput. However, modern AI applications rely heavily on **multi-call programs**:
- Multi-step reasoning chains (Tree of Thoughts, Graph of Thoughts)
- Structured schema extraction (JSON, Pydantic schemas)
- Agent loops requiring repeated system prompts and prefix sharing

Executing these patterns with standard engines incurs heavy re-computation overhead for shared prompt prefixes and token management. SGLang solves this by co-designing the programming model and the runtime execution engine.

## Core Technical Innovations

### 1. RadixAttention: Automatic Prefix KV-Cache Reuse

The core runtime innovation of SGLang is **RadixAttention**. Unlike static prefix caching, RadixAttention manages the Key-Value (KV) cache as a radix tree (patricia trie) in GPU memory.

- **Dynamic Prefix Search:** When a request arrives, SGLang searches the radix tree for the longest matching prefix.
- **LRU Cache Eviction:** Nodes in the radix tree are retained across requests and evicted using a Least Recently Used (LRU) policy when memory pressure increases.
- **Zero-Copy Forking:** Parallel decoding branches (such as beam search or self-consistency checks) share prompt KV nodes without duplication.

```
       [Root System Prompt]
           /          \
  [Tool Docs A]     [Tool Docs B]
       |                 |
  [Query 1]         [Query 2]
```

### 2. Compressed Finite State Machine (FSM) Decoding

Generating constrained structured outputs (e.g., valid JSON matching a specific schema) often slows down generation due to token-by-token regex checking. SGLang uses compressed FSM decoding:
- Pre-compiles JSON schemas or regex into efficient finite-state automata.
- Masks invalid tokens at the GPU logits level in parallel with vector operations.
- Jumps directly through static syntax tokens (e.g., `{"name": "`) without model inference passes.

## SGLang Frontend Language Primitives

SGLang provides a flexible Python-based domain-specific language (DSL) to orchestrate complex LLM logic cleanly.

```python
import sglang as sgl

@sgl.function
def text_summarization_and_qa(s, text):
    s += sgl.user(f"Summarize the following text:\n{text}")
    s += sgl.assistant(sgl.gen("summary", max_tokens=128))
    
    s += sgl.user("Based on the text and summary, what is the main takeaway?")
    s += sgl.assistant(sgl.gen("takeaway", max_tokens=64))

# Execute program with engine backend
state = text_summarization_and_qa.run(
    text="SGLang achieves state-of-the-art inference speed by optimizing KV cache retention...",
    backend=sgl.RuntimeEndpoint("http://localhost:30000")
)

print("Summary:", state["summary"])
print("Takeaway:", state["takeaway"])
```

### Parallel Branching (`sgl.fork`)

SGLang allows easy parallel execution branches (e.g., self-consistency or multi-agent debate) while automatically sharing parent KV caches:

```python
@sgl.function
def parallel_reasoning(s, question):
    s += sgl.user(question)
    # Fork into 3 independent paths
    forks = s.fork(3)
    for i, f in enumerate(forks):
        f += sgl.assistant(sgl.gen(f"reasoning_{i}", max_tokens=256, temperature=0.7))
    s.join(forks)
```

## Performance & Architecture Comparison

| Feature | SGLang | vLLM | TensorRT-LLM |
|---|---|---|---|
| **Prefix Caching** | Dynamic Radix Tree (RadixAttention) | Hash-based Block Cache | Static / Manual |
| **Structured Output** | Fast Compressed FSM / Outlines | Regex Masking | Basic Json Schema |
| **Multi-Turn KV Reuse** | Automatic across requests | Manual Session ID | Manual |
| **Programming Interface** | Native Python DSL (`sgl.gen`) | OpenAI Compatible API | C++ / Python API |
| **Speculative Decoding** | Supported (EAGLE, Medusa) | Supported | Supported |

## Deploying SGLang Server

SGLang can be launched as an OpenAI-compatible HTTP backend:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 30000 \
    --mem-fraction-static 0.8
```

You can then query it using standard OpenAI API clients or the SGLang native client.

## Summary

SGLang bridges the gap between high-level agentic programming and low-level GPU acceleration. By leveraging RadixAttention for automatic KV-cache sharing and FSM-driven structured decoding, it achieves up to 2x-5x higher throughput on multi-turn and agent workloads compared to traditional serving engines.

## Further Reading

- LMSYS SGLang GitHub Repository: `github.com/sgl-project/sglang`
- Zheng et al. (2024), *Efficiently Programming Large Language Models with SGLang*
- RadixAttention Deep Dive in SGLang Documentation
