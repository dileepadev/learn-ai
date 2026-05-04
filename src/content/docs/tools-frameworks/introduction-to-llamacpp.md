---
title: "Introduction to llama.cpp"
description: "A practical guide to llama.cpp, the high-performance C/C++ inference engine that enables local LLM inference on commodity hardware through efficient quantization, cross-platform support, and an active open-source ecosystem."
---

## What Is llama.cpp?

**llama.cpp** is an open-source C/C++ implementation of LLM inference, originally created by Georgi Gerganov in 2023 for running Meta's LLaMA model locally on a MacBook. It has since evolved into a comprehensive, highly optimized inference engine supporting dozens of model architectures and running on virtually any hardware — from Raspberry Pi to modern NVIDIA/AMD GPUs.

The project's defining characteristic is its commitment to **zero dependencies and maximum portability**: it requires no Python, no CUDA SDK (though CUDA is supported for acceleration), and compiles with standard C compilers. This makes it the engine of choice for embedding LLMs in native applications, running models on edge devices, and achieving maximum inference efficiency on commodity hardware.

---

## Why llama.cpp Matters

Before llama.cpp, running large language models required:
- Powerful NVIDIA GPUs with large VRAM (40–80 GB for 70B models).
- Python environments with heavy frameworks (PyTorch, Transformers).
- Cloud infrastructure for anything production-grade.

llama.cpp changed the calculus by introducing **aggressive quantization** — representing model weights in 4-bit or even 2-bit integers rather than 32-bit floats — combined with hand-written SIMD-optimized kernels for x86 (AVX/AVX2/AVX-512) and ARM (NEON, Apple Silicon). The results:

- A 7B parameter model runs on a laptop with 8 GB RAM.
- A 13B model runs acceptably on an M1 MacBook Air.
- A 70B model runs on a consumer GPU with 24 GB VRAM.
- Inference speed often matches or exceeds PyTorch/HuggingFace on CPU.

---

## Building and Installation

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build for CPU (default)
make -j$(nproc)

# Build with CUDA support (NVIDIA GPU)
make LLAMA_CUDA=1 -j$(nproc)

# Build with Metal support (Apple GPU)
make LLAMA_METAL=1 -j$(nproc)

# Build with ROCm support (AMD GPU)
make LLAMA_HIPBLAS=1 -j$(nproc)

# CMake build (recommended for integration into larger projects)
cmake -B build -DLLAMA_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

Python bindings are available via `llama-cpp-python`:

```bash
pip install llama-cpp-python

# With CUDA support
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# With Metal support (macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

---

## The GGUF Format

llama.cpp uses the **GGUF** (GGML Universal Format) file format for storing quantized models. GGUF is a binary format designed for efficient memory-mapped loading, storing model weights, tokenizer data, and metadata in a single self-contained file.

Model filenames encode the quantization level:
- `model-Q4_K_M.gguf` — 4-bit, K-quants, medium quality
- `model-Q5_K_M.gguf` — 5-bit, K-quants, medium quality
- `model-Q8_0.gguf` — 8-bit, best quality (near float16)
- `model-F16.gguf` — 16-bit float, no quantization

Thousands of GGUF models are available on Hugging Face under the `TheBloke` and `bartowski` namespaces, with pre-quantized versions of Llama 3, Mistral, Phi-3, Gemma, Qwen2, and many others.

---

## Quantization Methods

### Legacy Quantization (Q4_0, Q8_0)

Simple block-level uniform quantization. Each block of 32 weights is scaled to $n$-bit integers:

$$w_{\text{quant}} = \text{round}(w / \Delta), \quad \Delta = \max(|w_i|) / (2^{n-1} - 1)$$

### K-Quants (Q4_K, Q5_K, Q6_K)

More sophisticated blocked quantization that uses super-blocks with shared scale factors, achieving better quality than legacy methods at the same bit width. K-quants are the recommended quantization for most use cases.

### iQuants (IQ2_XXS, IQ3_S, etc.)

Importance-weighted quantization that assigns higher precision to weights that are more important for model quality (based on Fisher information or activation statistics). Achieves better quality than K-quants at very low bit widths (2–3 bits).

### Quantization Comparison

| Format | Bits/Weight | File Size (7B) | Quality Loss | Recommended Use |
|--------|------------|----------------|--------------|-----------------|
| F16 | 16 | 13.5 GB | None | GPU with 16 GB+ VRAM |
| Q8_0 | 8 | 7.2 GB | Negligible | High-quality CPU/GPU |
| Q5_K_M | 5 | 4.8 GB | Very small | Balanced quality/size |
| Q4_K_M | 4 | 4.1 GB | Small | Most common choice |
| Q3_K_M | 3 | 3.3 GB | Moderate | Constrained RAM |
| Q2_K | 2 | 2.7 GB | Significant | Minimum RAM use |

---

## Command-Line Inference

### Basic Text Generation

```bash
# Download a model (e.g., from Hugging Face)
# wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

./llama-cli \
    -m models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    -p "Explain the Turing test in simple terms." \
    -n 200 \
    --temp 0.7 \
    --top-p 0.9 \
    --repeat-penalty 1.1
```

### Interactive Chat Mode

```bash
./llama-cli \
    -m models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    -i \
    --chat-template llama3 \
    -sys "You are a helpful AI assistant." \
    -n -1 \
    --ctx-size 8192
```

### GPU Offloading

Use `-ngl` (number of GPU layers) to offload model layers to GPU VRAM:

```bash
# Offload all layers to GPU (requires enough VRAM)
./llama-cli -m model.gguf -ngl 99 -p "Hello"

# Partial offload: first 20 layers on GPU, rest on CPU
./llama-cli -m model.gguf -ngl 20 -p "Hello"
```

---

## Python API

```python
from llama_cpp import Llama

# Load model (auto-detects GPU if available)
llm = Llama(
    model_path="models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,    # offload all layers to GPU; 0 for CPU-only
    n_ctx=8192,         # context window size
    n_batch=512,        # batch size for prompt processing
    verbose=False,
)

# Basic completion
output = llm(
    "Q: What is the capital of France? A:",
    max_tokens=50,
    stop=["Q:", "\n"],
    echo=True,
)
print(output["choices"][0]["text"])

# Chat completions (OpenAI-compatible API)
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is a neural network?"},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response["choices"][0]["message"]["content"])
```

### Streaming

```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf", n_gpu_layers=-1, n_ctx=4096)

for chunk in llm.create_chat_completion(
    messages=[{"role": "user", "content": "Write a haiku about AI."}],
    stream=True,
):
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)
```

---

## OpenAI-Compatible REST Server

llama.cpp ships with a built-in HTTP server that exposes an OpenAI-compatible API, enabling drop-in replacement of cloud API calls with local inference:

```bash
./llama-server \
    -m models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    --ctx-size 8192 \
    --host 0.0.0.0 \
    --port 8080 \
    --n-predict 512
```

Use with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="local-model",  # model name is ignored
    messages=[{"role": "user", "content": "Summarize quantum entanglement."}],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

---

## Supported Hardware Backends

| Backend | Flag | Description |
|---------|------|-------------|
| CPU (default) | — | Optimized BLAS / SIMD kernels (AVX, NEON) |
| CUDA | `LLAMA_CUDA=1` | NVIDIA GPU acceleration |
| Metal | `LLAMA_METAL=1` | Apple GPU (M1/M2/M3) acceleration |
| ROCm/HIP | `LLAMA_HIPBLAS=1` | AMD GPU acceleration |
| OpenCL | `LLAMA_CLBLAST=1` | Generic GPU (legacy) |
| Vulkan | `LLAMA_VULKAN=1` | Cross-platform GPU (experimental) |
| SYCL | `LLAMA_SYCL=1` | Intel GPU acceleration |

Apple Silicon (M-series chips) is particularly well-served: Metal acceleration combined with the unified memory architecture allows offloading all layers to the GPU without PCIe bandwidth bottlenecks, making Apple laptops excellent llama.cpp machines.

---

## Model Conversion to GGUF

To convert a Hugging Face model to GGUF format:

```bash
# Install conversion dependencies
pip install transformers torch sentencepiece

# Convert to F16 GGUF
python convert_hf_to_gguf.py \
    /path/to/hf-model \
    --outfile models/my-model-f16.gguf \
    --outtype f16

# Quantize to Q4_K_M
./llama-quantize models/my-model-f16.gguf models/my-model-Q4_K_M.gguf Q4_K_M
```

---

## Performance Tips

**Use `-ngl 99` to maximize GPU offload** — placing all layers on GPU dramatically reduces inference latency compared to CPU-only or partial offload.

**Match context size to actual needs** — `n_ctx` determines the KV cache size. Setting it higher than needed wastes VRAM. Start with 4096 for chat, 8192 for long-form.

**Enable flash attention** — Add `--flash-attn` to reduce KV cache memory consumption for long contexts, enabling larger context windows within the same VRAM.

**Use batch processing for throughput** — For offline batch inference, set `--parallel` and `--batch-size` to process multiple requests simultaneously.

**mmap and mlock** — llama.cpp memory-maps model files by default (`--mmap`), allowing the OS to manage memory efficiently. For latency-critical applications, use `--mlock` to pin the model in RAM.

---

## Ecosystem

llama.cpp powers many popular local AI applications:
- **Ollama**: User-friendly wrapper around llama.cpp with model management.
- **LM Studio**: Desktop GUI for running and chatting with local models.
- **GPT4All**: Cross-platform desktop application for private, local AI.
- **Jan**: Open-source ChatGPT alternative running locally.
- **AnythingLLM**: Local RAG + agent platform powered by llama.cpp.

---

## Summary

llama.cpp democratized local LLM inference, making it accessible on consumer hardware through aggressive quantization, hand-optimized kernels, and a zero-dependency C/C++ architecture. Its support for GGUF quantized models, OpenAI-compatible server, GPU offloading, and broad hardware backend support make it the most versatile open-source LLM inference engine available. Whether you are building a local AI assistant, embedding LLMs in a native application, deploying on edge devices, or simply exploring models without cloud costs, llama.cpp is the foundation that most of the local AI ecosystem is built upon.
