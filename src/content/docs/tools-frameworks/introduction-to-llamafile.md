---
title: Introduction to Llamafile
description: A practical guide to Mozilla's llamafile — a distribution format that packages large language models and a runtime into a single executable file, enabling one-click LLM deployment on any operating system.
---

# Introduction to Llamafile

**Llamafile** is an open-source project by Mozilla that lets you distribute entire large language models as a **single executable file** — no installation, no dependency management, no Python environment. A llamafile bundles the model weights (in GGUF format) and a compact C/C++ runtime (built on llama.cpp and Cosmopolitan libc) into one portable binary that runs on Linux, macOS, and Windows without modification.

## Why Llamafile?

Deploying LLMs involves installing Python, creating virtual environments, downloading weights separately, and managing CUDA drivers. Llamafile collapses this to a single download:

```bash
# Traditional deployment (many steps)
pip install transformers accelerate
python -c "from transformers import pipeline; p = pipeline('text-generation', model='...')"

# Llamafile (one step)
./mistral-7b-instruct-v0.2.Q4_K_M.llamafile
```

The result is a local web server with a chat UI and an OpenAI-compatible API endpoint.

## How It Works

Llamafile uses **Cosmopolitan libc** — a C library that produces executables compatible with multiple operating systems from a single binary. The `Actually Portable Executable` (APE) format embeds a ZIP archive containing:

- A platform-native launcher stub (fat binary: x86-64 + ARM64)
- The GGUF model weights
- Runtime assets (HTML chat UI, server logic)

On first run, the launcher extracts and caches runtime files, then starts the inference server.

## Getting Started

### Downloading a Llamafile

```bash
# Download a llamafile (example: Mistral 7B Instruct Q4_K_M)
wget https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.llamafile

# Make executable
chmod +x mistral-7b-instruct-v0.2.Q4_K_M.llamafile

# Run — opens chat UI at http://localhost:8080
./mistral-7b-instruct-v0.2.Q4_K_M.llamafile
```

On Windows, rename the file to add `.exe` and double-click.

### Command-Line Inference

```bash
# Single prompt, no server
./mistral-7b-instruct-v0.2.Q4_K_M.llamafile \
  --cli \
  -p "Write a haiku about neural networks." \
  -n 200

# Pipe input
echo "Explain backpropagation in one paragraph." | \
  ./mistral-7b-instruct-v0.2.Q4_K_M.llamafile --cli -p -
```

### Server Mode with Options

```bash
./mistral-7b-instruct-v0.2.Q4_K_M.llamafile \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 8192 \
  --threads 8 \
  --gpu-layers 35        # offload 35 layers to GPU (CUDA/Metal)
```

## OpenAI-Compatible API

The llamafile server exposes an OpenAI-compatible REST API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="no-key-required",
)

response = client.chat.completions.create(
    model="mistral-7b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=256,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

Any tool that speaks the OpenAI API — LangChain, LlamaIndex, Instructor, etc. — works with a llamafile server out of the box by pointing to `http://localhost:8080/v1`.

## Integrating with LangChain

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="llamafile",
    model="local-model",
)

response = llm.invoke([HumanMessage(content="Summarize quantum entanglement.")])
print(response.content)
```

## Building a Custom Llamafile

You can package any GGUF model into a llamafile:

```bash
# Download zipalign tool from llamafile release
wget https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.13/llamafile-0.8.13.zip
unzip llamafile-0.8.13.zip

# Start from the base llamafile server binary
cp llamafile-0.8.13/bin/llamafile my-model.llamafile

# Embed your GGUF model weights into the binary
./llamafile-0.8.13/bin/zipalign -j0 \
  my-model.llamafile \
  ./my-model.Q4_K_M.gguf \
  .args

# Create .args file to set defaults
echo "-m
my-model.Q4_K_M.gguf
--host
0.0.0.0
..." > .args

chmod +x my-model.llamafile
./my-model.llamafile
```

## GPU Acceleration

Llamafile supports GPU inference on:

- **NVIDIA CUDA** (auto-detected on Linux/Windows)
- **Apple Metal** (auto-detected on macOS)
- **AMD ROCm** (via llama.cpp ROCm backend)
- **CPU fallback**: always available, uses AVX2/AVX512 SIMD

```bash
# Use GPU for all layers
./mistral-7b-instruct-v0.2.Q4_K_M.llamafile --gpu-layers 999

# Check GPU utilization
./mistral-7b-instruct-v0.2.Q4_K_M.llamafile --gpu-layers 999 --verbose 2>&1 | grep "offloaded"
```

## Available Models (Mozilla HuggingFace)

| Model | Size | Quantization | File Size |
|---|---|---|---|
| Mistral 7B Instruct v0.2 | 7B | Q4_K_M | 4.4 GB |
| Llama 3.1 8B Instruct | 8B | Q4_K_M | 4.9 GB |
| Phi-3 Mini Instruct | 3.8B | Q4_K_M | 2.4 GB |
| Gemma 2 9B Instruct | 9B | Q4_K_M | 5.8 GB |
| WizardCoder Python 13B | 13B | Q4_K_M | 8.0 GB |

## Embedding Support

Llamafile also supports embedding generation:

```python
import requests

response = requests.post(
    "http://localhost:8080/v1/embeddings",
    json={
        "model": "local-model",
        "input": "The quick brown fox jumps over the lazy dog.",
    },
)
embedding = response.json()["data"][0]["embedding"]
print(f"Embedding dimension: {len(embedding)}")
```

## Llamafile vs Alternatives

| Tool | Single Binary | API Server | GPU Support | GGUF Models | Chat UI |
|---|---|---|---|---|---|
| Llamafile | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ollama | ❌ (installer) | ✅ | ✅ | ✅ | ❌ |
| llama.cpp | ❌ (compile) | ✅ | ✅ | ✅ | ❌ |
| LM Studio | ❌ (GUI app) | ✅ | ✅ | ✅ | ✅ |
| Jan.ai | ❌ (GUI app) | ✅ | ✅ | ✅ | ✅ |

## Use Cases

- **Air-gapped environments**: ship a single file to machines without internet access or package managers
- **Developer tools**: embed a local LLM in a CLI tool distributed as a single binary
- **Education**: students run LLMs in one step without environment setup
- **Enterprise compliance**: run fully offline with no data leaving the machine
- **Rapid prototyping**: test different models without environment conflicts

## Summary

Llamafile solves a real distribution problem: getting a large language model running on an arbitrary machine with zero setup. By combining Cosmopolitan libc's cross-platform binary format with llama.cpp's efficient inference engine and GGUF quantization, Mozilla has created the most frictionless path to local LLM deployment. For developers shipping AI-powered CLI tools, educators introducing students to LLMs, or enterprises requiring air-gapped operation, llamafile provides a unique and compelling packaging model that no other LLM deployment tool currently matches.
