---
title: Introduction to Ollama
description: Learn how to run large language models locally with Ollama — covering installation, model management, the REST API, multimodal models, and integration with popular AI frameworks.
---

**Ollama** is an open-source tool that makes running large language models locally as simple as pulling a Docker image. It handles model downloading, quantization, resource management, and serving — exposing a clean REST API and CLI that works with most LLM frameworks.

## Why Run LLMs Locally?

Running models locally with Ollama offers several advantages:

- **Privacy** — Data never leaves your machine; no third-party API sees your prompts or responses.
- **No internet dependency** — Works fully offline once models are downloaded.
- **No API costs** — Run unlimited requests without per-token billing.
- **Low latency** — No network round-trips; especially fast on Apple Silicon with the Metal GPU backend.
- **Customization** — Modify system prompts, parameters, and fine-tuned model files freely.

## Installation

Ollama supports macOS, Linux, and Windows:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download installer from https://ollama.com/download
```

On macOS, Ollama runs as a background menu bar application and uses the Metal GPU backend for Apple Silicon acceleration. On Linux, it runs as a systemd service and supports NVIDIA and AMD GPUs via CUDA/ROCm.

## Running Your First Model

```bash
# Pull and run a model interactively (downloads on first run)
ollama run llama3.2

# Run with a single prompt (non-interactive)
ollama run llama3.2 "Explain backpropagation in simple terms"

# List downloaded models
ollama list

# Pull a model without running it
ollama pull mistral

# Remove a model
ollama rm mistral
```

On first run, Ollama downloads the model file (GGUF format, quantized) to `~/.ollama/models`. Subsequent runs reuse the cached model.

## Model Library

Ollama's model library covers the major open-source families:

| Model | Size | Use Case |
|---|---|---|
| `llama3.2` | 3B, 11B | General purpose, fast |
| `llama3.1` | 8B, 70B, 405B | General purpose |
| `mistral` | 7B | Fast, multilingual |
| `gemma3` | 1B, 4B, 12B, 27B | Google's efficient models |
| `qwen2.5` | 0.5B–72B | Strong multilingual + code |
| `deepseek-r1` | 1.5B–671B | Reasoning, math |
| `phi4` | 14B | Small but capable (Microsoft) |
| `codellama` | 7B–70B | Code generation |
| `nomic-embed-text` | 137M | Text embeddings |
| `llava` | 7B, 13B | Vision-language (multimodal) |

Models are downloaded in **GGUF format** with Q4, Q5, Q6, or Q8 quantization levels — balancing size against quality.

## The REST API

Ollama exposes a local REST API on `http://localhost:11434` that is **compatible with the OpenAI API format**:

### Generate (Completion)

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What is quantum entanglement?",
  "stream": false
}'
```

### Chat (Conversational)

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain gradient descent."}
  ],
  "stream": false
}'
```

### Embeddings

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "nomic-embed-text",
  "input": "Machine learning is a subset of AI."
}'
```

## OpenAI-Compatible Endpoint

Ollama's `/v1/` endpoint is compatible with the OpenAI client library — allowing you to swap cloud models for local ones with a single line change:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required but unused
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Write a haiku about neural networks."}]
)
print(response.choices[0].message.content)
```

## Custom Models with Modelfile

A **Modelfile** lets you create custom model variants with different system prompts, parameters, or base models — similar to a Dockerfile:

```dockerfile
FROM llama3.2

SYSTEM """
You are an expert Python developer. Always provide runnable code examples.
Prefer concise, idiomatic solutions following PEP 8.
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
```

```bash
# Build and run the custom model
ollama create python-expert -f Modelfile
ollama run python-expert "How do I use dataclasses in Python?"
```

You can also base a Modelfile on a local GGUF file to run fine-tuned models:

```dockerfile
FROM ./my-fine-tuned-model.gguf
```

## Multimodal Models

Ollama supports vision models that accept image inputs:

```python
import ollama

response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        'content': 'Describe what you see in this image.',
        'images': ['./photo.jpg']
    }]
)
print(response['message']['content'])
```

## Integration with LLM Frameworks

Ollama's API compatibility makes it a drop-in local backend for most frameworks:

### LangChain

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)
response = llm.invoke("What is the Transformer architecture?")
```

### LlamaIndex

```python
from llama_index.llms.ollama import Ollama

llm = Ollama(model="mistral", request_timeout=120.0)
```

### DSPy

```python
import dspy
lm = dspy.LM("ollama/llama3.2", api_base="http://localhost:11434")
dspy.configure(lm=lm)
```

## Performance Tuning

- **GPU acceleration** — Ollama auto-detects NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (Metal). Ensure GPU drivers are installed.
- **Context length** — `num_ctx` controls how many tokens the model considers. Higher = more memory. Default is model-dependent.
- **Concurrency** — Set `OLLAMA_NUM_PARALLEL` to serve multiple requests simultaneously (requires more VRAM).
- **Model offloading** — For models larger than VRAM, layers are offloaded to CPU RAM at a cost to speed.
- **Quantization choice** — `q4_K_M` is a good default (4-bit, fast, good quality). `q8_0` is near-lossless but requires more memory.

## Practical Use Cases

- **Local Copilot** — Power Continue.dev or GitHub Copilot alternatives in VS Code with a local model.
- **Private document Q&A** — Build a local RAG pipeline over sensitive docs using Ollama + LlamaIndex + a local vector store.
- **Offline development and testing** — Prototype LLM features without internet access or API costs.
- **Edge deployment** — Run Ollama on a local server in air-gapped enterprise or industrial environments.

Ollama has become the de facto standard for local LLM inference — abstracting away the complexity of GGUF files, quantization choices, and inference engines (llama.cpp under the hood) into a simple, developer-friendly interface.
