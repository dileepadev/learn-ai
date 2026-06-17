---
title: Introduction to Together AI
description: An overview of Together AI — a cloud platform for running, fine-tuning, and deploying open-source LLMs with a focus on speed, cost-efficiency, and developer experience.
---

Together AI is a cloud inference and fine-tuning platform purpose-built for open-source large language models. It provides fast, cost-effective API access to a wide catalog of open models — including Llama, Mistral, Qwen, DeepSeek, and others — along with tools for custom fine-tuning and deployment.

## What Together AI Offers

### Inference API
Together AI's core offering is a fast inference API that is OpenAI-compatible. This means you can switch from OpenAI's API to Together AI by changing only the base URL and API key — your existing code continues to work.

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-together-api-key",
    base_url="https://api.together.xyz/v1",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3-70b-chat-hf",
    messages=[{"role": "user", "content": "Explain transformer attention."}],
)
print(response.choices[0].message.content)
```

### Model Catalog
Together hosts dozens of popular open models, including:
- **Meta Llama 3 / 3.1 / 3.2** (8B, 70B, 405B)
- **Mistral and Mixtral** (7B, 8x7B, 8x22B)
- **Qwen 2.5** (7B, 72B)
- **DeepSeek** models
- **Gemma** (Google's open models)
- **Code-specialized models:** CodeLlama, DeepSeek-Coder
- **Embedding models** for vector search

Models are regularly updated as new open releases appear.

### Speed: FlashAttention and Custom Kernels
Together AI is known for low-latency, high-throughput inference. They use custom CUDA kernels, speculative decoding, and FlashAttention optimizations to achieve throughput that is often significantly faster than self-hosted deployments on equivalent hardware.

### Fine-Tuning
Together AI provides a managed fine-tuning service:
- Upload a dataset in JSONL format.
- Specify a base model and hyperparameters.
- Train a LoRA adapter or full fine-tune.
- The resulting model is deployed as a private API endpoint.

```python
import together

# Start a fine-tuning job
response = together.FineTuning.create(
    training_file="file-abc123",
    model="meta-llama/Llama-3-8b-hf",
    n_epochs=3,
    learning_rate=1e-5,
)
```

### Dedicated Endpoints
For production use, Together AI offers dedicated GPU instances for a single tenant. This provides consistent latency guarantees, eliminating the variability of shared inference infrastructure.

## Key Use Cases

- **Prototyping with open models:** Quickly test Llama, Mistral, or other open models without setting up infrastructure.
- **Cost reduction:** Open-source model inference on Together AI is typically 3–10× cheaper per million tokens than equivalent OpenAI or Anthropic endpoints for many use cases.
- **Privacy-sensitive applications:** Open models on Together AI mean your data is not used by a model provider to train future models (unlike some proprietary APIs).
- **Custom fine-tuned models:** Build domain-specific models and deploy them without managing GPU clusters.
- **High-throughput batch workloads:** Together's infrastructure handles large-scale batch inference efficiently.

## Pricing Model

Together AI charges per token (input + output), similar to other inference providers. Prices vary by model size:
- Smaller models (7B–8B) are the most cost-effective.
- Larger models (70B, 405B) cost more but are still substantially cheaper than comparable proprietary models.
- Fine-tuning is billed separately by compute time and token count.

Check the Together AI pricing page for current rates, as pricing changes frequently in this competitive market.

## Comparison to Alternatives

| Platform | Focus | Open Models | Fine-Tuning | Speed |
|----------|-------|-------------|-------------|-------|
| Together AI | Open LLM inference + FT | ✓ | ✓ | Very fast |
| Groq | Ultra-low latency | ✓ | ✗ | Fastest (LPU) |
| Replicate | Open models, diverse | ✓ | ✓ | Moderate |
| Fireworks AI | Fast open inference | ✓ | ✓ | Fast |
| OpenAI | Proprietary GPT models | ✗ | ✓ | Fast |
| Ollama | Local/self-hosted | ✓ | ✓ | Hardware-dependent |

Together AI occupies a sweet spot: broad model support, fine-tuning capability, and competitive speed and pricing for production-scale workloads.

## Getting Started

1. Create an account at [together.ai](https://together.ai).
2. Generate an API key from the dashboard.
3. Install the SDK: `pip install together`
4. Browse the model playground to test models interactively before writing code.
5. Start with the OpenAI-compatible API for the easiest integration path.

The Python SDK also supports direct Together AI features (fine-tuning management, file uploads, model listing) beyond what the OpenAI compatibility layer covers.
