---
title: Introduction to Modal
description: Learn how Modal provides a Python-native cloud computing platform for running ML workloads — offering serverless GPU access, containerized functions, persistent volumes, and web endpoints with no infrastructure management required.
---

**Modal** is a cloud computing platform designed specifically for Python developers running compute-intensive workloads — model training, batch inference, data processing pipelines, and web endpoints — with a programming model that eliminates traditional infrastructure management. Where other cloud platforms require configuring VMs, containers, Kubernetes clusters, or serverless functions through separate tooling, Modal lets you define your compute environment and deploy it using pure Python decorators, with GPU instances provisioning in seconds.

Modal's key design philosophy is **infrastructure as code in the application layer**: the cloud environment is described directly in Python alongside the application code — no Dockerfiles (Modal builds them for you), no YAML manifests, no cluster configuration.

## Why Modal for ML Workloads

Running ML workloads in the cloud traditionally involves:

- Writing Dockerfiles to containerize dependencies.
- Provisioning and managing GPU instances (EC2, GCE, Azure VMs).
- Setting up job schedulers (SLURM, Kubernetes) for batch processing.
- Managing cloud storage for data and model artifacts.
- Handling startup latency — GPU VMs take 3-5 minutes to boot.

Modal addresses all of these:

- **Auto-containerization**: Define Python packages and system dependencies in code; Modal builds the container.
- **On-demand GPUs**: GPU containers start in ~5 seconds — pay only for actual compute time, not idle time.
- **Serverless scaling**: Functions scale to zero when not running; scale to hundreds of parallel instances on demand.
- **Persistent volumes**: Built-in distributed file storage for model weights, datasets, and outputs.

## Installation and Setup

```bash
pip install modal
modal setup  # Authenticate with your Modal account
```

## Core Concepts

### Apps and Functions

A **Modal App** is a container for related functions. Functions decorated with `@app.function()` run in the cloud:

```python
import modal

app = modal.App("my-ml-app")

@app.function()
def hello():
    return "Hello from the cloud!"

# Run locally (executes in Modal's cloud)
with app.run():
    result = hello.remote()
    print(result)
```

### Defining the Container Image

The execution environment is defined in Python — no Dockerfile required:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.41.0",
        "accelerate",
        "sentencepiece",
    )
    .apt_install("git")
    .run_commands("huggingface-cli download meta-llama/Meta-Llama-3-8B")
)

@app.function(image=image, gpu="A100", timeout=3600)
def run_inference(prompt: str) -> str:
    from transformers import pipeline
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")
    return pipe(prompt, max_new_tokens=200)[0]["generated_text"]
```

### GPU Selection

Modal supports a range of GPU types specified per function:

```python
# Single GPU
@app.function(gpu="T4")         # NVIDIA T4 — cost-effective for inference
@app.function(gpu="A10G")       # A10G — good for mid-range fine-tuning
@app.function(gpu="A100")       # A100 80GB — large model training
@app.function(gpu="H100")       # H100 — highest performance

# Multiple GPUs
@app.function(gpu=modal.gpu.A100(count=4))  # 4× A100 for multi-GPU training
```

## Batch Inference at Scale

Modal excels at embarrassingly parallel batch inference — running the same model on thousands of inputs in parallel:

```python
import modal
from typing import Iterator

app = modal.App("batch-inference")

image = modal.Image.debian_slim().pip_install("transformers", "torch", "sentencepiece")

@app.function(image=image, gpu="A10G", concurrency_limit=20)
def classify_text(text: str) -> dict:
    """Classify a single text — runs in parallel across many GPU instances."""
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(text, candidate_labels=["positive", "negative", "neutral"])
    return {"text": text, "label": result["labels"][0], "score": result["scores"][0]}

@app.local_entrypoint()
def main():
    texts = load_dataset_texts()  # Load your texts locally
    
    # .map() distributes work across up to 20 parallel GPU containers
    results = list(classify_text.map(texts, order_outputs=True))
    save_results(results)
```

The `.map()` call automatically spawns parallel containers, distributes inputs, and collects results — with automatic retry on failure.

## Model Fine-Tuning on Modal

Running a fine-tuning job is straightforward:

```python
import modal

app = modal.App("fine-tune-llm")

# Persistent volume to store model weights between runs
volume = modal.Volume.from_name("model-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("axolotl[flash-attn,deepspeed]", "torch", "transformers")
)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=2),
    volumes={"/model-weights": volume},
    timeout=86400,  # 24 hours max
    memory=131072,  # 128GB RAM
)
def fine_tune():
    import subprocess
    # Run axolotl fine-tuning with config stored in the volume
    subprocess.run([
        "accelerate", "launch", "-m", "axolotl.cli.train",
        "/model-weights/configs/qlora_config.yaml"
    ], check=True)
    # Fine-tuned weights are saved to /model-weights/ (persisted in the volume)

@app.local_entrypoint()
def main():
    fine_tune.remote()  # Kicks off the job in the cloud
```

## Web Endpoints

Modal functions can be deployed as persistent HTTP endpoints using `@app.cls` and `modal.web_endpoint`:

```python
from modal import web_endpoint

@app.cls(image=image, gpu="A10G", min_containers=1)
class InferenceAPI:
    
    @modal.enter()
    def load_model(self):
        """Load model once when the container starts."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    @web_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        prompt = request["prompt"]
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": text}
```

Deploy with `modal deploy app.py` — Modal provisions the endpoint at a stable URL and keeps the container warm (`min_containers=1`) to eliminate cold-start latency.

## Scheduled Jobs and Cron Tasks

```python
@app.function(schedule=modal.Cron("0 2 * * *"))  # Every day at 2am
def nightly_retraining():
    """Retrain model nightly on new data."""
    download_new_training_data()
    fine_tune_model()
    evaluate_and_push_if_improved()
```

## Persistent Volumes and Secrets

```python
# Secrets: securely inject API keys and credentials
@app.function(
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("openai-api-key"),
    ]
)
def use_external_apis():
    import os
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    openai_key = os.environ["OPENAI_API_KEY"]
    ...

# Network file system for large shared datasets
nfs = modal.NetworkFileSystem.from_name("shared-datasets", create_if_missing=True)

@app.function(network_file_systems={"/datasets": nfs})
def train_on_shared_data():
    # /datasets is accessible across all container instances
    data = load_data("/datasets/my_large_dataset.parquet")
```

## Comparison with Alternatives

| Feature | Modal | AWS Lambda | Replicate | RunPod |
| --- | --- | --- | --- | --- |
| GPU support | Yes (T4–H100) | No | Yes | Yes |
| Python-native config | Yes | No (YAML/CLI) | No | No |
| Cold start time | ~5 seconds | <1s (CPU) / N/A | ~30-60s | Manual |
| Parallel map | Built-in | Manual (SQS+Lambda) | No | No |
| Persistent volumes | Yes | S3 only | No | Yes |
| Free tier | Limited | Yes | No | No |
| Best for | ML workloads | Web APIs | Model serving | Long training |

Modal is particularly well-suited for data scientists and ML engineers who want cloud GPU access with minimal DevOps overhead — the entire workflow from local development to cloud deployment uses the same Python code.
