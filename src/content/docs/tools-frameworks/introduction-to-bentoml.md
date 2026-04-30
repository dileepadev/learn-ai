---
title: Introduction to BentoML
description: Get started with BentoML — the unified model serving framework for packaging, deploying, and scaling machine learning models and LLM pipelines as production-ready REST APIs and gRPC services, with built-in support for batching, multi-model composition, and cloud deployment.
---

**BentoML** is an open-source Python framework for serving machine learning models in production. It bridges the gap between model training and production deployment: a data scientist can package any model — scikit-learn, PyTorch, TensorFlow, Hugging Face, vLLM, or a custom pipeline — into a standardized **Bento** artifact that is portable, versioned, containerized, and deployable anywhere.

Where frameworks like vLLM or Triton Inference Server are specialized (LLM inference and NVIDIA GPU serving respectively), BentoML is a general-purpose serving layer that composes any combination of models, preprocessing steps, business logic, and external API calls into a unified service.

## Core Concepts

**Bento**: The core packaging unit — an immutable, versioned bundle containing model artifacts, source code, dependencies (`requirements.txt` or `pyproject.toml`), and service definition. A Bento is analogous to a Docker image but ML-native.

**Service**: A Python class decorated with `@bentoml.service` that defines the API — input/output schemas, hardware configuration (CPU/GPU/memory), batching behavior, and scaling strategy.

**Runner** (v1) / **Dependency** (v2): An isolated inference unit that wraps a model and can be scaled independently of the service.

**BentoCloud**: BentoML's managed deployment platform (similar to Modal or Replicate) — optional, the framework works equally well deploying to any Kubernetes cluster or cloud VM.

## Installation and Setup

```bash
pip install bentoml

# Optional: GPU support
pip install bentoml[all]

# Verify installation
bentoml --version
```

## Quickstart: Serving a Scikit-Learn Model

```python
# train_and_save.py
import bentoml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Train model
X, y = load_iris(return_X_y=True)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X, y)

# Save to BentoML model store — versioned, content-addressed
saved_model = bentoml.sklearn.save_model(
    "iris_classifier",
    pipeline,
    signatures={"predict": {"batchable": True}},
    metadata={"accuracy": 0.97, "dataset": "iris", "framework": "sklearn"}
)
print(f"Model saved: {saved_model.tag}")
# Output: iris_classifier:j2xldgu5kwjkuaav
```

```python
# service.py
import bentoml
import numpy as np
from pydantic import BaseModel

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisPrediction(BaseModel):
    species: str
    confidence: float

SPECIES_NAMES = ["setosa", "versicolor", "virginica"]

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10}
)
class IrisClassifier:
    # BentoML automatically loads the latest version of this model
    model = bentoml.models.BentoModel("iris_classifier:latest")
    
    def __init__(self):
        import joblib
        self.clf = self.model.load_model()
    
    @bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=50)
    def predict(self, features: list[IrisFeatures]) -> list[IrisPrediction]:
        # Convert to numpy array for sklearn
        X = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
                      for f in features])
        
        proba = self.clf.predict_proba(X)
        class_ids = proba.argmax(axis=1)
        
        return [
            IrisPrediction(
                species=SPECIES_NAMES[idx],
                confidence=float(proba[i, idx])
            )
            for i, idx in enumerate(class_ids)
        ]
```

Serve locally:

```bash
bentoml serve service:IrisClassifier --reload
# Server running at http://localhost:3000
# Swagger UI at http://localhost:3000/docs
```

Test:

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '[{"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 5.1, "petal_width": 1.8}]'
# [{"species": "virginica", "confidence": 0.94}]
```

## Multi-Model Pipelines

BentoML's dependency injection composes multiple models into a single service while allowing each to scale independently:

```python
import bentoml
from pydantic import BaseModel

# Multi-stage NLP pipeline:
# 1. Classify document language
# 2. Route to language-specific summarizer
# 3. Extract entities from the summary

@bentoml.service(resources={"cpu": "1"})
class LanguageClassifier:
    model = bentoml.models.BentoModel("language_classifier:latest")
    
    def __init__(self):
        from transformers import pipeline
        self.classifier = pipeline("text-classification", 
                                   model=self.model.path)
    
    @bentoml.api
    def classify(self, text: str) -> str:
        result = self.classifier(text[:512])[0]
        return result["label"]


@bentoml.service(resources={"cpu": "2", "memory": "4Gi"})
class Summarizer:
    model = bentoml.models.BentoModel("multilingual_summarizer:latest")
    
    def __init__(self):
        from transformers import pipeline
        self.summarizer = pipeline("summarization", model=self.model.path)
    
    @bentoml.api(batchable=True, max_batch_size=8)
    def summarize(self, texts: list[str]) -> list[str]:
        results = self.summarizer(texts, max_length=150, min_length=30)
        return [r["summary_text"] for r in results]


@bentoml.service(
    resources={"cpu": "4"},
    # Each dependency can be scaled independently
)
class DocumentPipeline:
    # Inject dependent services — BentoML handles deployment and scaling
    lang_classifier = bentoml.depends(LanguageClassifier)
    summarizer = bentoml.depends(Summarizer)
    
    @bentoml.api
    async def process(self, document: str) -> dict:
        # Classify language
        language = await self.lang_classifier.to_async.classify(document)
        
        # Summarize
        summary = await self.summarizer.to_async.summarize([document])
        
        return {
            "language": language,
            "summary": summary[0],
            "original_length": len(document),
            "summary_length": len(summary[0])
        }
```

## LLM Serving with BentoML

BentoML integrates with vLLM, llama.cpp, and Hugging Face TGI for high-performance LLM serving:

```python
import bentoml
from annotated_types import Annotated, Ge, Le
from typing import AsyncGenerator

MAX_TOKENS = 2048

@bentoml.service(
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
        "memory": "80Gi"
    },
    traffic={"timeout": 300}
)
class LLMService:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    def __init__(self) -> None:
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            max_model_len=MAX_TOKENS,
            dtype="bfloat16",
            tensor_parallel_size=1
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    @bentoml.api
    async def generate(
        self,
        prompt: str,
        max_tokens: Annotated[int, Ge(1), Le(MAX_TOKENS)] = 512,
        temperature: Annotated[float, Ge(0.0), Le(2.0)] = 0.7,
        stream: bool = False
    ) -> AsyncGenerator[str, None]:
        """Generate text with optional streaming."""
        from vllm import SamplingParams
        import uuid
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )
        request_id = str(uuid.uuid4())
        
        stream_result = self.engine.generate(prompt, sampling_params, request_id)
        
        cursor = 0
        async for request_output in stream_result:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
```

## Building and Containerizing

```bash
# Build a Bento (packages code + models + dependencies)
bentoml build

# Output:
# Successfully built Bento(tag="document_pipeline:abc123xyz")

# Containerize as a Docker image
bentoml containerize document_pipeline:abc123xyz

# Output:
# Successfully built Docker image "document_pipeline:abc123xyz"

# Run the container
docker run -p 3000:3000 document_pipeline:abc123xyz serve

# Push to a registry for deployment
docker tag document_pipeline:abc123xyz your-registry/document_pipeline:abc123xyz
docker push your-registry/document_pipeline:abc123xyz
```

The `bentofile.yaml` defines the build configuration:

```yaml
service: "service:DocumentPipeline"
labels:
  owner: ml-team
  project: document-processing
include:
  - "*.py"
  - "config/"
python:
  packages:
    - transformers>=4.40.0
    - torch>=2.3.0
    - sentencepiece
docker:
  base_image: "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
  env:
    - TOKENIZERS_PARALLELISM=false
```

## Adaptive Batching

BentoML's adaptive batching collects individual requests arriving within a time window and processes them as a batch — maximizing GPU utilization without adding fixed latency:

```python
@bentoml.service
class EmbeddingService:
    model = bentoml.models.BentoModel("bge-large-en:latest")
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(self.model.path)
    
    @bentoml.api(
        batchable=True,
        max_batch_size=128,      # Never exceed 128 items per batch
        max_latency_ms=25        # Wait at most 25ms to fill the batch
    )
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        BentoML dynamically batches concurrent requests:
        - If 10 requests arrive within 25ms, they're batched together
        - If 200 items arrive, processed as 2 batches of 128 and 72
        """
        return self.encoder.encode(texts, normalize_embeddings=True)
```

## BentoML vs. Alternatives

| Feature | BentoML | Triton Inference Server | vLLM | FastAPI |
|---|---|---|---|---|
| **Model packaging** | Native (Bento) | Manual | Manual | Manual |
| **LLM serving** | Via vLLM/TGI | Via TensorRT-LLM | Native | Manual |
| **Multi-model pipelines** | First-class | Limited | No | Manual |
| **Adaptive batching** | Built-in | Built-in | Built-in | Manual |
| **NVIDIA GPU focus** | Agnostic | Required | Recommended | None |
| **Cloud deployment** | BentoCloud / K8s | K8s | K8s | K8s |
| **Learning curve** | Low | High | Low | Low |

BentoML is the right choice when you need to serve **any ML model type** (not just LLMs), compose multi-step pipelines, and maintain model versioning — all without writing custom Docker and Kubernetes boilerplate. Its convention-over-configuration approach makes it the fastest path from a trained model to a production API endpoint.
