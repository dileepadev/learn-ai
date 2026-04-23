---
title: Introduction to NVIDIA Triton Inference Server
description: Learn how NVIDIA Triton Inference Server enables scalable, high-performance model serving across multiple frameworks and hardware backends — with dynamic batching, ensemble pipelines, model versioning, and performance analysis tools.
---

**NVIDIA Triton Inference Server** is an open-source inference serving software that standardizes AI model deployment across diverse hardware (NVIDIA GPUs, x86 and ARM CPUs) and frameworks (TensorRT, ONNX Runtime, PyTorch, TensorFlow, OpenVINO, Python). Triton provides a production-grade serving infrastructure with dynamic batching, concurrent model execution, model versioning, and extensive performance monitoring — enabling organizations to serve multiple models efficiently within a single server.

Unlike framework-specific servers (TorchServe for PyTorch, TensorFlow Serving for TensorFlow), Triton is framework-agnostic — a single server can simultaneously serve a TensorRT-optimized image classifier, an ONNX Runtime text encoder, and a PyTorch sequence model, all with a unified API and shared GPU memory management.

## Core Architecture

### Model Repository

Triton's **model repository** is a file system directory (local, NFS, or object storage) containing all models to be served. Each model is a subdirectory with a required layout:

```
model_repository/
├── text_classifier/
│   ├── config.pbtxt          # Model configuration
│   ├── 1/                    # Version 1
│   │   └── model.onnx        # Model file
│   └── 2/                    # Version 2
│       └── model.onnx
├── image_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan        # TensorRT engine
└── llm_decoder/
    ├── config.pbtxt
    └── 1/
        └── model.py          # Python backend
```

Triton can be launched pointing at this repository:

```bash
tritonserver --model-repository=/models --http-port=8000 --grpc-port=8001 --metrics-port=8002
```

### Model Configuration

Each model's `config.pbtxt` declares its inputs, outputs, backend, batching strategy, and resource allocation:

```protobuf
name: "text_classifier"
backend: "onnxruntime"
max_batch_size: 64

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 128 ]         # sequence length
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ 128 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 10 ]          # 10 classes
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

## Supported Backends

Triton supports a rich set of backends, each optimized for different frameworks and use cases:

### TensorRT Backend

**TensorRT** (NVIDIA's high-performance inference optimizer) is the highest-performance backend for NVIDIA GPUs:

- TensorRT engines are compiled from ONNX models and optimized for specific GPU architectures using layer fusion, precision calibration (FP16/INT8), and kernel auto-tuning.
- Engines are hardware-specific — a TensorRT engine compiled for an A100 will not run on a T4.
- Triton's `tensorrt` backend loads `.plan` files (serialized TensorRT engines).

```bash
# Convert ONNX model to TensorRT engine
trtexec --onnx=model.onnx \
        --saveEngine=model.plan \
        --fp16 \
        --minShapes=input:1x128 \
        --optShapes=input:32x128 \
        --maxShapes=input:64x128
```

### ONNX Runtime Backend

The **ONNX Runtime** backend supports any ONNX-format model, executing on GPU (via CUDA Execution Provider) or CPU:

- Broader hardware compatibility than TensorRT.
- Supports dynamic shapes without engine recompilation.
- Works across GPU generations without model changes.

### PyTorch (LibTorch) Backend

The **pytorch** backend loads TorchScript models:

```python
import torch

model = MyModel().eval()
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

### Python Backend

The **Python backend** enables arbitrary Python code as a Triton model — useful for:

- Preprocessing and postprocessing logic.
- Models in frameworks without native Triton backends.
- Ensemble orchestration logic.
- LLM serving via `transformers` or `vllm`.

```python
import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import pipeline

class TritonPythonModel:
    def initialize(self, args):
        self.pipe = pipeline("sentiment-analysis",
                             model="distilbert-base-uncased-finetuned-sst-2-english",
                             device=0)

    def execute(self, requests):
        responses = []
        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text_list = [t.decode() for t in texts.as_numpy().flatten()]

            results = self.pipe(text_list)
            scores = np.array([[r["score"]] for r in results], dtype=np.float32)

            output_tensor = pb_utils.Tensor("SCORE", scores)
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses

    def finalize(self):
        pass
```

## Dynamic Batching

**Dynamic batching** is one of Triton's most impactful throughput features. Rather than processing each request individually, Triton accumulates requests in a queue and groups them into batches before sending to the model:

- **Preferred batch sizes**: Triton prefers batches of specified sizes (e.g., 8, 16, 32) which align with GPU compute efficiency.
- **Max queue delay**: Triton waits up to a specified microsecond delay to accumulate a larger batch — trading a small latency increase for significant throughput gains.
- **Sequence batching**: For sequence models (RNNs, stateful models), Triton maintains per-sequence state across requests, batching different sequences together.

Dynamic batching is transparent to clients — each client sends a single-sample request and receives a single-sample response; batching happens entirely within the server.

## Concurrent Model Execution

Triton can run **multiple instances** of the same model simultaneously on the same GPU or across multiple GPUs:

```protobuf
instance_group [
  {
    count: 4        # 4 model instances on GPU 0
    kind: KIND_GPU
    gpus: [ 0 ]
  },
  {
    count: 2        # 2 model instances on GPU 1
    kind: KIND_GPU
    gpus: [ 1 ]
  }
]
```

Multiple instances allow Triton to keep the GPU fully utilized even when individual model instances are waiting for memory transfers or compute — maximizing throughput for low-latency models that don't fully utilize the GPU alone.

## Ensemble Models

**Ensemble models** (also called **pipeline models**) chain multiple models together, with the output of one model feeding as input to the next — all within a single Triton server:

```protobuf
name: "text_classification_pipeline"
platform: "ensemble"
max_batch_size: 32

input [
  { name: "RAW_TEXT", data_type: TYPE_STRING, dims: [1] }
]

output [
  { name: "CLASS_PROBS", data_type: TYPE_FP32, dims: [10] }
]

ensemble_scheduling {
  step [
    {
      model_name: "tokenizer"
      model_version: 1
      input_map { key: "TEXT" value: "RAW_TEXT" }
      output_map { key: "INPUT_IDS" value: "input_ids" }
      output_map { key: "ATTENTION_MASK" value: "attention_mask" }
    },
    {
      model_name: "text_classifier"
      model_version: 1
      input_map { key: "input_ids" value: "input_ids" }
      input_map { key: "attention_mask" value: "attention_mask" }
      output_map { key: "logits" value: "CLASS_PROBS" }
    }
  ]
}
```

The ensemble is atomic from the client's perspective — a single HTTP/gRPC call routes through the full pipeline. Triton schedules the steps efficiently, passing tensors in GPU memory between steps where possible to avoid unnecessary CPU transfers.

## Client Libraries and API

Triton exposes **HTTP/REST** and **gRPC** endpoints. The official client libraries (`tritonclient`) provide Python, C++, Java, and Go bindings:

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare inputs
input_ids = np.array([[101, 7592, 1010, 2088, 102, 0, 0, 0]], dtype=np.int64)
attention_mask = np.array([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=np.int64)

inputs = [
    httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
    httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
]
inputs[0].set_data_from_numpy(input_ids)
inputs[1].set_data_from_numpy(attention_mask)

outputs = [httpclient.InferRequestedOutput("logits")]

# Send inference request
response = client.infer(
    model_name="text_classifier",
    inputs=inputs,
    outputs=outputs
)

logits = response.as_numpy("logits")
print(f"Predicted class: {logits.argmax()}")
```

## Model Versioning and Management

Triton supports **model versioning** — multiple numbered versions of each model can coexist in the repository:

- **Version policy**: Configure which versions are loaded (all, latest N, or specific versions).
- **Hot model swap**: New model versions can be loaded without restarting the server — Triton detects repository changes and loads/unloads versions dynamically.
- **A/B testing**: Serve multiple versions simultaneously and route traffic between them (via client-side version specification or an upstream load balancer).

## Performance Analysis with Perf Analyzer

The **Perf Analyzer** tool (`perf_analyzer`) measures Triton model throughput and latency under controlled load:

```bash
# Measure throughput and latency for text_classifier
perf_analyzer \
  -m text_classifier \
  -u localhost:8000 \
  --protocol http \
  --concurrency-range 1:32:2 \
  --measurement-interval 5000 \
  --shape input_ids:1,128 \
  --shape attention_mask:1,128
```

Perf Analyzer sweeps concurrency levels and reports:

- **Throughput** (requests/second) at each concurrency level.
- **Latency percentiles** (p50, p90, p99) at each concurrency level.
- The **saturation point** — the concurrency level where adding more clients no longer increases throughput.

These measurements inform deployment decisions: instance count, dynamic batching settings, and whether TensorRT optimization is worth the compilation overhead.

## Triton vs. Other Serving Solutions

| Feature | Triton | TorchServe | TF Serving | vLLM |
| --- | --- | --- | --- | --- |
| Multi-framework | Yes | PyTorch only | TensorFlow only | Yes (via backends) |
| TensorRT support | Native | Via ONNX | Limited | Via backend |
| Dynamic batching | Yes | Yes | Yes | Continuous batching |
| Ensemble/pipeline | Yes | No | Yes | No |
| LLM optimized | Via TRT-LLM | No | No | Yes (primary focus) |
| Python backend | Yes | Yes | No | N/A |
| Metrics (Prometheus) | Yes | Yes | Yes | Yes |

Triton excels in **heterogeneous model serving environments** — when you need to serve models from multiple frameworks, with TensorRT optimization, or as multi-step ensembles. For pure LLM serving at maximum throughput, vLLM's PagedAttention and continuous batching provide advantages for that specific use case.

## Integration with Kubernetes and Cloud

Triton is commonly deployed on Kubernetes using:

- **NVIDIA GPU Operator**: Manages GPU device plugins and drivers on Kubernetes nodes.
- **Helm charts**: Triton provides official Helm charts for deployment.
- **Horizontal Pod Autoscaler**: Scales Triton pod replicas based on GPU utilization or request rate metrics from Prometheus.
- **Cloud-managed inference**: AWS SageMaker Multi-Model Server and Google Vertex AI Prediction both support Triton as a container runtime, enabling managed infrastructure with Triton's performance.
