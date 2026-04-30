---
title: Introduction to ONNX
description: Learn how ONNX (Open Neural Network Exchange) provides a universal model format for interoperability between ML frameworks — covering the ONNX graph format, exporting models from PyTorch and TensorFlow, running inference with ONNX Runtime, optimization passes, quantization, and deployment across hardware targets.
---

**ONNX (Open Neural Network Exchange)** is an open standard for representing machine learning models. Developed jointly by Microsoft and Facebook in 2017, it defines a common computation graph format that allows models trained in one framework to be deployed in another — eliminating the need to retrain or reimplement models when changing runtimes, hardware, or deployment environments.

The core problem ONNX solves: a PyTorch model trained by a researcher can be exported to ONNX and then deployed using ONNX Runtime on an edge device, optimized with TensorRT for NVIDIA GPUs, or compiled with OpenVINO for Intel hardware — without touching the original training code.

## The ONNX Format

An ONNX model is a **computation graph** serialized as a Protocol Buffer (protobuf) file. The graph consists of:

- **Nodes**: Operations (Conv, Gemm, Relu, BatchNormalization, etc.) drawn from the ONNX operator set.
- **Edges**: Named tensors flowing between nodes.
- **Inputs/Outputs**: Named tensors defining the model's interface.
- **Initializers**: Constant tensors (model weights) embedded in the graph.

ONNX defines a versioned **opset** — operators are versioned so models specify which opset version they target (opset 17 is current as of 2024). This provides forward and backward compatibility.

```
ONNX Model (protobuf):
  ir_version: 8
  opset_imports: [ai.onnx: 17]
  graph:
    nodes:
      - Conv(input, weight, bias) → conv_output
      - BatchNormalization(conv_output, scale, B, mean, var) → bn_output
      - Relu(bn_output) → relu_output
    initializers: [weight, bias, scale, B, mean, var]
    inputs: [input]
    outputs: [relu_output]
```

## Exporting Models to ONNX

### From PyTorch

PyTorch's `torch.onnx.export` traces the model execution and converts it to ONNX:

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Any PyTorch model
model = models.resnet50(pretrained=False)
model.eval()

# Dummy input that matches your actual input shape
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["logits"],
    # Enable dynamic batch size (others are fixed)
    dynamic_axes={
        "image": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    verbose=False
)
print("Model exported to resnet50.onnx")
```

For models with control flow (if/else, loops), use `torch.onnx.export` with `dynamo=True` (PyTorch 2.x) which captures the full graph rather than tracing:

```python
# PyTorch 2.x dynamo export — handles dynamic control flow
export_output = torch.onnx.dynamo_export(model, dummy_input)
export_output.save("resnet50_dynamo.onnx")
```

### From TensorFlow/Keras

```python
import tensorflow as tf
import tf2onnx
import numpy as np

# TensorFlow/Keras model
model = tf.keras.applications.MobileNetV2(weights=None)

# Convert via tf2onnx
spec = (tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input"),)
output_path = "mobilenetv2.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=spec,
    opset=17,
    output_path=output_path
)
print(f"Model exported to {output_path}")
```

### From scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=10).fit(X, y)

# Define input type
initial_type = [("float_input", FloatTensorType([None, 4]))]

# Convert to ONNX
onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=17)

with open("iris_rf.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

## ONNX Model Inspection

```python
import onnx

model = onnx.load("resnet50.onnx")

# Validate the model graph
onnx.checker.check_model(model)

# Inspect graph structure
graph = model.graph
print(f"Inputs: {[i.name for i in graph.input]}")
print(f"Outputs: {[o.name for o in graph.output]}")
print(f"Nodes: {len(graph.node)}")
print(f"Opset: {model.opset_import[0].version}")

# List all operator types used
op_types = {node.op_type for node in graph.node}
print(f"Operators used: {sorted(op_types)}")

# Get input shape
for input_tensor in graph.input:
    shape = [dim.dim_value or dim.dim_param 
             for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Input '{input_tensor.name}' shape: {shape}")
```

Alternatively, **Netron** (a visual model viewer) provides an interactive graph browser for ONNX files at netron.app.

## ONNX Runtime Inference

**ONNX Runtime** (ORT) is a high-performance inference engine for ONNX models, developed by Microsoft. It supports CPU, CUDA, DirectML, TensorRT, CoreML, ROCm, and more via **Execution Providers**:

```python
import onnxruntime as ort
import numpy as np

# List available execution providers
print(ort.get_available_providers())
# e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Create inference session — ORT automatically selects best available provider
session = ort.InferenceSession(
    "resnet50.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # priority order
)

# Inspect I/O metadata
for inp in session.get_inputs():
    print(f"Input: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")

for out in session.get_outputs():
    print(f"Output: {out.name}, shape: {out.shape}")

# Run inference
image = np.random.randn(1, 3, 224, 224).astype(np.float32)

outputs = session.run(
    output_names=["logits"],
    input_feed={"image": image}
)
logits = outputs[0]
print(f"Output shape: {logits.shape}")  # (1, 1000)
predicted_class = logits.argmax(axis=1)[0]
```

### Configuring Session Options

```python
# Performance tuning
sess_options = ort.SessionOptions()

# Thread configuration
sess_options.intra_op_num_threads = 4   # Parallelism within an op
sess_options.inter_op_num_threads = 2   # Parallelism between independent ops

# Graph optimization level
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Save optimized model for reuse (skip optimization on next load)
sess_options.optimized_model_filepath = "resnet50_optimized.onnx"

# Enable profiling for performance analysis
sess_options.enable_profiling = True

session = ort.InferenceSession("resnet50.onnx", sess_options=sess_options,
                                providers=["CPUExecutionProvider"])
```

## ONNX Runtime Graph Optimizations

ORT applies graph-level optimizations automatically:

- **Operator fusion**: Fusing Conv + BatchNorm + Relu into a single kernel (CBR fusion).
- **Constant folding**: Pre-computing subgraphs with only constant inputs.
- **Common subexpression elimination**: Deduplicating identical subgraphs.
- **Layout transformation**: Converting between NCHW and NHWC for optimal hardware performance.

These optimizations typically provide 20–50% speedup on top of the raw ONNX execution.

## Quantization with ONNX Runtime

Post-training quantization converts FP32 weights and activations to INT8, reducing model size by ~4× and improving inference speed on CPUs and quantization-aware hardware:

```python
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader
import numpy as np

# --- Dynamic Quantization (no calibration data needed) ---
# Weights are quantized to INT8; activations remain FP32/INT8 dynamically
quantize_dynamic(
    model_input="resnet50.onnx",
    model_output="resnet50_dynamic_int8.onnx",
    weight_type=QuantType.QInt8
)

# --- Static Quantization (better accuracy, requires calibration data) ---
class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_images: np.ndarray):
        self.images = calibration_images
        self.idx = 0
    
    def get_next(self):
        if self.idx >= len(self.images):
            return None
        batch = {"image": self.images[self.idx:self.idx+1]}
        self.idx += 1
        return batch

# Calibration data: representative subset of production inputs
calibration_data = np.random.randn(100, 3, 224, 224).astype(np.float32)
calibration_reader = ImageCalibrationReader(calibration_data)

quantize_static(
    model_input="resnet50.onnx",
    model_output="resnet50_static_int8.onnx",
    calibration_data_reader=calibration_reader,
    quant_format="QDQ",  # QuantizeLinear/DequantizeLinear operators
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)
```

## Deployment Targets via Execution Providers

| Execution Provider | Hardware | Install |
|---|---|---|
| `CPUExecutionProvider` | x86/ARM CPU | Included |
| `CUDAExecutionProvider` | NVIDIA GPU | `onnxruntime-gpu` |
| `TensorrtExecutionProvider` | NVIDIA TensorRT | `onnxruntime-gpu` |
| `CoreMLExecutionProvider` | Apple Neural Engine | macOS/iOS |
| `OpenVINOExecutionProvider` | Intel CPU/GPU/VPU | `onnxruntime-openvino` |
| `DirectMLExecutionProvider` | DirectX 12 hardware | Windows |
| `ROCmExecutionProvider` | AMD GPU | `onnxruntime-rocm` |

This is ONNX's key value proposition: write the inference code once using ORT's unified API, and switch hardware targets by changing the execution provider.

## ONNX in the ML Pipeline

A practical ONNX workflow:

```
Train (PyTorch/TF) → Export (.onnx) → Validate → Optimize (ORT) → Quantize → Deploy

Tools at each stage:
- Export: torch.onnx.export, tf2onnx, skl2onnx
- Validate: onnx.checker, onnxruntime
- Visualize: Netron
- Optimize: onnxruntime SessionOptions, onnxoptimizer
- Quantize: onnxruntime.quantization
- Deploy: ONNX Runtime (CPU/GPU/edge), TensorRT, OpenVINO, CoreML
```

ONNX is especially valuable in organizations where training and deployment teams use different frameworks or hardware. It cleanly separates the model development lifecycle from the deployment lifecycle — a Python researcher exports once, and platform engineers deploy everywhere.
