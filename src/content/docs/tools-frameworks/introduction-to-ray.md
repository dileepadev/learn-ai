---
title: Introduction to Ray
description: Learn how Ray enables scalable distributed Python for AI and ML workloads — from parallel task execution and actor-based services to Ray Train, Ray Tune, Ray Serve, and Ray Data for end-to-end ML pipelines.
---

**Ray** is an open-source distributed computing framework for Python that makes it straightforward to scale AI and machine learning workloads from a laptop to a cluster of thousands of machines. Originally developed at UC Berkeley's RISELab and now maintained by Anyscale, Ray provides a unified compute layer for the full AI/ML lifecycle — from data preprocessing to distributed training, hyperparameter tuning, and production model serving.

Unlike frameworks that address a single part of the ML workflow, Ray is a general-purpose distributed runtime with a rich ecosystem of AI-specific libraries (collectively known as **Ray AIR** — AI Runtime) built on top of it. This enables organizations to build end-to-end ML pipelines on a single, unified infrastructure rather than stitching together multiple incompatible distributed systems.

## Core Concepts

### Tasks: Stateless Distributed Functions

A **Ray task** is a stateless function that executes remotely and asynchronously. Converting a regular Python function to a distributed Ray task requires a single decorator:

```python
import ray

ray.init()  # Connect to a Ray cluster (or start a local one)

# Regular Python function
def compute_square(x):
    return x * x

# Ray remote task — executes on a worker in the cluster
@ray.remote
def compute_square_remote(x):
    return x * x

# Execute 1,000 tasks in parallel across the cluster
futures = [compute_square_remote.remote(i) for i in range(1000)]
results = ray.get(futures)  # Block until all tasks complete
```

Tasks support dependency passing — one task's output can be passed as input to another without `ray.get`, enabling pipelined parallel execution:

```python
@ray.remote
def load_data(path):
    return read_file(path)

@ray.remote
def preprocess(data):
    return transform(data)

@ray.remote
def train_model(processed_data):
    return fit(processed_data)

# Pipelined execution — no intermediate ray.get()
data_ref = load_data.remote("data.csv")
processed_ref = preprocess.remote(data_ref)
model_ref = train_model.remote(processed_ref)

model = ray.get(model_ref)
```

### Actors: Stateful Distributed Services

A **Ray actor** is a stateful worker — a class instance that runs in the cluster and can be called remotely. Actors maintain state across method calls:

```python
@ray.remote
class ModelServer:
    def __init__(self, model_path):
        import torch
        self.model = torch.load(model_path)
        self.request_count = 0

    def predict(self, input_data):
        self.request_count += 1
        return self.model(input_data)

    def get_request_count(self):
        return self.request_count

# Create two actor instances (each runs in its own process)
server1 = ModelServer.remote("model_v1.pt")
server2 = ModelServer.remote("model_v2.pt")

# Call methods asynchronously
result1 = server1.predict.remote(input_batch)
result2 = server2.predict.remote(input_batch)

outputs = ray.get([result1, result2])
count = ray.get(server1.get_request_count.remote())
```

### Object Store: Shared Memory

Ray's **distributed object store** (built on Apache Arrow / Plasma) enables zero-copy data sharing between tasks and actors on the same node, and efficient serialization for cross-node transfers. Large numpy arrays and pandas DataFrames can be placed in the object store and referenced by multiple workers without copying:

```python
# Put a large dataset in the shared object store once
large_dataset = numpy.random.randn(1_000_000, 512)
dataset_ref = ray.put(large_dataset)

# 100 workers all reference the same object — zero copies on same node
results = [process_chunk.remote(dataset_ref, i) for i in range(100)]
```

## Ray Train: Distributed Model Training

**Ray Train** provides distributed training for PyTorch, TensorFlow, Hugging Face, and XGBoost — handling the distributed setup, checkpoint management, and fault tolerance automatically.

### PyTorch Distributed Training

```python
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func(config):
    # This function runs on each worker
    import torch
    from torch.nn.parallel import DistributedDataParallel

    # Ray Train automatically sets up the distributed environment
    model = MyModel().to(train.get_context().get_local_rank())
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    train_loader = train.torch.prepare_data_loader(get_dataloader(config))

    for epoch in range(config["epochs"]):
        for batch in train_loader:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Checkpoint from any worker — Ray Train handles coordination
        train.report({"loss": loss.item()}, checkpoint=train.Checkpoint.from_dict(
            {"model_state": model.state_dict()}
        ))

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"lr": 1e-4, "epochs": 10},
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 4},
    ),
)

result = trainer.fit()
```

### Hugging Face Integration

```python
from ray.train.huggingface import TransformersTrainer

def train_func(config):
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir="/tmp/results",
        num_train_epochs=3,
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        # Ray Train handles distributed coordination automatically
    )

    trainer = Trainer(
        model=get_model(),
        args=training_args,
        train_dataset=get_dataset(),
    )
    trainer.train()

trainer = TransformersTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"batch_size": 16, "lr": 2e-5},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
```

## Ray Tune: Distributed Hyperparameter Optimization

**Ray Tune** is a distributed hyperparameter optimization framework supporting grid search, random search, Bayesian optimization, and advanced algorithms like PBT (Population Based Training) and ASHA.

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def trainable(config):
    model = build_model(config["hidden_size"], config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(100):
        train_loss = train_epoch(model, optimizer)
        val_accuracy = validate(model)

        # Report metrics — Tune uses these for scheduling decisions
        tune.report({"val_accuracy": val_accuracy, "train_loss": train_loss})

tuner = tune.Tuner(
    trainable,
    param_space={
        "lr": tune.loguniform(1e-5, 1e-2),
        "hidden_size": tune.choice([128, 256, 512, 1024]),
        "dropout": tune.uniform(0.0, 0.5),
        "batch_size": tune.choice([32, 64, 128]),
    },
    tune_config=tune.TuneConfig(
        metric="val_accuracy",
        mode="max",
        num_samples=100,
        search_alg=OptunaSearch(),
        scheduler=ASHAScheduler(
            max_t=100,
            grace_period=10,
            reduction_factor=2,
        ),
    ),
    run_config=train.RunConfig(
        storage_path="s3://my-bucket/ray-results",
    ),
)

results = tuner.fit()
best_result = results.get_best_result("val_accuracy", "max")
print(f"Best config: {best_result.config}")
```

**ASHA (Asynchronous Successive Halving)** aggressively terminates underperforming trials, allocating compute to the most promising configurations — achieving high-quality results with far fewer total GPU-hours than exhaustive search.

## Ray Serve: Production Model Serving

**Ray Serve** is a scalable model serving library built on top of Ray actors. It supports:

- HTTP and Python API endpoints.
- Multiple models per deployment.
- A/B testing and traffic splitting.
- Dynamic batching for throughput optimization.
- Autoscaling based on request rate.

```python
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 10,
    },
)
class LLMDeployment:
    def __init__(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

    async def __call__(self, request):
        data = await request.json()
        prompt = data["prompt"]
        outputs = self.llm.generate([prompt], self.sampling_params)
        return {"response": outputs[0].outputs[0].text}

# Deploy and get a handle
app = LLMDeployment.bind()
serve.run(app)
```

### Deployment Graphs

Ray Serve supports **deployment graphs** — composing multiple models into a pipeline where the output of one deployment feeds into the next:

```python
@serve.deployment
class EmbeddingModel:
    def __call__(self, text): ...

@serve.deployment
class RerankingModel:
    def __call__(self, query, candidates): ...

@serve.deployment
class RAGPipeline:
    def __init__(self, embedder, reranker):
        self.embedder = embedder
        self.reranker = reranker

    async def __call__(self, request):
        data = await request.json()
        query_embedding = await self.embedder.remote(data["query"])
        candidates = retrieve_from_vector_db(query_embedding)
        reranked = await self.reranker.remote(data["query"], candidates)
        return generate_answer(reranked)

# Compose the pipeline
app = RAGPipeline.bind(EmbeddingModel.bind(), RerankingModel.bind())
```

## Ray Data: Scalable Data Processing

**Ray Data** provides distributed dataset processing for ML data pipelines — loading, transforming, and streaming data across a cluster at scale.

```python
import ray

# Create a dataset from various sources
ds = ray.data.read_parquet("s3://my-bucket/train-data/")

# Apply transformations in parallel
ds = ds.map(preprocess_text, num_cpus=2)
ds = ds.filter(lambda row: len(row["text"]) > 100)

# Map GPU-accelerated batch transforms
ds = ds.map_batches(
    run_embedding_model,
    batch_size=256,
    num_gpus=1,
    concurrency=4,
)

# Write results
ds.write_parquet("s3://my-bucket/processed-data/")
```

Ray Data is particularly powerful for **streaming preprocessing** — loading and transforming batches from object storage or databases just in time for GPU training, avoiding the need to hold the full dataset in memory.

## Ray Cluster Architecture

A Ray cluster consists of:

- **Head node**: Runs the Ray global control store (GCS), dashboard, and job submission API. Coordinates scheduling and maintains cluster state.
- **Worker nodes**: Execute tasks and host actor processes. Each node contributes its CPUs, GPUs, and memory to the cluster resource pool.
- **Raylet**: A per-node process that manages local resource scheduling and communicates with the GCS.

Clusters can be launched on:

- **Local machine**: `ray.init()` starts a local cluster automatically.
- **Kubernetes**: Ray Operator manages Ray clusters as Kubernetes custom resources.
- **AWS/GCP/Azure**: KubeRay and cloud-provider integrations automate cluster provisioning.
- **Anyscale**: The commercial platform built on Ray, providing managed cluster operations, monitoring, and enterprise support.

## Why Ray for AI/ML

Ray's design makes it particularly well-suited for the specific patterns that appear in AI/ML workloads:

- **Heterogeneous resources**: Tasks can request fractional GPUs (`@ray.remote(num_gpus=0.25)`) — enabling multiple small inference tasks to share a GPU.
- **Dynamic task graphs**: ML pipelines often have conditional branching, nested parallelism, and dynamic graph structures that static frameworks like Apache Spark handle poorly.
- **Python-native**: Ray works entirely within Python, without requiring code changes to move from local to distributed execution.
- **Fault tolerance**: Lineage-based reconstruction automatically re-executes failed tasks, enabling long-running training jobs on preemptible instances.

Ray has become a core component of the modern AI infrastructure stack, used alongside PyTorch, vLLM, Hugging Face Transformers, and cloud object storage to build scalable, reproducible ML systems.
