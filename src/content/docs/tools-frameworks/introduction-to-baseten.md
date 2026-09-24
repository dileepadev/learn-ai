---
title: Introduction to Baseten - Deploying Custom Models to Production
description: Learn how Baseten packages custom and open-source model deployment with autoscaling GPU infrastructure and a focus on production latency.
---

Baseten is a model deployment platform focused on taking custom-trained or open-source models from a checkpoint to a production-ready, autoscaling API endpoint, with particular emphasis on inference performance engineering.

## Packaging and Deploying a Model

Baseten uses Truss, an open-source model packaging framework, to define a model's dependencies, preprocessing, and inference logic in a standard, portable format:

```python
# model/model.py
class Model:
    def load(self):
        self._model = load_my_checkpoint()

    def predict(self, model_input):
        return self._model.generate(model_input["prompt"])
```

```bash
truss push --publish
```

Because Truss packages are a standard format rather than tied to one platform, the same package definition can be tested locally, deployed to Baseten, or in principle ported to other Truss-compatible infrastructure, reducing platform lock-in compared to a fully proprietary packaging format.

## Performance-Focused Infrastructure

Baseten emphasizes inference-specific optimizations — efficient GPU autoscaling that keeps cold-start latency low, request batching, and support for optimized inference engines like TensorRT-LLM and vLLM under the hood — aimed at teams whose primary constraint is serving a specific custom model with production-grade latency rather than using a pre-packaged model from a catalog.

## When Custom Deployment Makes Sense

Custom deployment platforms matter most when you're serving a fine-tuned model, a model not available through any standard hosted API, or a model requiring custom pre/post-processing logic that a generic hosted inference API doesn't accommodate — situations where a one-size-fits-all managed model API doesn't fit your specific model or pipeline.

## Practical Guidance

Choose a platform like Baseten when you need to deploy a custom or fine-tuned model to production without building and operating GPU autoscaling infrastructure yourself. If you're only using widely available open or proprietary models without custom fine-tuning or custom inference logic, a simpler managed inference API is usually less operational overhead than packaging and deploying the model yourself.
