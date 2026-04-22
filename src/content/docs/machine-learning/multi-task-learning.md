---
title: Multi-Task Learning
description: Learn how multi-task learning trains a single model on multiple tasks simultaneously, sharing representations to improve generalization, reduce data requirements, and enable transfer across related problems.
---

**Multi-task learning (MTL)** is a machine learning paradigm in which a model is trained to perform **multiple tasks simultaneously**, sharing parameters and representations across tasks rather than training separate models for each. The shared learning process acts as an inductive bias — knowledge useful for one task regularizes and improves the model's representations for other tasks, often improving generalization, especially when individual tasks have limited data.

MTL is motivated by how humans learn: we do not develop completely independent modules for reading, writing, and arithmetic — we develop shared cognitive capabilities that transfer across these activities.

## The Core Principle: Shared Representations

In single-task learning, a model optimizes:

$$\min_\theta \mathcal{L}_\text{task}(\theta)$$

In multi-task learning, a model with shared parameters $\theta_\text{shared}$ and task-specific parameters $\{\theta_k\}$ optimizes:

$$\min_{\theta_\text{shared}, \{\theta_k\}} \sum_{k=1}^{K} \lambda_k \mathcal{L}_k(\theta_\text{shared}, \theta_k)$$

where $\lambda_k$ is the weight of task $k$'s loss, and $\mathcal{L}_k$ is the loss for task $k$. The shared parameters are updated by gradients from all tasks simultaneously.

The intuition: gradients from related tasks provide additional training signal for the shared layers, improving representation quality even for tasks with limited training data.

## Why Multi-Task Learning Works

### Implicit Data Augmentation

Training on multiple tasks effectively augments the training data for each task. The shared layers see more diverse examples than any single task would provide, improving the quality and generalizability of the learned representations.

### Regularization Through Auxiliary Tasks

Even if an auxiliary task is not the primary goal, it regularizes the shared representation. A model learning both document classification and named entity recognition simultaneously cannot learn representations that are useful only for one — it must develop representations that capture linguistic structure useful for both.

This regularization effect is particularly valuable when the primary task has limited labeled data.

### Attention Focusing

MTL can help a model focus on relevant features by using auxiliary tasks to indicate which input features matter. For example, adding a part-of-speech tagging auxiliary task to a sentiment analysis model encourages the shared encoder to capture syntactic structure, which turns out to be useful for sentiment.

### Eavesdropping

Some features are easy to learn for one task but difficult for another. MTL allows the "difficult" task to eavesdrop on representations learned for the "easy" task — benefiting from features it would struggle to learn independently.

## Architectural Patterns

### Hard Parameter Sharing

The most common MTL architecture. All tasks share a common backbone (the "trunk"), with task-specific heads:

```
               Input
                 │
         ┌───────▼───────┐
         │  Shared       │
         │  Encoder      │
         └───┬───┬───┬───┘
             │   │   │
           Head Head Head
          Task1 Task2 Task3
```

The shared encoder is updated by gradients from all task heads simultaneously. Task-specific heads are updated only by their own task's gradient.

**Trade-off**: Hard sharing forces all tasks to use the same representation, which is efficient but may hurt when tasks require very different representations.

### Soft Parameter Sharing

Each task has its own model, but the parameters are regularized to be similar across tasks. For example, L2 regularization between corresponding parameters of different task models:

$$\mathcal{L}_{reg} = \sum_{i < j} \| \theta_i - \theta_j \|^2$$

This allows task-specific representations while still encouraging knowledge sharing. Bayesian approaches place priors that couple the models of related tasks.

### Cross-Stitch Networks

**Cross-stitch networks** (Misra et al., 2016) learn how to combine task-specific representations at each layer. Each task has its own network, but the activation maps of each network at each layer are linearly combined with the activation maps of other networks before being passed to the next layer. The combination weights ($\alpha_{ij}$ determining how much network $i$'s features are passed to network $j$) are learned during training.

### Multi-Gate Mixture of Experts (MMoE)

**MMoE** (Ma et al., 2018, Google) uses a set of **expert networks** (each a small MLP) with a learned **gating network** per task:

- Each expert processes the shared input.
- Each task's gating network computes a weighted combination of expert outputs.
- Task-specific towers take their gate's mixture of experts as input.

Different tasks can learn to use different mixtures of experts, enabling flexible sharing: related tasks share experts; less related tasks use different experts.

### Task-Specific Adapters

Inspired by parameter-efficient fine-tuning, **adapter-based MTL** inserts small task-specific adapter modules into a shared backbone. The backbone is frozen or only lightly updated; the adapters are the primary per-task parameters. This enables efficient scaling to hundreds of tasks.

## Loss Weighting and Task Balancing

A fundamental challenge in MTL is **how to weight the losses of different tasks**. Poor weighting can cause one task to dominate training, degrading performance on others.

### Fixed Weighting

The simplest approach: manually assign $\lambda_k$ for each task. Requires hyperparameter tuning and doesn't adapt during training.

### Uncertainty Weighting (Kendall et al., 2018)

Tasks with higher output uncertainty should receive lower loss weights — otherwise the model over-focuses on highly variable tasks. **Uncertainty weighting** models the noise level of each task as a learnable parameter $\sigma_k$ and weights losses accordingly:

$$\mathcal{L} = \sum_k \frac{1}{2\sigma_k^2} \mathcal{L}_k + \log \sigma_k$$

The model learns to down-weight noisy tasks automatically. The $\log \sigma_k$ term prevents the model from simply setting all $\sigma_k \to \infty$ to minimize loss.

### GradNorm

**GradNorm** (Chen et al., 2018) normalizes the gradient magnitudes of all tasks to be approximately equal. At each step, it adjusts task weights $\lambda_k$ to equalize the gradient norms — preventing any single task from dominating the gradient update.

### PCGrad (Projecting Conflicting Gradients)

Tasks may have **conflicting gradients** — gradients that point in opposite directions, causing one task's update to undo another's. **PCGrad** detects conflicting task gradients (negative cosine similarity) and projects each task's gradient onto the plane orthogonal to the conflicting task's gradient, removing the conflicting component before applying the update.

## Task Relationship and Negative Transfer

A critical consideration: **not all tasks benefit from sharing**. When tasks are unrelated or have conflicting objectives, forced sharing can hurt performance — a phenomenon called **negative transfer**.

### Signs of Negative Transfer

- Multi-task performance is worse than single-task on one or more tasks.
- Gradient conflicts between tasks are frequent and severe.
- Task-specific heads diverge significantly from the shared representation.

### Measuring Task Relatedness

- **Gradient similarity**: High cosine similarity between task gradients suggests beneficial sharing.
- **Representational alignment**: Measuring whether the features useful for task A overlap with features useful for task B (e.g., CKA similarity between single-task models).
- **Performance correlation**: Tasks that improve together in single-task learning (when given more data) tend to benefit from sharing.

### Selective Sharing

Rather than sharing all parameters, **hierarchical MTL** shares lower-level features (syntax, morphology) across all tasks and uses task-specific higher-level features. Tasks that are more similar share more layers; less related tasks branch off earlier.

## Multi-Task Learning in NLP

MTL has had a transformative impact on NLP. The **MT-DNN** (Liu et al., 2019) extended BERT pretraining with multi-task fine-tuning on GLUE tasks simultaneously, achieving state-of-the-art performance across 8 NLP tasks.

**T5** and **mT5** can be viewed as extreme MTL: the same model is fine-tuned on translation, summarization, question answering, classification, and other tasks simultaneously — all framed as text-to-text transformations.

**GPT-3 and successors** achieve implicit MTL through instruction fine-tuning: training on hundreds of diverse task demonstrations enables generalization to new tasks without explicit sharing architecture.

Common NLP auxiliary tasks that improve primary task performance:

- **POS tagging**: Improves Named Entity Recognition and sentiment analysis.
- **Dependency parsing**: Improves semantic role labeling.
- **Language modeling**: Improves virtually all NLP tasks.
- **Named entity recognition**: Improves question answering and relation extraction.

## Multi-Task Learning in Computer Vision

**Multi-task vision models** train a single model for detection, segmentation, and classification simultaneously:

- **Detectron2 and Mask R-CNN**: Jointly train classification, detection, and instance segmentation.
- **HydraNet** (Tesla Autopilot): A single backbone with multiple task heads for lane detection, object detection, depth estimation, and traffic sign classification — all processing the same camera frames.
- **UberNet**: Early work training a single network on 7 computer vision tasks simultaneously.
- **ViT-Adapter**: Adds task-specific adapter modules to a ViT backbone for dense prediction tasks (detection, segmentation, depth).

## Multi-Task vs. Transfer Learning vs. Meta-Learning

These paradigms are often confused:

**Multi-task learning**: Train jointly on multiple tasks from the start. All tasks are known at training time. The goal is to improve performance on all tasks simultaneously.

**Transfer learning**: Train on a source task, then fine-tune on a target task. The tasks are not trained simultaneously. The goal is to improve the target task using knowledge from the source.

**Meta-learning**: Train on many tasks to learn *how to learn*, enabling fast adaptation to new tasks with few examples. MTL and meta-learning overlap but differ in objective — MTL optimizes current task performance; meta-learning optimizes learning speed.

In practice, these approaches are often combined: pretrain with CLM (transfer learning), fine-tune with MTL on downstream tasks (multi-task), and use prompting for few-shot adaptation (meta-learning).

## Practical Recommendations

**Start with hard parameter sharing**: It is the simplest approach and works well when tasks are related. Add task-specific components only if performance is insufficient.

**Balance your training batches**: Sample examples from tasks proportionally (or use temperature sampling with $T = 0.1$ to $0.3$ for tasks with very different dataset sizes) to prevent large tasks from dominating.

**Monitor per-task validation metrics separately**: A single combined metric can mask that multi-task training is helping some tasks while hurting others.

**Experiment with task auxiliary loss weights**: Even simple grid search over $\lambda_k \in \{0.1, 0.5, 1.0\}$ can make a significant difference.

**Check for negative transfer**: If a task's multi-task performance is worse than its single-task baseline, consider whether it is truly related to the other tasks or should be trained separately.

Multi-task learning is a powerful tool when applied to genuinely related tasks — enabling data-efficient, generalizable, and compact models that transfer well to new domains. As AI systems increasingly need to handle diverse real-world tasks, MTL provides a principled framework for building capable, general-purpose models.
