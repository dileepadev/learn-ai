---
title: Multi-Task and Transfer Learning
description: Leveraging knowledge across tasks and domains — fine-tuning, domain adaptation, and training on multiple related objectives.
---

**Multi-task learning** and **transfer learning** are fundamental techniques for improving generalization and sample efficiency by sharing knowledge across tasks and domains. Rather than solving each task in isolation, we leverage related tasks and domain knowledge.

## Transfer Learning

**Transfer learning**: Train on a source task/domain, then adapt to a target task/domain. Enables learning from limited target data by leveraging larger source datasets.

### Pre-training and Fine-Tuning

The dominant paradigm in modern deep learning:

1. **Pre-train** on a large, general source task (e.g., ImageNet for vision, web text for NLP).
2. **Fine-tune** on the target task with limited labeled data.

**Why it works**:
- Early layers learn general features (edges, textures in vision; semantics in NLP).
- Fine-tuning adapts higher layers to task-specific patterns.
- Reduces data requirements for the target task by 10-100x.

### Example: Fine-Tuning BERT

**BERT** (Bidirectional Encoder Representations from Transformers) is pre-trained on 3.3 billion words from Wikipedia and BookCorpus via masked language modeling.

Fine-tuning on downstream tasks:
- Sentiment analysis: ~2K labeled examples needed (vs. 100K without pretrain).
- Named entity recognition: ~5K examples (vs. 50K without pretrain).

The transfer significantly improves sample efficiency.

### Feature Extraction

Simpler than fine-tuning: use a pre-trained model as a fixed feature extractor.

```python
pretrained_features = pretrained_model(input)  # frozen
classifier = LinearLayer(pretrained_features)  # train only this
```

Trade-off: Faster training but less adaptation than fine-tuning. Works well when target task is similar to source task.

## Domain Adaptation

**Domain shift**: Distribution of source and target data differ. Example:

- **Source domain**: Natural photos of everyday objects.
- **Target domain**: Synthetic renderings, sketch drawings, or thermal images of the same objects.

A model trained on source domain overfits to source-specific patterns (lighting, texture) and fails on target domain.

### Domain-Invariant Features

Learn representations invariant to domain while predictive of the task:

1. **Feature extractor** $f$: Maps input to representation.
2. **Task classifier** $h$: Predicts task label.
3. **Domain classifier** $d$: Predicts domain (adversarial).

Optimize:

$$\min_f \min_h L_{\text{task}}(h(f(x_s)), y_s) - \lambda \max_d L_{\text{domain}}(d(f(x_t)), \text{target domain})$$

The domain adversary pushes representations toward domain invariance; the task classifier ensures task information is retained.

### Distribution Matching

Explicitly align source and target distributions:

$$\mathcal{L}_{\text{da}} = \|\mu_s - \mu_t\|_2^2$$

where $\mu_s, \mu_t$ are mean embeddings of source/target data.

Variants:
- **Maximum Mean Discrepancy (MMD)**: Match all moments.
- **Correlation alignment (CORAL)**: Align correlation structures.

### Self-Training

Use predictions on unlabeled target data as pseudo-labels:

1. Train on source domain.
2. Make predictions on target domain (unlabeled).
3. Select high-confidence predictions as pseudo-labels.
4. Retrain on pseudo-labeled target data.

Improves adaptation by leveraging target domain statistics. Risk: Incorrect pseudo-labels can degrade performance.

## Multi-Task Learning

**Multi-task learning**: Train a single model on multiple related tasks simultaneously.

### Hard Sharing

Share layers early in the network; task-specific heads branch off:

```
Shared layers (encoders)
        |
    ----+----
    |   |   |
Task1 Task2 Task3 (task-specific heads)
```

Early layers learn general features; task-specific heads learn task-specialized patterns.

**When to use**: Tasks share low-level features (e.g., vision tasks: segmentation, depth estimation, surface normals).

### Soft Sharing

Different task models share parameters via regularization:

$$\mathcal{L} = \sum_t \mathcal{L}_t(\theta_t) + \lambda \sum_{t \neq t'} \|\theta_t - \theta_{t'}\|_2^2$$

Encourages task-specific parameters $\theta_t$ to be similar without forcing them identical.

**Flexibility**: Each task has its own model; parameters are coupled but distinct.

### Multi-Task Objective

Optimize multiple losses simultaneously:

$$\mathcal{L}_{\text{total}} = \sum_t w_t \mathcal{L}_t$$

where $w_t$ are task weights. Naive approach: equal weights ($w_t = 1$). Better approaches:

#### Uncertainty Weighting

Learn task weights that reflect task uncertainty:

$$\mathcal{L}_{\text{total}} = \sum_t \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t$$

Tasks with high uncertainty get lower weight. As the model improves a task (lower loss), learned uncertainty decreases, increasing weight to focus on harder tasks.

#### Gradient Normalization

Normalize gradients by their magnitude to prevent any single task from dominating:

$$\mathcal{L}_{\text{total}} = \sum_t \mathcal{L}_t / \|\nabla \mathcal{L}_t\|$$

All tasks contribute equally to optimization regardless of loss scale.

## Task Relationships

### Positive Transfer

One task helps another. Example: Pre-training on ImageNet helps vision tasks. Positive transfer occurs when tasks share useful patterns.

### Negative Transfer

One task hurts another. Example: Multitask learning on vision and language together may degrade language performance if architecture bottlenecks vision. Avoid by:
- Careful architecture design (separate task heads).
- Task selection (related tasks transfer better).

### Task Similarity Metrics

Quantify task similarity to predict transfer:

- **Representation distance**: How different are learned representations?
- **Loss correlation**: Do tasks improve/degrade together during training?
- **Feature importance**: Do tasks use similar features?

Use these to select which tasks to combine for multitask learning.

## Meta-Learning Perspective

Multi-task learning is related to **meta-learning** ("learning to learn"):

- **Multitask learning**: Learn parameters shared across tasks.
- **Meta-learning**: Learn an initialization or algorithm that generalizes to new tasks.

A meta-learned model adapts quickly to new tasks with few examples (few-shot learning).

## Applications

### Natural Language Processing

Pre-train on multiple NLP tasks jointly (masked language modeling, next sentence prediction, named entity recognition). The shared representations improve performance on all tasks.

### Computer Vision

Auxiliary tasks improve main task:
- **Main task**: Object detection.
- **Auxiliary tasks**: Semantic segmentation, depth estimation, surface normals.
- Shared encoder benefits from multi-task supervision.

### Recommendation Systems

Predict multiple user properties simultaneously:
- Whether user will click (CTR prediction).
- User demographics (age, location).
- User interests (categories).

Shared representation improves all predictions.

### Medical Imaging

Diagnose multiple diseases from the same image using multi-task learning, improving accuracy for rare conditions via knowledge transfer from common conditions.

## Challenges

### Task Interference

Some task pairs hurt each other. Careful task selection and architecture design needed.

### Scalability

Multitask learning with hundreds of tasks becomes complex. Efficient approaches:
- Modularity: Task-specific modules.
- Hierarchical task structures: Organize tasks into hierarchies; share higher-level knowledge.

### Fairness Across Tasks

Optimizing aggregate performance can hurt minority tasks. Monitor per-task performance; reweight if needed.

## Best Practices

1. **Task selection**: Choose related tasks that share features or structure.
2. **Weight tuning**: Invest in task weights (uncertainty weighting, gradient normalization).
3. **Architecture design**: Clearly separate shared and task-specific components.
4. **Monitor per-task performance**: Don't just optimize aggregate loss; track individual tasks.
5. **Positive transfer validation**: Confirm tasks help each other; add auxiliary tasks only if they improve main task.

Multi-task and transfer learning are essential for practical AI, enabling efficient learning from limited data by leveraging related tasks and domains.
