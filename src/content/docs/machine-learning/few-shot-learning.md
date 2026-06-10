---
title: Few-Shot Learning
description: Learning from limited data — exploring meta-learning approaches that enable models to quickly adapt to new tasks with minimal examples.
---

**Few-shot learning** is the challenge of training a machine learning model to perform well on new tasks using only a handful of labeled examples — typically 1 to 10 samples per class. This mirrors human learning: we can recognize a new animal species from just a few photos, yet standard deep learning models require thousands of examples.

## The Problem

Traditional supervised learning relies on large labeled datasets. When data is scarce — in rare disease diagnosis, low-resource languages, or specialized domains — standard approaches fail. Few-shot learning addresses this critical bottleneck.

## Meta-Learning Framework

Few-shot learning is approached through **meta-learning** ("learning to learn"): training a model on many diverse tasks such that it develops strategies to quickly adapt to new tasks.

### Training Setup

1. **Task Distribution**: Sample many related tasks from a task distribution.
2. **Support Set**: A small labeled dataset for a new task (the "few shots").
3. **Query Set**: Unlabeled examples to evaluate performance on the new task.

The model is trained end-to-end to minimize query-set loss, encouraging it to learn representations and initialization strategies that generalize quickly.

## Key Approaches

### Model-Agnostic Meta-Learning (MAML)

**MAML** (Finn et al., 2017) finds an initialization of model parameters that can be quickly fine-tuned to new tasks with just a few gradient steps.

The algorithm:
1. Sample a batch of tasks from the task distribution.
2. For each task: take a few gradient steps on the support set to get task-specific parameters.
3. Compute loss on the query set using the updated parameters.
4. Perform a meta-update to the original parameters.

This encourages the base model to learn general representations that require minimal task-specific adaptation.

### Prototypical Networks

**Prototypical Networks** (Snell et al., 2017) learn a **metric space** where classification is performed by distance to class prototypes.

For each class $c$:
- Compute the class prototype: the mean embedding of support examples.
- Classify query examples by nearest prototype (e.g., using Euclidean distance).

The model learns embeddings such that examples of the same class cluster together, while different classes separate.

### Relation Networks

**Relation Networks** (Sung et al., 2018) learn a similarity metric between support and query examples through a neural network.

Rather than using a fixed distance metric (Euclidean, cosine), the model learns a task-specific metric by training on many tasks. This is more flexible and often achieves better performance than Prototypical Networks.

### Matching Networks

**Matching Networks** (Vinyals et al., 2016) use **attention** to soft-match query examples to support examples.

Each query is classified as a weighted combination of support classes, where weights come from an attention mechanism computed between query and support embeddings. This approach is naturally differentiable and easy to train end-to-end.

## N-Way, K-Shot Notation

Few-shot learning problems are typically described as **N-way, K-shot**:
- **N-way**: The number of classes in a task.
- **K-shot**: The number of labeled examples per class.

For example, **5-way, 1-shot** means classifying between 5 classes with only 1 example per class.

## Transfer Learning vs. Meta-Learning

| Approach | Few-Shot Use | Data Efficiency | Task Diversity |
|----------|--------------|-----------------|-----------------|
| **Fine-tuning** | Moderate | Moderate | Limited — assumes similar domains |
| **Meta-Learning** | Strong | High — learns to learn | Excellent — trained on task diversity |

Fine-tuning a pretrained model is effective but assumes the pretrain task is similar to the target. Meta-learning is designed specifically for rapid adaptation to diverse tasks.

## Cross-Domain Few-Shot Learning

Practical few-shot learning often involves **domain shift**: training on one set of tasks (e.g., common objects) and testing on different tasks (e.g., medical images).

Approaches to reduce domain shift:
- **Task augmentation**: Generate synthetic tasks to increase task diversity during meta-training.
- **Domain-agnostic learning**: Use data augmentation and regularization to learn more general representations.
- **Episodic training**: Explicitly structure training to mimic the domain shift seen at test time.

## Few-Shot Learning in NLP

Few-shot learning is crucial in NLP due to the high cost of annotation:

- **Intent detection**: Classifying user queries with only a few examples per intent.
- **Relation extraction**: Extracting relationships between entities with minimal labeled data.
- **Named entity recognition**: Tagging entities in languages with limited training data.

Prompt-based few-shot methods (in-context learning with large language models) are particularly effective in NLP, where models like GPT-3 perform remarkably well using only textual demonstrations in the prompt.

## Practical Challenges

**Class imbalance**: Few-shot datasets are inherently imbalanced — only a few examples per class. Standard techniques for handling imbalance (reweighting, oversampling) remain important.

**Overfitting**: With so few examples, overfitting is a serious risk. Regularization, data augmentation, and careful validation are essential.

**Benchmark saturation**: Standard benchmarks (mini-ImageNet, CUB, Omniglot) are becoming saturated. Researchers are moving toward more diverse and challenging benchmarks that require genuine generalization.

## Current Research Directions

- **Zero-shot learning**: Classifying entirely new classes without any support examples, using semantic information like class descriptions.
- **Open-set few-shot learning**: Handling both known and unknown classes in few-shot scenarios.
- **Continual few-shot learning**: Adapting to new tasks sequentially without catastrophic forgetting of earlier tasks.

Few-shot learning remains an active frontier, crucial for real-world AI systems where large annotated datasets are unrealistic.
