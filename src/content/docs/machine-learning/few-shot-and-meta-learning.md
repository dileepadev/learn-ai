---
title: Few-Shot Learning and Meta-Learning
description: Understand few-shot learning and meta-learning — techniques that enable AI models to learn new tasks quickly from very few examples.
---

Few-shot learning and meta-learning address one of the most fundamental limitations of conventional machine learning: the need for large amounts of labeled training data. Inspired by how humans generalize from a handful of examples, these approaches aim to build models that can **adapt to new tasks rapidly with minimal data**.

## The Core Problem

Traditional deep learning models require thousands to millions of labeled examples to learn a task well. However, in many real-world scenarios:

- Labeled data is **scarce or expensive** (medical image annotation, rare event classification).
- New **classes appear continuously** (new product categories, emerging intents).
- Models must generalize to **domains not seen during training**.

Humans, by contrast, can recognize a new animal species after seeing just one or two images. Few-shot learning attempts to close this gap.

## Terminology

| Term | Definition |
|---|---|
| **N-way K-shot task** | A classification task with $N$ classes, each with $K$ labeled examples |
| **Support set** | The $N \times K$ labeled examples available during a few-shot task |
| **Query set** | Unlabeled examples the model must classify using the support set |
| **Episode** | One complete few-shot task (support set + query set) |
| **Meta-training** | Training a model across many episodes to learn *how to learn* |

A **5-way 1-shot** task means: classify inputs into 5 classes, given only 1 labeled example per class.

## Few-Shot Learning Approaches

### 1. Metric-Based Methods

These methods learn an **embedding space** where examples from the same class are close together. Classification is performed by comparing a query to support set examples using a distance metric.

**Siamese Networks:**

- Train a neural network to determine if two inputs are from the same class.
- Uses contrastive loss to bring same-class pairs together and push different-class pairs apart.
- At inference: compare the query to each labeled support example.

**Prototypical Networks:**

- Compute a **prototype** (mean embedding) for each class in the support set.
- Classify a query by finding the nearest prototype.

$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\theta(x_i)$$

$$p_\theta(y = k \mid x) = \frac{\exp(-d(f_\theta(x), c_k))}{\sum_{k'} \exp(-d(f_\theta(x), c_{k'}))}$$

where $f_\theta$ is the embedding network and $d$ is a distance function (e.g., Euclidean).

**Matching Networks:**

- Classify queries using a weighted sum over the entire support set using an attention mechanism.
- The full context of the support set influences each classification.

**Relation Networks:**

- Rather than using a fixed distance metric, learn a **relation module** that scores the similarity between a query and support examples.

### 2. Optimization-Based Methods (Meta-Learning)

Optimization-based meta-learning trains a model's **initial parameters** or **learning algorithm** so that it can adapt to a new task with just a few gradient steps.

**MAML (Model-Agnostic Meta-Learning):**

MAML is the landmark optimization-based meta-learning algorithm. It finds an initialization $\theta$ such that a small number of gradient steps on any new task results in high performance.

**Inner loop** (task adaptation):
$$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$$

**Outer loop** (meta-update across tasks):
$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})$$

MAML is model-agnostic — it works with any differentiable model trained with gradient descent.

**Reptile:**

- A first-order approximation of MAML that is cheaper to compute.
- Repeatedly fine-tunes on tasks and moves the meta-parameters toward the fine-tuned parameters.

**MAML++:**

- Addresses training instability in MAML with per-layer learning rates, annealed learning rates, and multi-step loss.

### 3. Memory-Augmented Methods

These approaches add an external memory to a neural network so it can store and retrieve information from prior episodes.

**Neural Turing Machines (NTMs) and Memory-Augmented Neural Networks (MANNs):**

- Use differentiable read/write operations on an external memory matrix.
- When encountering a new task, new information is written to memory and retrieved for classification.

### 4. Data Augmentation and Hallucination

Another strategy is to **generate synthetic training examples** for new classes to increase the effective number of examples.

- **DAGAN (Data Augmentation GAN):** Trains a conditional GAN to generate new examples by transforming existing ones.
- Diffusion-based augmentation pipelines can similarly hallucinate new variations of scarce classes.

## Meta-Learning: Learning to Learn

Meta-learning — also called **learning to learn** — is the broader framework behind many few-shot learning algorithms. Rather than learning to solve a single task, a meta-learner learns a **general strategy for solving tasks**.

### What Is Learned?

| What is meta-learned | Example |
|---|---|
| Initial model parameters | MAML |
| An optimizer / learning rate | Meta-SGD, Learned Optimizer |
| An embedding space | Prototypical Networks |
| Memory retrieval strategy | MANN |
| Hyperparameters | AutoML / NAS (related area) |

### The Meta-Training Procedure

1. **Sample a task** $\mathcal{T}_i$ from a distribution of tasks $p(\mathcal{T})$.
2. **Adapt** the model to $\mathcal{T}_i$ using the support set (inner loop).
3. **Evaluate** the adapted model on the query set.
4. **Update** the meta-parameters based on query set performance (outer loop).
5. Repeat across thousands of tasks.

The key insight: by training across many different tasks, the model learns to acquire general inductive biases that make adapting to new tasks fast.

## Few-Shot Learning in Large Language Models

Modern large language models have demonstrated remarkable few-shot capabilities through **in-context learning (ICL)** — no gradient updates are required.

**GPT-3 (Brown et al., 2020)** showed that LLMs can perform few-shot tasks by simply including examples in the prompt:

```
Translate to French:
English: The sky is blue. → French: Le ciel est bleu.
English: I love coffee. → French:
```

This is **few-shot prompting** — the model adapts its behavior from examples in the context window without any weight updates.

| Setting | Definition |
|---|---|
| Zero-shot | No examples, task is described in natural language |
| One-shot | One example in the prompt |
| Few-shot | 2–32 examples in the prompt |
| Fine-tuning | Gradient updates on a small labeled dataset |

## Few-Shot Learning Benchmarks

| Benchmark | Domain | Description |
|---|---|---|
| miniImageNet | Vision | 100 classes from ImageNet, 600 images each |
| Omniglot | Vision | 1,623 handwritten characters, 20 instances each |
| SNAIL | Vision + RL | Tasks sampling across multiple domains |
| FLAN / SuperGLUE | NLP | Diverse language understanding tasks |
| CrossFit | NLP | 160 NLP tasks for cross-task generalization |

## Applications

- **Computer vision:** Identifying new product categories, medical conditions, or species from few samples.
- **NLP:** Adapting conversational agents to new intents or domains with minimal annotation.
- **Drug discovery:** Predicting molecular properties when experimental data is limited.
- **Robotics:** Learning new manipulation skills from a handful of demonstrations.
- **Personalization:** Adapting user behavior models from sparse interaction histories.

## Challenges

| Challenge | Details |
|---|---|
| Task distribution mismatch | Meta-training tasks must be representative of test tasks |
| Overfitting to meta-training | Model memorizes episodes instead of learning to generalize |
| Scalability | Episodic training with inner/outer loops is computationally expensive |
| Evaluation variance | Few-shot accuracy estimates have high variance with small support sets |
| LLM in-context limits | Context window constrains how many examples can be used |

## Summary

Few-shot learning and meta-learning represent a paradigm shift toward more data-efficient AI. Key takeaways:

- **Few-shot learning** classifies new examples from very few labeled instances using metric, memory, or optimization-based methods.
- **Meta-learning** trains models to *learn how to learn* by generalizing across many tasks.
- **MAML** learns an initialization that adapts quickly; **Prototypical Networks** learn a distance-based embedding space.
- **Large language models** perform few-shot learning implicitly via in-context learning, without any gradient updates.
- These techniques are critical wherever labeled data is scarce or new tasks emerge continuously.
