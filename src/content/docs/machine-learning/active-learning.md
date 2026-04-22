---
title: Active Learning
description: Learn how active learning reduces labeling costs by strategically selecting which data points to annotate, enabling high-performing ML models with a fraction of the labeled data required by passive approaches.
---

**Active learning** is a machine learning paradigm in which the learning algorithm interactively queries a human annotator (or oracle) to label specific data points, rather than passively learning from a randomly sampled labeled dataset. By intelligently choosing *which* examples to label, active learning can achieve the same model performance as fully supervised learning with dramatically fewer labeled examples.

This matters because labeling data is expensive. Medical imaging annotation requires expert radiologists; legal document classification requires lawyers; speech transcription requires skilled linguists. Active learning directly attacks this bottleneck.

## The Core Idea

In a typical passive supervised learning workflow:

1. Collect a large pool of unlabeled data.
2. Randomly sample a subset and pay humans to label it.
3. Train a model on the labeled set.
4. Evaluate and deploy.

**Active learning** replaces step 2 with a strategic selection process:

1. Start with a small initial labeled set (or even none).
2. Train an initial model.
3. **Query**: Use a selection strategy to identify the most informative unlabeled examples.
4. **Annotate**: A human labels the selected examples.
5. Add newly labeled examples to the training set.
6. Retrain and repeat from step 3.

The loop continues until a budget (labeling cost or target performance) is exhausted.

## Why Some Examples Are More Informative Than Others

Not all unlabeled examples contribute equally to model improvement. An example is most informative if:

- The model is **uncertain** about its prediction — the model doesn't know how to classify it and would benefit from knowing the true label.
- It is **representative** of a large cluster of unlabeled data — labeling one well-chosen example provides information applicable to many similar unlabeled points.
- It **contradicts or refines** the current decision boundary.

Random sampling ignores all of this structure. Active learning exploits it.

## Query Strategies

The heart of active learning is the **query strategy** — the algorithm that scores unlabeled examples by expected informativeness and selects the best ones to label.

### Uncertainty Sampling

The simplest and most widely used family of strategies. Select examples where the model is least confident:

**Least confidence**: Select the example with the lowest maximum class probability.

$$x^* = \underset{x}{\arg\min} \, \max_c P(y = c \mid x)$$

**Margin sampling**: Select the example with the smallest gap between the top two predicted class probabilities. A small margin means the model is nearly indifferent between two classes — it would benefit most from knowing which is correct.

$$x^* = \underset{x}{\arg\min} \left( P(\hat{y}_1 \mid x) - P(\hat{y}_2 \mid x) \right)$$

**Entropy sampling**: Select the example with the highest predictive entropy — uncertainty spread across all classes.

$$x^* = \underset{x}{\arg\max} \, H[P(y \mid x)] = -\sum_c P(y = c \mid x) \log P(y = c \mid x)$$

**Limitation of uncertainty sampling**: Selects examples near the current decision boundary, which may be clustered in a small part of the input space. If these examples are outliers or noisy, the model may thrash rather than improve.

### Query by Committee (QBC)

Train a **committee** of models on the same labeled data using different algorithms or initializations. Each committee member votes on the label of each unlabeled example. Select the example where committee members **disagree most** — disagreement signals that the labeled data is insufficient to resolve the true class.

Disagreement is measured by:

- **Vote entropy**: Shannon entropy over the committee's votes.
- **Average KL divergence**: How far each committee member's distribution is from the average.

QBC is computationally expensive (requires training multiple models) but less susceptible to outlier selection than simple uncertainty sampling.

### Expected Model Change

Select the example that, if labeled and added to training, would cause the **greatest change** to the model parameters.

For gradient-based models, this corresponds to selecting the example with the largest expected gradient magnitude:

$$x^* = \underset{x}{\arg\max} \, \mathbb{E}_{y \sim P(\cdot \mid x)} \left[ \| \nabla_\theta \mathcal{L}(\theta; x, y) \| \right]$$

This is expensive to compute exactly — it requires evaluating all possible labels and their gradient contributions — but approximations exist.

### Expected Error Reduction

Select the example whose labeling would most reduce the model's **expected generalization error** on the remaining unlabeled pool.

$$x^* = \underset{x}{\arg\min} \, \mathbb{E}_{y \sim P(\cdot \mid x)} \left[ \text{Error}(\theta_{x,y}) \right]$$

This is the most principled strategy but computationally prohibitive for large models. Approximations using Fisher information matrices or Bayesian neural networks make it more tractable.

### Core-Set Selection

Rather than selecting based on model uncertainty, **core-set methods** select examples to ensure good geometric coverage of the input space. The goal is to find a labeled set $S$ such that the maximum distance from any unlabeled point to its nearest labeled neighbor is minimized:

$$x^* = \underset{x \in \mathcal{U}}{\arg\max} \, \min_{s \in \mathcal{S}} d(x, s)$$

This greedy algorithm (k-Center Greedy) builds a labeled set that represents the full diversity of the unlabeled pool — avoiding the clustering behavior of uncertainty-only methods.

### BADGE: Batch Active Learning by Diverse Gradient Embeddings

**BADGE** combines uncertainty and diversity:

1. Compute a gradient embedding for each unlabeled example (gradient of the loss with respect to the last layer's parameters, using the predicted pseudo-label).
2. Use **k-means++** initialization on these gradient embeddings to select a diverse batch.

BADGE naturally selects examples that are both uncertain (large gradient magnitude) and diverse (spread across gradient space). It achieves state-of-the-art performance on many benchmarks.

## Pool-Based, Stream-Based, and Membership Query Synthesis

Active learning comes in three settings:

**Pool-based active learning**: A large pool of unlabeled examples is available upfront. The learner selects from the pool. This is the most common setting in practice.

**Stream-based active learning**: Unlabeled examples arrive one at a time. For each example, the learner decides on the spot whether to query its label (paying a cost) or discard it. Useful for real-time systems where storing all data is impractical.

**Membership query synthesis**: The learner can generate arbitrary queries — not just selecting from existing unlabeled data, but constructing hypothetical inputs. Powerful in principle but generates queries that may be unnatural and difficult for human annotators to label reliably.

## Batch Mode Active Learning

Rather than selecting one example at a time and retraining after each label (which is computationally prohibitive for large models), **batch mode active learning** selects a batch of $b$ examples per round, labels all of them, retrains once, and repeats.

Naive batching — taking the top-$b$ most uncertain examples — tends to select highly similar, redundant examples clustered around the same ambiguous region. Effective batch strategies enforce **diversity** within each selected batch.

Common approaches:

- **Greedy submodular optimization**: Select a batch that maximizes a submodular objective combining uncertainty and coverage. Submodular functions capture the diminishing returns property: adding a new example to an already-large batch is less informative than adding it to a small batch.
- **Determinantal Point Processes (DPPs)**: Sample batches with high diversity and high quality using a probabilistic model based on kernel matrices.

## Active Learning with Deep Neural Networks

Classical active learning theory was developed for simple models (logistic regression, SVMs). Deep neural networks pose new challenges:

**Calibration**: Deep networks are often overconfident — their predicted probabilities do not accurately reflect true uncertainty. Uncertainty-based selection strategies fail when confidence scores are poorly calibrated. Solutions include:

- **MC Dropout**: Running inference with dropout enabled multiple times to approximate Bayesian uncertainty.
- **Deep ensembles**: Training multiple networks and measuring disagreement.
- **Temperature scaling**: Post-hoc calibration of softmax outputs.

**Cold start**: With very few initial labeled examples, a deep network may not have learned useful representations. Strategies:

- Use **self-supervised pretraining** to initialize good representations before the active learning loop begins.
- Start with diversity-based selection (core-set) rather than uncertainty-based selection.

**Computational cost**: Retraining a large model after every annotation round is expensive. Techniques like **warm-starting** (resuming from the previous checkpoint rather than training from scratch) and **continual learning** strategies reduce this overhead.

## Label Efficiency: What Active Learning Achieves

The benchmark for active learning is **label efficiency** — how much smaller the actively selected labeled set can be while achieving the same performance as random sampling.

Typical reported results across domains:

| Domain | Task | Labels saved |
|--------|------|-------------|
| Medical imaging | CT scan classification | 60–80% |
| NLP | Text classification | 40–70% |
| Computer vision | Object detection | 50–75% |
| Speech | Keyword spotting | 30–60% |

Results vary significantly by dataset and query strategy. Active learning provides the greatest benefit when labeled data is scarce relative to total data, and when the data distribution is heterogeneous.

## Annotation Interface Design

Active learning is only as good as the annotations it receives. Poor annotation quality — inconsistent labeling, annotator confusion — can actively harm model performance if the queried examples are systematically hard to label.

Best practices:

- **Design clear annotation guidelines** with examples for edge cases.
- **Show context**: Provide annotators with surrounding context (neighboring sentences, related images) for ambiguous examples.
- **Use redundancy strategically**: For high-uncertainty examples, obtain multiple labels and aggregate with majority vote or Dawid-Skene models.
- **Monitor annotator agreement**: Track inter-annotator agreement metrics (Cohen's Kappa, Krippendorff's Alpha) to detect annotator fatigue or ambiguous label definitions.

## Active Learning for NLP

**Named entity recognition (NER)** active learning selects sentences where the model has high token-level uncertainty, accounting for the sequential structure of the labeling task (labeling a sentence requires labeling every token in it).

**Text classification** active learning must account for **class imbalance** — uncertainty sampling will rarely select examples from rare classes since the model is typically already uncertain about them. Hybrid strategies combining uncertainty with class-balanced sampling address this.

**Fine-tuning large language models**: Active learning for LLM fine-tuning is an active research area. Selecting which training examples to include in a few-shot prompt (in-context active learning) has emerged as an effective way to improve few-shot performance.

## Practical Considerations

**When active learning is NOT worth it:**

- When labeling is cheap (e.g., crowdsourced tasks with clear criteria).
- When a large labeled dataset already exists.
- When model retraining is prohibitively expensive between rounds.
- When data collection can be accelerated through other means (programmatic labeling, weak supervision).

**When active learning provides the most value:**

- Expert annotation is expensive (medical, legal, scientific domains).
- A large unlabeled corpus exists.
- Model training is fast relative to labeling cost.
- The unlabeled data is diverse and the distribution is heterogeneous.

## Integration with Weak Supervision and Semi-Supervised Learning

Active learning is most powerful when combined with complementary approaches:

- **Semi-supervised learning**: Use unlabeled data to improve model representations, then apply active learning to select the most valuable examples to label.
- **Weak supervision (Snorkel)**: Generate noisy programmatic labels for the full dataset using labeling functions, then use active learning to identify and correct the most impactful errors.
- **Self-training**: Pseudo-label high-confidence unlabeled examples to augment the labeled set, reserving the human labeling budget for genuinely uncertain cases identified by active learning.

Active learning is a pragmatic strategy for the real-world constraint that data labeling is expensive. As AI applications expand into specialized domains with high annotation costs, active learning becomes an increasingly important tool in the practitioner's toolkit.
