---
title: Self-Supervised Representation Learning - Learning Without Labels
description: See how models learn useful representations from unlabeled data by solving automatically generated pretext tasks.
---

Self-supervised learning trains a model on a task where labels are generated automatically from the data itself, rather than annotated by humans. The goal is not the pretext task itself but the reusable representation learned along the way.

```text
raw data -> automatically constructed task -> learned representation -> fine-tune on real task
```

## Pretext Tasks

Early self-supervised vision methods used tasks like predicting the rotation applied to an image, solving a jigsaw puzzle made from image patches, or colorizing a grayscale photo. Solving these tasks well requires the model to understand object shape, texture, and context—useful general-purpose features—even though none of it directly requires human labels.

## Contrastive Learning

A dominant modern approach, contrastive learning, trains a model to pull representations of augmented views of the same input closer together while pushing representations of different inputs apart:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j} \exp(\text{sim}(z_i, z_j) / \tau)}$$

Here `z_i` and `z_i^+` are embeddings of two augmented views of the same image, and the sum in the denominator runs over all other examples in the batch, treated as negatives. `τ` is a temperature parameter controlling how sharply the model separates positives from negatives. SimCLR and MoCo are well-known implementations of this idea.

## Masked Prediction

An alternative family masks part of the input and trains the model to reconstruct or predict it—BERT's masked language modeling in NLP, and masked autoencoders (MAE) in vision, which reconstruct missing image patches from the visible ones. This approach avoids the need for carefully designed augmentations and scales well with large unlabeled datasets.

## Why It Matters

Self-supervised pretraining lets models leverage the vast amount of unlabeled data available (raw text, images, audio) before fine-tuning on a small labeled dataset for a specific task. It has become the standard first stage for nearly all large-scale foundation models, since collecting labels at the scale needed for strong performance is far more expensive than collecting raw unlabeled data. The main risk is that pretext tasks can encourage the model to latch onto shortcuts (e.g., color statistics from a specific augmentation) that don't transfer to the downstream task, so evaluating on real downstream performance remains essential.
