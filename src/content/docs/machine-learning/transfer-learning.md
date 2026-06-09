---
title: Transfer Learning
description: How transfer learning allows models trained on one task to be adapted for another, dramatically reducing data and compute requirements.
---

Transfer learning is a technique where a model trained on one task is reused as the starting point for a model on a different but related task. Rather than training from scratch, you leverage representations already learned from large datasets, saving enormous amounts of time, data, and compute.

## Why Transfer Learning Works

Deep neural networks trained on large datasets learn hierarchical representations. In computer vision, early layers detect edges and textures; later layers detect shapes and objects. These general features are useful across many visual tasks, not just the original one. Transfer learning exploits this: keep the learned features, adapt only what needs to change for the new task.

## The Two Main Approaches

### Feature Extraction
Freeze all the pretrained model's weights and use it as a fixed feature extractor. Only a new classifier head added on top is trained. Fast and effective when your dataset is small and similar to the original training data.

### Fine-Tuning
Start from pretrained weights and continue training the entire model (or the last few layers) on the new task. More powerful — allows the model to adapt its representations to the new domain. Requires more data and compute than feature extraction, but less than training from scratch.

## In Natural Language Processing

Transfer learning transformed NLP with models like BERT, GPT, and RoBERTa. Pretrained on massive text corpora (predicting masked words or next tokens), they learn rich language representations. Fine-tuning on downstream tasks — sentiment analysis, question answering, named entity recognition — typically requires only a small labeled dataset and a few epochs.

The pretraining → fine-tuning paradigm is now standard across all of NLP.

## In Computer Vision

ImageNet-pretrained models (ResNet, EfficientNet, ViT) serve as backbone features for virtually every vision task: object detection, medical image segmentation, satellite imagery analysis. Models like CLIP, pretrained on image-text pairs, transfer exceptionally well to novel visual tasks with zero or few examples.

## Domain Adaptation

A related concept: when the source and target domains differ significantly (e.g., training on natural photos, deploying on medical scans), naive fine-tuning may underperform. Domain adaptation techniques explicitly minimize the distribution gap between source and target, often using adversarial training or domain-invariant representations.

## When Transfer Learning Helps Most

- **Small labeled datasets:** A few hundred to a few thousand examples is often enough when starting from a strong pretrained model.
- **Limited compute:** Training from scratch is expensive; fine-tuning can be done on a single GPU in hours.
- **Related domains:** The more similar the source and target task, the more transferable the learned representations.

## Practical Tips

- Choose a pretrained model trained on data similar to your domain.
- Use a lower learning rate for fine-tuning than for training from scratch (e.g., 1e-5 instead of 1e-3).
- Fine-tune the last few layers first; unfreeze more layers if performance plateaus.
- Watch for catastrophic forgetting — the model losing pretrained knowledge when fine-tuned aggressively.
