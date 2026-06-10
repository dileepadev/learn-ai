---
title: Transfer Learning
description: How transfer learning allows models trained on one task to be adapted for another, reducing data and compute requirements.
---

Transfer learning is a technique where a model pretrained on a large dataset is reused as the starting point for a different but related task. Instead of training from scratch, you leverage representations the model already learned, saving significant data, time, and compute.

## How It Works

Deep neural networks learn hierarchical features. Early layers capture generic patterns (edges, textures in vision; syntax in NLP) while later layers capture task-specific features. Transfer learning keeps the generic layers and replaces or fine-tunes only the task-specific parts.

**Two main strategies:**

- **Feature extraction:** Freeze all pretrained weights. Train only a new head added on top. Fast and effective for small datasets.
- **Fine-tuning:** Continue training some or all of the pretrained weights on the new task. More powerful but requires more data and careful tuning.

## In NLP

BERT, GPT, and RoBERTa are pretrained on massive text corpora, then fine-tuned on downstream tasks (sentiment analysis, NER, QA) with small labeled datasets. This pretraining → fine-tuning paradigm is now standard across all of NLP.

## In Computer Vision

ImageNet-pretrained models (ResNet, EfficientNet, ViT) are used as backbones for object detection, medical imaging, and satellite analysis. CLIP's image-text pretraining transfers exceptionally well to novel visual tasks with few or zero examples.

## When It Helps Most

- You have a small labeled dataset (hundreds to thousands of examples).
- Compute is limited — fine-tuning is far cheaper than training from scratch.
- The source and target domains are related.

## Practical Tips

- Use a lower learning rate when fine-tuning (e.g., 1e-5 vs 1e-3).
- Fine-tune later layers first; unfreeze earlier layers if performance plateaus.
- Watch for catastrophic forgetting when fine-tuning aggressively on small data.
