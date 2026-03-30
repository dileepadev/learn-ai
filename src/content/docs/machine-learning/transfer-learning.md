---
title: Transfer Learning
description: How pre-trained models are adapted to new tasks with minimal data and compute.
---

Transfer learning is a technique where a model trained on one task is reused as the starting point for a model on a different but related task. It is one of the most practical and widely adopted approaches in modern machine learning.

## The Core Idea

Training a large model from scratch requires vast amounts of labelled data, significant compute, and considerable time. Transfer learning sidesteps this by reusing knowledge a model has already acquired.

A model trained on a large general-purpose dataset — such as ImageNet for images or a large text corpus for language — develops feature representations that generalise well. These representations can be transferred to narrower, domain-specific tasks with far less data.

## How It Works

The general workflow for transfer learning has two phases:

1. **Pre-training:** A base model is trained on a large dataset using a general objective (e.g., image classification across 1,000 categories or predicting masked tokens in text).
2. **Fine-tuning:** The pre-trained model is then trained further on a smaller, task-specific dataset, either by updating all weights or only part of the network.

### Variants

- **Feature extraction:** The pre-trained model's weights are frozen. Only a new output layer is trained on top.
- **Full fine-tuning:** All weights are updated during the second training phase, giving more flexibility but requiring more data.
- **Partial fine-tuning:** Only the later (task-specific) layers are updated while early layers remain frozen.

## Why It Works

Deep neural networks learn hierarchical representations. Early layers tend to capture general low-level features (edges, textures, basic syntax) that are useful across many tasks. Later layers capture higher-level, task-specific patterns. By keeping the general layers and replacing or adapting task-specific ones, a model can perform well on a new task without starting from scratch.

## Applications

- **Computer Vision:** Fine-tuning ResNet or EfficientNet for medical image classification, defect detection, or satellite imagery analysis.
- **Natural Language Processing:** Adapting BERT, RoBERTa, or GPT-series models for sentiment analysis, named entity recognition, or question answering.
- **Speech Recognition:** Adapting Whisper or wav2vec models for low-resource languages.
- **Code Generation:** Fine-tuning code-focused models for proprietary APIs or specific programming frameworks.

## Benefits

- **Less Data Required:** A few hundred to a few thousand labelled examples may be sufficient when fine-tuning a strong base model.
- **Faster Training:** Pre-trained weights serve as a warm start, drastically reducing training time and compute costs.
- **Better Generalisation:** Models trained on broad tasks often generalise more robustly than models trained on small domain-specific datasets alone.

## Challenges

- **Domain Gap:** If the pre-training domain is very different from the target domain, transfer may be limited or require more data to bridge the gap.
- **Negative Transfer:** In rare cases, pre-trained knowledge can hurt performance on the target task if the tasks are sufficiently dissimilar.
- **Catastrophic Forgetting:** Full fine-tuning can overwrite general knowledge learned during pre-training, especially when the fine-tuning dataset is small.

## Relationship to Foundation Models

Modern foundation models — large pre-trained models such as GPT-4, Claude, LLaMA, and CLIP — are designed explicitly for transfer. They act as universal starting points that can be adapted to thousands of downstream tasks. Techniques like LoRA and adapter layers have made this process even more parameter-efficient.

## Summary

Transfer learning has shifted the default approach in both NLP and computer vision from training specialist models from scratch to adapting general-purpose pre-trained models. It lowers the barrier to building high-quality AI systems and is a foundational skill for any machine learning practitioner.
