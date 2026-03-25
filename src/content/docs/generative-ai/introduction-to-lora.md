---
title: Introduction to LoRA
description: An introduction to Low-Rank Adaptation (LoRA) for efficient LLM fine-tuning.
---

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning (PEFT) technique for fine-tuning large language models (LLMs).

## How LoRA Works

LoRA freezes the pre-trained model weights and injects trainable rank-decomposition matrices into each layer of the Transformer architecture.

1. **Freeze:** Keep the original pre-trained LLM weights.
2. **Inject:** Add small, trainable "adapter" layers with fewer parameters.
3. **Train:** Update only the adapter weights, significantly reducing training time.

## Why Use LoRA?

- **Low VRAM:** Fine-tune models on consumer-grade hardware.
- **Portability:** Fast switching between different fine-tuned models by swapping tiny adapter weight files.
- **No Training Loss:** LoRA doesn't suffer from catastrophic forgetting like traditional fine-tuning.

## Applications of LoRA

- **Domain Adaptation:** Fine-tuning LLMs for medical, legal, or code generation.
- **Style Injection:** Adapting LLMs to a specific writing style or brand tone.
- **Personalized AI:** Building individual-user specialized assistants.
