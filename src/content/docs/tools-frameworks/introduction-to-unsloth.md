---
title: Introduction to Unsloth
description: A practical guide to Unsloth, an open-source library for 2–5x faster LLM fine-tuning with significantly reduced VRAM usage through kernel-level optimizations.
---

# Introduction to Unsloth

**Unsloth** is an open-source library that dramatically accelerates fine-tuning of large language models — delivering **2–5× speed improvements** and **40–70% VRAM reductions** compared to baseline Hugging Face + PEFT pipelines. It achieves this through hand-written CUDA/Triton kernels that fuse operations, eliminate redundant memory transfers, and apply mathematical reformulations unavailable in general-purpose frameworks. Unsloth is fully compatible with `transformers`, `trl`, and `peft`, making it a drop-in optimization layer for existing fine-tuning workflows.

## Why Unsloth?

Standard fine-tuning pipelines leave significant performance on the table:

- **Redundant memory copies**: intermediate activations stored naively for backward pass
- **Non-fused operations**: LayerNorm, RoPE, attention computed in separate CUDA kernels with memory round-trips
- **Generic autograd**: PyTorch autograd tracks gradients for operations that could be computed more efficiently with custom backward passes

Unsloth addresses each of these with specialized kernels, enabling training of 70B-parameter models on single consumer GPUs.

## Installation

```bash
# Install with pip (CUDA 12.1)
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Or for CUDA 12.4 / PyTorch 2.4
pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Check GPU compatibility
python -c "import unsloth; print(unsloth.__version__)"
```

## Quick Start: SFT with Llama 3

```python
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=4096,
    dtype=None,               # auto-detect: bfloat16 on Ampere+, float16 on older
    load_in_4bit=True,        # QLoRA — saves ~70% VRAM
)

# Apply LoRA adapters (Unsloth's optimized version)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                     # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,         # Dropout=0 is faster; Unsloth supports non-zero too
    bias="none",
    use_gradient_checkpointing="unsloth",   # Unsloth's memory-efficient checkpointing
    random_state=42,
)

print(model.print_trainable_parameters())
```

## Training with TRL's SFTTrainer

Unsloth integrates seamlessly with TRL:

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("trl-lib/tldr", split="train")

def format_prompt(example):
    return {"text": f"Summarize: {example['prompt']}\n\n{example['completion']}"}

dataset = dataset.map(format_prompt)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="unsloth-llama3",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",           # 8-bit Adam reduces optimizer memory
        lr_scheduler_type="cosine",
        seed=42,
        dataset_text_field="text",
        max_seq_length=4096,
        packing=True,                 # pack short sequences for efficiency
    ),
)
trainer.train()
```

## Chat Template Support

Unsloth includes helpers for common chat formats:

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",   # or "chatml", "mistral", "gemma", "phi-3"
)

def format_conversations(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in convos
    ]
    return {"text": texts}

dataset = dataset.map(format_conversations, batched=True)
```

## Saving and Exporting Models

### Save LoRA Adapters

```python
# Save only the LoRA delta weights (small — typically <1 GB)
model.save_pretrained("lora-adapters")
tokenizer.save_pretrained("lora-adapters")
```

### Merge and Export to GGUF (for llama.cpp)

```python
# Merge adapters into base model and quantize to GGUF
model.save_pretrained_gguf(
    "model-gguf",
    tokenizer,
    quantization_method="q4_k_m",   # recommended for quality/size balance
)
# Outputs: model-gguf/unsloth.Q4_K_M.gguf — ready for llama.cpp / Ollama
```

### Push to Hugging Face Hub

```python
model.push_to_hub_gguf(
    "username/my-fine-tuned-model",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0", "f16"],
    token="hf_...",
)
```

## DPO Fine-Tuning

```python
from trl import DPOTrainer, DPOConfig

# Convert model for DPO (enables gradient computation on reference too)
model = FastLanguageModel.get_peft_model(model, r=32, ...)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,          # Unsloth handles frozen reference internally
    tokenizer=tokenizer,
    train_dataset=preference_dataset,
    args=DPOConfig(
        output_dir="dpo-model",
        beta=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-7,
        optim="adamw_8bit",
    ),
)
dpo_trainer.train()
```

## Supported Models

| Model Family | Base Sizes | Context |
|---|---|---|
| Llama 3.1 / 3.2 / 3.3 | 1B, 3B, 8B, 70B | Up to 128K |
| Mistral / Mixtral | 7B, 8×7B | 32K |
| Gemma 2 | 2B, 9B, 27B | 8K |
| Phi-3 / Phi-4 | 3.8B, 14B | 128K |
| Qwen 2.5 | 0.5B–72B | 128K |
| DeepSeek-R1 | 1.5B–70B | 128K |

## Performance Benchmarks

On an A100 80GB GPU fine-tuning Llama 3.1 8B with QLoRA:

| Setting | Tokens/sec | VRAM (GB) |
|---|---|---|
| Baseline HF + PEFT | 1,850 | 42 |
| + Flash Attention 2 | 2,100 | 38 |
| Unsloth (this library) | 4,700 | 24 |
| Unsloth + packing | 5,900 | 22 |

## Key Optimizations Under the Hood

### RoPE Embedding Fusion

Rotary Position Embeddings are applied inside a single fused CUDA kernel alongside the QK projection, eliminating a separate memory round-trip for position encoding.

### Layernorm Backward Reformulation

Unsloth derives a custom backward pass for RMSNorm that avoids storing the full intermediate normalized tensor — saving activations memory proportional to batch size × sequence length × hidden dim.

### Gradient Checkpointing (Unsloth variant)

Standard gradient checkpointing recomputes all activations on backward pass. Unsloth's variant selectively checkpoints only the most memory-intensive layers (attention) while caching lighter operations, providing a better speed/memory trade-off.

### 8-bit Optimizer

`adamw_8bit` from `bitsandbytes` quantizes optimizer state (momentum, variance) to 8-bit, cutting optimizer memory by 4× with negligible accuracy loss.

## Recommended Configurations

```python
# Rank selection guidelines
# r=8:  small dataset (<10k), simple task
# r=16: general-purpose (recommended starting point)
# r=32: complex tasks, domain adaptation
# r=64+: full fine-tuning approximation (high VRAM)

# alpha = r * 2 is a common heuristic
# target_modules: include all projection layers for best results
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)
```

## Comparison with Alternatives

| Tool | Speed vs Baseline | VRAM Savings | Ease of Use | GGUF Export |
|---|---|---|---|---|
| Unsloth | 2–5× | 40–70% | Easy | ✅ |
| Axolotl | 1.2–1.5× | 20–30% | Moderate | Partial |
| LLaMA-Factory | 1.3–1.8× | 25–40% | Easy | ✅ |
| torchtune | 1.1–1.4× | 15–25% | Moderate | ❌ |

## Summary

Unsloth makes fine-tuning frontier language models accessible on consumer and prosumer hardware by combining hand-crafted CUDA kernels, memory-efficient gradient checkpointing, 8-bit optimizer states, and QLoRA. Its TRL/PEFT compatibility means existing workflows require minimal changes — often just replacing model loading with `FastLanguageModel.from_pretrained`. The native GGUF export pipeline closes the gap between fine-tuning and local deployment via llama.cpp or Ollama, making Unsloth a compelling choice for the complete fine-tuning-to-deployment cycle.
