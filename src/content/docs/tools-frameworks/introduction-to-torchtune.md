---
title: Introduction to Torchtune
description: Learn PyTorch's native library for authoring, fine-tuning, and evaluating LLMs with recipe-based workflows, memory-efficient LoRA, QLoRA, and FSDP integration.
---

Fine-tuning modern Large Language Models (LLMs) often forces engineers to choose between two extremes: complex monolithic distributed codebases or rigid high-level wrappers that obscure the underlying PyTorch abstractions.

**Torchtune** is PyTorch's official, native library dedicated to authoring, fine-tuning, and experimenting with LLMs. Developed by Meta and the core PyTorch team, Torchtune adheres strictly to the **PyTorch philosophy**: clean, hackable, modular code blocks with minimal abstractions, zero bloat, and full native support for **PyTorch 2.x features** (including `torch.compile`, Fully Sharded Data Parallelism - FSDP2, and 4-bit/8-bit quantization through `torchao`).

---

## Architectural Principles of Torchtune

Torchtune was designed around four core tenets:

1. **PyTorch-Native:** No complex abstraction layers. Every recipe is transparent Python code reading and writing standard PyTorch tensors and modules.
2. **Recipe-Based Workflows:** Workflows are structured as self-contained **Recipes** that couple a specific model architecture with an optimization technique (e.g., LoRA on LLaMA-3 with FSDP).
3. **Memory Efficiency by Default:** Out-of-the-box integration with `torchao` for 4-bit NF4 quantization (QLoRA) and activation checkpointing, enabling fine-tuning of 8B parameter models on consumer GPUs (e.g., 24GB RTX 3090 / 4090).
4. **Composability:** Every layer—tokenizers, dataset formatters, attention modules, and loss functions—can be imported independently and integrated into custom training scripts.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Torchtune Architecture                                                      │
│                                                                             │
│  ┌───────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │ CLI & Config Layer    │  │ Recipes              │  │ Evaluators       │  │
│  │ YAML-based parameters │  │ Full Finetune, LoRA, │  │ EleutherAI eval  │  │
│  │ tune run / tune cp    │  │ QLoRA, DPO, PPO      │  │ harness integration││
│  └───────────────────────┘  └──────────────────────┘  └──────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Composable Building Blocks (Native PyTorch)                           │  │
│  │ • Models: LLaMA 3, Gemma, Mistral, Phi-3                             │  │
│  │ • Optimizers & Schedules: AdamW, Cosine, Fused CUDA                   │  │
│  │ • Distributed: torch.distributed (FSDP2, Activation Checkpointing)   │  │
│  │ • Quantization: torchao (NF4, INT8 weight-only, FP8)                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Torchtune Workflow

Torchtune organizes the complete lifecycle into three streamlined steps:

```
[ Step 1: Download ] ──► tune download meta-llama/Meta-Llama-3.1-8B-Instruct
                                │
                                ▼
[ Step 2: Recipe Config ] ──► tune cp llama3_1/8B_lora_single_device custom_config.yaml
                                │
                                ▼
[ Step 3: Run Training ] ──► tune run lora_finetune_single_device --config custom_config.yaml
```

---

## Hands-On Walkthrough: QLoRA Fine-Tuning

### Installation

Torchtune requires PyTorch 2.2 or higher:

```bash
pip install torchtune
# Optional: Install torchao for memory-efficient 4-bit and 8-bit quantization
pip install torchao
```

### 1. Inspecting Built-In Recipes

Torchtune ships with curated, battle-tested recipes across single-device and distributed multi-GPU environments:

```bash
tune ls
```

Output includes:
- `full_finetune_single_device`: Full-parameter training on a single GPU.
- `full_finetune_distributed`: Multi-GPU training utilizing PyTorch FSDP2.
- `lora_finetune_single_device`: Parameter-efficient LoRA on a single GPU.
- `lora_dpo_distributed`: Direct Preference Optimization using LoRA across nodes.

### 2. Copying and Customizing a Configuration

Copy a predefined YAML configuration to your local working directory:

```bash
tune cp llama3_1/8B_lora_single_device ./my_llama3_lora.yaml
```

Inspect and modify `./my_llama3_lora.yaml`:

```yaml
# Model Configuration
model:
  _component_: torchtune.models.llama3_1.lora_llama3_1_8b
  lora_attn_modules: ['q_proj', 'v_proj']
  apply_lora_to_mlp: True
  lora_rank: 16
  lora_alpha: 32

# Tokenizer Configuration
tokenizer:
  _component_: torchtune.models.llama3_1.llama3_1_tokenizer
  path: /tmp/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model

# Dataset Configuration
dataset:
  _component_: torchtune.datasets.instruct_dataset
  source: yahma/alpaca-cleaned
  split: train

# Training Hyperparameters
epochs: 3
batch_size: 2
gradient_accumulation_steps: 8
optimizer:
  _component_: torch.optim.AdamW
  lr: 2e-4
  fused: True

# Memory Optimizations
enable_activation_checkpointing: True
compile: True # Leverage torch.compile for graph fusion
```

### 3. Executing Training

Run training directly from the terminal. Torchtune will validate all components, download the dataset, wrap target linear layers with LoRA adapters, apply activation checkpointing, and stream loss metrics:

```bash
tune run lora_finetune_single_device --config ./my_llama3_lora.yaml
```

---

## Writing Custom Workflows with Torchtune Components

Because Torchtune is pure PyTorch, developers can import its modules directly into custom Python scripts without using the CLI:

```python
import torch
from torchtune.models.llama3_1 import lora_llama3_1_8b
from torchtune.modules.peft import get_adapter_params, set_trainable_params

# Instantiate native PyTorch model with LoRA adapters injected
model = lora_llama3_1_8b(
    lora_attn_modules=["q_proj", "v_proj"],
    apply_lora_to_mlp=True,
    lora_rank=8,
    lora_alpha=16
)

# Freeze base model weights and activate adapter gradients only
set_trainable_params(model, adapter_params_only=True)

# Verify trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
```

---

## Evaluating Checkpoints with EleutherAI lm-eval

Evaluating fine-tuned models on standard benchmarks (e.g., MMLU, GSM8K, ARC) is supported natively via the EleutherAI Evaluation Harness integration:

```bash
tune run eleuther_evaluation \
    --config eleuther_evaluation \
    model._component_=torchtune.models.llama3_1.llama3_1_8b \
    checkpointer.checkpoint_files=[./lora_output/checkpoint_epoch_3.pt] \
    tasks=[gsm8k,arc_challenge] \
    batch_size=8
```

---

## Why Choose Torchtune?

| Dimension | Torchtune | Hugging Face TRL / PEFT | Megatron-LM |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | Pure native PyTorch | High-level Trainer abstraction | Heavy distributed framework |
| **Debugging Experience** | Standard Python breakpoints & stack traces | Complex nested wrapper classes | Complex multi-node MPI/CUDA |
| **PyTorch 2.x Integration**| Day-0 native support (FSDP2, `torch.compile`)| Good (via Accelerate) | Custom implementation |
| **Quantization** | Native `torchao` (NF4, INT8) | `bitsandbytes` | Specialized FP8 |
| **Evaluation** | Integrated `lm-evaluation-harness` | Separate workflow | Custom benchmarking |

---

## Key Takeaways

- Torchtune brings the simplicity, transparency, and hackability of native PyTorch to LLM fine-tuning.
- Recipe-based configurations streamline repeatable experiments while allowing full customization of Python source code.
- Out-of-the-box integration with `torchao`, activation checkpointing, and `torch.compile` makes fine-tuning 8B models accessible on consumer GPUs.
