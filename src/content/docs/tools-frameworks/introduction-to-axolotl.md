---
title: Introduction to Axolotl
description: Learn how Axolotl simplifies fine-tuning large language models with a YAML-driven configuration system, supporting LoRA, QLoRA, full fine-tuning, multi-GPU training with DeepSpeed and FSDP, and a wide variety of dataset formats out of the box.
---

**Axolotl** is an open-source fine-tuning framework built on top of the Hugging Face ecosystem (Transformers, PEFT, datasets) that dramatically simplifies the process of fine-tuning large language models. Where raw Hugging Face Trainer code requires hundreds of lines of boilerplate for a typical LoRA fine-tuning run, Axolotl reduces the same configuration to a single YAML file — handling dataset preprocessing, tokenization, model loading, training loop configuration, and multi-GPU distribution automatically.

Axolotl is particularly popular in the open-source LLM community for its breadth of supported model architectures, dataset formats, and training techniques — enabling researchers and practitioners to focus on experiments rather than engineering plumbing.

## Why Axolotl?

Fine-tuning LLMs from scratch involves assembling many interconnected components:

- Loading and quantizing the base model correctly (4-bit, 8-bit, bfloat16).
- Configuring LoRA/QLoRA adapters with appropriate rank and target modules.
- Preprocessing diverse dataset formats (alpaca, sharegpt, completion, instruction).
- Setting up gradient checkpointing, flash attention, and memory optimizations.
- Configuring distributed training across multiple GPUs with DeepSpeed or FSDP.
- Monitoring training with wandb or MLflow.

Axolotl integrates and configures all of these components through a single YAML file, eliminating the integration burden while providing sensible defaults and clear documentation for each option.

## Installation

```bash
pip install axolotl

# Or for development with all extras
pip install axolotl[flash-attn,deepspeed]

# Or from source
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e '.[flash-attn,deepspeed]'
```

## Core Concepts: The YAML Configuration

Every Axolotl fine-tuning run is controlled by a YAML configuration file. Here's a complete example for QLoRA fine-tuning:

```yaml
# model
base_model: meta-llama/Meta-Llama-3-8B
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

# Quantization
load_in_4bit: true
bnb_4bit_use_double_quant: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16

# LoRA adapter configuration
adapter: qlora
lora_r: 16                     # Rank
lora_alpha: 32                 # Scaling factor (alpha/r = 2 recommended)
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Datasets
datasets:
  - path: tatsu-lab/alpaca     # Hugging Face dataset
    type: alpaca
  - path: ./my_custom_data.jsonl  # Local file
    type: sharegpt
    conversation: chatml

# Sequence length
sequence_len: 4096
sample_packing: true           # Pack multiple short examples into one sequence

# Training
output_dir: ./outputs/llama3-qlora
num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 0.0002
optimizer: adamw_bnb_8bit      # 8-bit optimizer for memory efficiency
lr_scheduler: cosine
warmup_ratio: 0.05

# Memory optimizations
gradient_checkpointing: true
flash_attention: true

# Logging
logging_steps: 10
eval_steps: 200
save_steps: 500
wandb_project: my-fine-tune
wandb_run_id: llama3-qlora-run-1
```

Run fine-tuning with:

```bash
axolotl train config.yaml

# Multi-GPU with accelerate
accelerate launch -m axolotl.cli.train config.yaml

# With DeepSpeed ZeRO-2
accelerate launch -m axolotl.cli.train config.yaml \
    --deepspeed deepspeed_configs/zero2.json
```

## Supported Dataset Formats

Axolotl handles the preprocessing for many common dataset formats out of the box:

### Alpaca Format

```json
{
  "instruction": "Translate the following sentence to French.",
  "input": "The weather is beautiful today.",
  "output": "Le temps est magnifique aujourd'hui."
}
```

### ShareGPT / ChatML Format

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What is 2 + 2?"},
    {"from": "gpt", "value": "4"}
  ]
}
```

### Completion Format (Raw Text)

```json
{"text": "Here is a complete text that will be trained on as-is, without any prompt formatting."}
```

### Custom Prompt Templates

For formats not natively supported, Axolotl allows defining custom prompt templates in YAML:

```yaml
datasets:
  - path: my_dataset
    type: input_output
    field_instruction: question
    field_output: answer
```

## Training Techniques

### Full Fine-Tuning

Training all model parameters — appropriate for smaller models or when maximum task performance is needed:

```yaml
adapter: null  # No adapter = full fine-tuning
load_in_4bit: false
bf16: true
```

### LoRA (Low-Rank Adaptation)

Train only the adapter parameters, keeping the base model frozen:

```yaml
adapter: lora
lora_r: 64
lora_alpha: 128
lora_target_modules:
  - q_proj
  - v_proj
```

### QLoRA (Quantized LoRA)

The most memory-efficient option — 4-bit quantization with LoRA adapters:

```yaml
adapter: qlora
load_in_4bit: true
bnb_4bit_quant_type: nf4
lora_r: 16
lora_alpha: 32
```

QLoRA enables fine-tuning 70B+ models on a single 80GB A100 GPU — reducing the hardware requirements for large model fine-tuning from multiple high-end GPUs to a single machine.

## Sample Packing

**Sample packing** is a key memory efficiency feature: instead of padding short sequences to the maximum length and wasting compute on padding tokens, Axolotl packs multiple training examples into a single sequence up to `sequence_len`, with attention masking ensuring examples don't attend to each other:

```yaml
sequence_len: 8192
sample_packing: true
pad_to_sequence_len: true
```

Sample packing can 2-4x increase effective batch size for datasets with variable-length short sequences, significantly accelerating training.

## Multi-GPU Training with DeepSpeed

For large models or high-throughput training, Axolotl integrates with DeepSpeed ZeRO:

```yaml
# In config.yaml
deepspeed: deepspeed_configs/zero2.json
```

```json
// zero2.json — splits optimizer states and gradients across GPUs
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true
  },
  "fp16": {"enabled": "auto"},
  "bf16": {"enabled": "auto"},
  "gradient_clipping": 1.0
}
```

ZeRO Stage 2 distributes optimizer states and gradients across GPUs — enabling training of larger models on the same hardware. ZeRO Stage 3 additionally shards model parameters.

## Evaluation and Inference

After training, Axolotl provides utilities for inference and evaluation:

```bash
# Merge LoRA adapter into base model weights
axolotl merge-lora config.yaml --lora-model-dir ./outputs/checkpoint-1000

# Inference with the fine-tuned model
axolotl inference config.yaml \
    --lora-model-dir ./outputs/checkpoint-1000 \
    --prompter alpaca \
    --prompt "What is the capital of France?"
```

```python
# Or use the merged model directly with Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "./outputs/merged",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./outputs/merged")

inputs = tokenizer("What is the capital of France?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Supported Base Models

Axolotl supports any model in the Hugging Face Transformers ecosystem:

| Family | Example Models |
| --- | --- |
| LLaMA / Llama 2 / Llama 3 | meta-llama/Meta-Llama-3-8B |
| Mistral / Mixtral | mistralai/Mistral-7B-v0.1 |
| Qwen | Qwen/Qwen2-7B |
| Gemma / Gemma 2 | google/gemma-2-9b |
| Falcon | tiiuae/falcon-7b |
| Phi | microsoft/phi-3-mini-4k-instruct |
| Yi | 01-ai/Yi-34B |

Any model with a supported architecture can be fine-tuned by specifying the `base_model` and `model_type` fields in the YAML.

## Comparison with Alternatives

| Feature | Axolotl | LLaMA-Factory | Unsloth | Raw Hugging Face |
| --- | --- | --- | --- | --- |
| YAML configuration | Yes | Yes | No | No |
| QLoRA support | Yes | Yes | Yes | Yes |
| Dataset format handling | 15+ formats | 10+ formats | Limited | Manual |
| DeepSpeed integration | Yes | Yes | No | Manual |
| Sample packing | Yes | No | Yes | No |
| Learning curve | Low | Low | Medium | High |

Axolotl strikes a balance between ease of use and flexibility — suitable for both quick experiments (change a few YAML lines) and production fine-tuning pipelines (full DeepSpeed multi-GPU runs on custom datasets).
