---
title: Introduction to TRL
description: A practical guide to TRL (Transformer Reinforcement Learning), Hugging Face's library for fine-tuning language models with RLHF, DPO, PPO, and reward modeling.
---

# Introduction to TRL

**TRL** (Transformer Reinforcement Learning) is Hugging Face's open-source library for fine-tuning large language models with reinforcement learning from human feedback (RLHF) and related alignment techniques. It provides production-ready trainers for supervised fine-tuning (SFT), reward modeling, PPO, DPO, GRPO, and more — all built on top of `transformers` and `accelerate`.

## Why TRL?

Training aligned language models involves multiple stages:

1. **Supervised Fine-Tuning (SFT)**: teach the model to follow instructions
2. **Reward Modeling (RM)**: learn a scalar reward from human preference data
3. **RL Fine-Tuning (PPO / GRPO)**: optimize the policy against the reward model
4. **Direct Alignment (DPO / KTO)**: skip the RL loop and align directly from preferences

TRL implements all of these with a unified, trainer-based API compatible with any `transformers` model.

## Installation

```bash
pip install trl
# With optional extras
pip install "trl[peft]"          # LoRA/QLoRA support
pip install "trl[diffusers]"     # Diffusion model RLHF
```

## Supervised Fine-Tuning with SFTTrainer

The `SFTTrainer` wraps `Trainer` to handle chat template formatting, packing, and LoRA integration automatically.

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

model_id = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("trl-lib/tldr", split="train")

training_args = SFTConfig(
    output_dir="sft-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=50,
    save_steps=500,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model("sft-model")
```

### Using Chat Templates

For instruction datasets in conversational format:

```python
def format_chat(example):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_chat)
```

### QLoRA Fine-Tuning

```python
from peft import LoraConfig
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.train()
```

## Reward Modeling with RewardTrainer

A reward model is a sequence classifier that outputs a scalar given a prompt + response.

```python
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "sft-model", num_labels=1
)

# Dataset must have "chosen" and "rejected" columns
reward_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

reward_config = RewardConfig(
    output_dir="reward-model",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    gradient_checkpointing=True,
)

reward_trainer = RewardTrainer(
    model=reward_model,
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=reward_dataset,
)
reward_trainer.train()
```

## PPO Training with PPOTrainer

PPO fine-tunes the SFT model to maximize reward while maintaining proximity to the SFT policy via a KL penalty.

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import pipeline

ppo_config = PPOConfig(
    model_name="sft-model",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    ppo_epochs=4,
    kl_penalty="kl",
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained("sft-model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft-model")
reward_pipe = pipeline("text-classification", model="reward-model")

ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

for batch in ppo_trainer.dataloader:
    query_tensors = batch["input_ids"]

    # Generate responses
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=100)

    # Score with reward model
    texts = tokenizer.batch_decode(response_tensors)
    rewards = [torch.tensor(r["score"]) for r in reward_pipe(texts)]

    # PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
```

## Direct Preference Optimization (DPO)

DPO eliminates the RL loop by treating preference alignment as a classification problem on preference pairs.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="dpo-model",
    beta=0.1,                       # KL regularization strength
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=5e-7,
    fp16=True,
    loss_type="sigmoid",            # or "hinge", "ipo", "kto_pair"
)

# Dataset: {"prompt": ..., "chosen": ..., "rejected": ...}
dpo_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,   # automatically creates frozen reference copy
    tokenizer=tokenizer,
    args=dpo_config,
    train_dataset=dpo_dataset,
)
dpo_trainer.train()
```

## GRPO — Group Relative Policy Optimization

GRPO (used in DeepSeek-R1) eliminates the value/critic model by computing advantages from within-batch reward variation:

```python
from trl import GRPOTrainer, GRPOConfig

def reward_fn(completions, **kwargs):
    """Custom reward: prefer longer responses (toy example)."""
    return [float(len(c)) / 200.0 for c in completions]

grpo_config = GRPOConfig(
    output_dir="grpo-model",
    num_generations=8,   # G samples per prompt for relative reward
    max_new_tokens=256,
    learning_rate=5e-7,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=reward_fn,
    args=grpo_config,
    train_dataset=dataset,
)
trainer.train()
```

## Algorithm Comparison

| Algorithm | RM Required | Reference Model | Online | Best For |
|---|---|---|---|---|
| SFT | ❌ | ❌ | ❌ | Instruction following |
| PPO | ✅ | ✅ | ✅ | Complex reward shaping |
| DPO | ❌ | ✅ | ❌ | Preference alignment |
| GRPO | ❌ | ✅ | ✅ | Reasoning, math |
| KTO | ❌ | ✅ | ❌ | Unpaired feedback |
| ORPO | ❌ | ❌ | ❌ | Combined SFT + alignment |

## Key Features

### vLLM Integration for Fast Generation

```python
grpo_config = GRPOConfig(
    use_vllm=True,           # use vLLM for generation
    vllm_device="cuda:1",    # separate GPU for inference
    vllm_gpu_memory_utilization=0.5,
)
```

### Weights & Biases / TensorBoard Logging

TRL integrates with Hugging Face `Trainer` reporting — set `report_to="wandb"` in any config.

### Multi-GPU and DeepSpeed

```bash
accelerate launch --config_file deepspeed_z3.yaml train_dpo.py
```

## Best Practices

- **SFT before alignment**: always start with instruction fine-tuning before DPO/PPO
- **Small $\beta$ for DPO**: `beta=0.01`–`0.1` — larger values make alignment weaker
- **Reference model matters**: for DPO, the reference should be the SFT checkpoint, not the base model
- **KL monitoring**: watch `train/kl` in PPO — divergence > 20 signals instability
- **Reward hacking**: PPO models can exploit reward model weaknesses — regularly evaluate on held-out human preferences
- **Gradient checkpointing**: always enable for models >7B parameters

## Summary

TRL provides a complete, modular toolkit for aligning language models with human preferences. Starting from supervised fine-tuning through `SFTTrainer`, to reward modeling, PPO, DPO, and GRPO, the library handles distributed training, LoRA, vLLM generation, and evaluation out of the box. Its clean API and tight `transformers` integration make it the de facto standard for RLHF research and production alignment pipelines.
