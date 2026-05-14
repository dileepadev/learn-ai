---
title: "Introduction to TRL: Transformer Reinforcement Learning"
description: "Learn how to use TRL and TRLX for fine-tuning LLMs with RLHF, DPO, and other reinforcement learning techniques from the Hugging Face ecosystem."
---

TRL (Transformer Reinforcement Learning) is Hugging Face's library for fine-tuning large language models with reinforcement learning. It provides components for the full RLHF pipeline and simpler alignment techniques like DPO.

## Why Use TRL?

TRL simplifies the complex RLHF pipeline:
- **PPO Trainer**: Full RLHF with Proximal Policy Optimization.
- **DPOTrainer**: Direct Preference Optimization without the complexity of PPO.
- **CPO Trainer**: Combined SFT and preference optimization.
- **Reward Trainer**: Train reward models from preference data.

## Installation

```bash
pip install trl[chatbot]  # For chatbot fine-tuning
pip install trl           # Core library
pip install trl[benchmarking]  # For evaluation
```

## Reward Model Training

First, train a reward model from preference data:

```python
from trl import RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels=1,  # Reward score
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create a dummy dataset
from datasets import Dataset
import pandas as pd

data = {
    "prompt": [
        "What is the capital of France?",
        "Explain quantum mechanics",
    ],
    "chosen": [
        "Paris is the capital of France.",
        "Quantum mechanics is a fundamental theory...",
    ],
    "rejected": [
        "I don't know where France is.",
        "Quantum stuff is really confusing.",
    ],
}
dataset = Dataset.from_pandas(pd.DataFrame(data))

# Initialize trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    per_device_train_batch_size=4,
)

trainer.train()
```

## Supervised Fine-Tuning (SFT)

Before RLHF, fine-tune on high-quality demonstrations:

```python
from trl import SFTTrainer
from datasets import Dataset

# Format data as conversations
train_data = [
    {"text": "Human: What is Python?\n\nAssistant: Python is a programming language."},
    {"text": "Human: Explain photosynthesis\n\nAssistant: Photosynthesis is..."},
]

dataset = Dataset.from_pandas({"text": train_data})

trainer = SFTTrainer(
    model="gpt2",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    packing=True,  # Pack multiple examples
)

trainer.train()
```

## Full RLHF with PPO

The complete RLHF pipeline with PPO optimization:

```python
from trl import PPOConfig, PPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize models
model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
reward_model = AutoModelForSequenceClassification.from_pretrained("reward_model")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Configure PPO
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    clip_range=0.2,
    target_kl=0.1,
)

# Initialize trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
)

# Training loop
batch_size = 16
prompts = ["What is AI?", "Explain machine learning", ...]

for epoch in range(10):
    # Generate responses
    query_tensors = tokenizer(prompts, return_tensors="pt", padding=True)
    
    response_tensors = []
    for i in range(batch_size):
        gen = model.generate(
            query_tensors[i].unsqueeze(0),
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )
        response_tensors.append(gen.squeeze()[len(query_tensors[i]):])
    
    # Compute rewards
    texts = [tokenizer.decode(r) for r in response_tensors]
    reward_scores = [get_reward(text) for text in texts]
    
    # PPO update
    ppo_trainer.step(query_tensors.input_ids, response_tensors, reward_scores)
    
    # Log metrics
    ppo_trainer.log_stats()
```

## Direct Preference Optimization (DPO)

DPO simplifies RLHF by directly optimizing on preference pairs:

```python
from trl import DPOTrainer
from datasets import Dataset

# Preference data
dpo_data = {
    "prompt": ["What is AI?", "Explain quantum computing"],
    "chosen": ["AI is artificial intelligence, systems that can..."],
    "rejected": ["AI is like robots and stuff I think."],
}
dataset = Dataset.from_pandas(dpo_data)

# Initialize DPO trainer
dpo_trainer = DPOTrainer(
    model=model,           # Policy model
    ref_model=ref_model,   # Reference model (frozen)
    beta=0.1,              # Temperature parameter
    train_dataset=dataset,
    tokenizer=tokenizer,
    per_device_train_batch_size=4,
    max_steps=1000,
)

dpo_trainer.train()
```

The DPO loss:
```
Loss = -E[(x,y_w,y_l) ~ D] [log σ( r_θ(x,y_w) - r_θ(x,y_l) - β log (π(y_w|x)/π_ref(y_w|x)) + β log(π(y_l|x)/π_ref(y_l|x)) )]
```

## Comparative Preference Optimization (CPO)

CPO combines SFT with preference optimization:

```python
from trl import CPOTrainer

cpo_trainer = CPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    CPO_beta=0.1,      # Preference loss coefficient
    SFT_beta=1.0,      # SFT loss coefficient
    max_length=512,
)

cpo_trainer.train()
```

## Training Configuration Options

### PPO Configuration

```python
config = PPOConfig(
    # Learning
    learning_rate=1e-5,
    adam_eps=1e-8,
    adam_beta1=0.9,
    adam_beta2=0.99,
    
    # Batch sizes
    batch_size=64,
    mini_batch_size=16,
    gradient_accumulation_steps=1,
    
    # PPO hyperparameters
    clip_range=0.2,
    clip_range_value=0.2,
    target_kl=0.1,
    ppo_epochs=4,
    gamma=1.0,            # Reward discount
    lam=0.95,             # GAE lambda
    
    # KL divergence
    use_kl_loss=False,
    kl_penalty="kl",      # "kl", "abs", "mse", "full"
    kl_coefficient=0.2,
)
```

### DPO Configuration

```python
dpo_config = {
    "beta": 0.1,              # Temperature (lower = more conservative)
    "loss_type": "sigmoid",   # "sigmoid", "hinge", "ipo", "bowman"
    "label_smoothing": 0.0,   # Label smoothing
    "reverse_ratio": False,   # Whether to reverse preference direction
    "f divergence type": "js_divergence",  # D_f divergence
}
```

## Training Callbacks

TRL integrates with Transformers callbacks:

```python
from transformers import EarlyStoppingCallback
from trl import PPOTrainerCallback

class MetricsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % 100 == 0:
            logs = {
                "policy_loss": get_policy_loss(),
                "kl_div": get_kl_divergence(),
                "reward": get_mean_reward(),
            }
            print(f"Step {state.global_step}: {logs}")
```

## Distributed Training

TRL supports distributed training with Accelerate:

```bash
accelerate launch train_rlhf.py \
    --multi_gpu \
    --num_machines=2 \
    --num_processes=8 \
    --mixed_precision=bf16
```

## Common Issues and Solutions

### Training Instability
```python
# Reduce learning rate and add KL penalty
config = PPOConfig(
    learning_rate=1e-6,  # Much lower
    use_kl_loss=True,
    kl_penalty="kl",
    kl_coefficient=0.1,
)
```

### Reference Model Divergence
```python
# More frequent reference model updates or stronger KL penalty
ref_model = copy.deepcopy(model)
trainer = PPOTrainer(
    ref_model=ref_model,
    # ...
)
```

### Reward Hacking
```python
# Add entropy bonus and diversity penalties
def reward_with_entropy(reward, response):
    base_reward = get_reward(response)
    entropy_bonus = compute_entropy(response)
    return base_reward + 0.01 * entropy_bonus
```

TRL provides the complete toolkit for RLHF-based alignment. Start with SFT, train a reward model, then use PPO or DPO to align the model to human preferences.