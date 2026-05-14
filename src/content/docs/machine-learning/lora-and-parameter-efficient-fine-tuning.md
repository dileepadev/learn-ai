---
title: "LoRA and Parameter-Efficient Fine-Tuning Techniques"
description: "Master LoRA, QLoRA, and other parameter-efficient fine-tuning methods that enable training large language models on consumer hardware."
---

Parameter-efficient fine-tuning (PEFT) has revolutionized how we adapt large language models. Rather than updating all billions of parameters, PEFT methods modify only a small fraction — enabling fine-tuning on a single GPU.

## The Challenge of Full Fine-Tuning

Fine-tuning a 70B parameter model naively requires:

| Resource | Full Fine-Tuning | LoRA |
|----------|------------------|------|
| GPU Memory | 280GB+ | ~16GB |
| Storage | 140GB (fp16) | ~100MB |
| Training Time | Weeks | Hours |

The fundamental problem: gradient descent requires storing gradients and optimizer states for every parameter, multiplying memory requirements by 3–4×.

## LoRA: Low-Rank Adaptation

LoRA adds trainable rank-decomposition matrices to existing weights:

```python
# Original: W is frozen
y = W @ x

# LoRA: Add low-rank decomposition
W_new = W + B @ A
y = (W + B @ A) @ x = W @ x + B @ A @ x
```

Where:
- `W ∈ R^(d × k)` - frozen pretrained weights
- `B ∈ R^(d × r)` - new trainable matrix
- `A ∈ R^(r × k)` - new trainable matrix
- `r << min(d, k)` - the rank (typically 8–64)

### Why Low-Rank?

The key insight: adapting a model to a new task doesn't require changes across the full rank of the weight matrix. The task-adaptive information can be compressed into a low-rank subspace.

### Implementing LoRA

```python
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# Define LoRA configuration
lora_config = LoraConfig(
    lora_alpha=16,           # Scaling factor
    lora_dropout=0.05,       # Dropout for regularization
    r=16,                    # Rank of the adaptation
    bias="none",             # Don't train biases
    task_type="CAUSAL_LM",   # Task type
    target_modules=[
        "q_proj",            # Query projection
        "k_proj",            # Key projection
        "v_proj",            # Value projection
        "o_proj",            # Output projection
    ],
)

# Apply to model
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 6.28M || all params: 7.12B || trainable%: 0.09%
```

### Merging LoRA Weights

After training, merge LoRA weights into the base model for deployment:

```python
from peft import PeftModel

# Merge and unload
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_output")
```

## QLoRA: Quantized LoRA

QLoRA combines 4-bit quantization with LoRA for even greater efficiency:

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_int8_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_use_double_quant=True,  # Double quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for int8 training
model = prepare_model_for_int8_training(model)

# Apply LoRA
config = LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
```

## Adapter Layers

Insert trainable adapter layers between transformer layers:

```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_size)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_size, hidden_size)
        self.skip = nn.Identity()
    
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

# Insert after attention and MLP in each transformer layer
```

## Prefix Tuning

Add trainable "soft prompts" to each layer:

```python
class PrefixTuning(nn.Module):
    def __init__(self, num_layers, hidden_size, prefix_length=20):
        super().__init__()
        self.prefix_tokens = nn.Parameter(
            torch.randn(num_layers, 2, prefix_length, hidden_size)
        )
    
    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        prefix = self.prefix_tokens.expand(batch_size, -1, -1, -1)
        return torch.cat([prefix, hidden_states], dim=1)
```

## Comparing PEFT Methods

| Method | Memory | Quality | Inference Cost | Best For |
|--------|--------|---------|----------------|----------|
| LoRA | Low | Excellent | None (merged) | General purpose |
| QLoRA | Very Low | Good | None | Consumer GPU fine-tuning |
| Adapter | Low | Good | Small add | Medium models |
| Prefix | Very Low | Moderate | Small add | Very small models |
| IA3 | Low | Good | Small add | Task adaptation |

## LoRA Hyperparameters

### Rank Selection

| Rank | Memory | Quality | Use Case |
|------|--------|---------|----------|
| 8 | Minimal | Good | Simple tasks, small models |
| 16 | Low | Better | Most use cases |
| 32 | Medium | Best | Complex tasks, large models |
| 64 | Higher | Best | Specialized adaptation |

### Target Modules

For different architectures:

```python
# LLaMA, Mistral (transformer)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# T5, encoder-decoder
target_modules = ["q", "k", "v", "o", "wi", "wo"]

# RoBERTa, BERT (MLM)
target_modules = ["query", "key", "value", "output.dense"]
```

### Scaling Factor (alpha)

`lora_alpha` controls the magnitude of LoRA updates:

```python
# Rule of thumb: alpha = 2 × rank
lora_config = LoraConfig(r=16, lora_alpha=32)
```

## Advanced LoRA Techniques

### DoRA: Weight-Decomposed LoRA

Decomposes weights into magnitude and direction:

```python
from peft import DoRAConfig

dora_config = DoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)
```

### AdaLoRA: Adaptive LoRA

Dynamically allocates rank based on importance:

```python
from peft import AdaLoraConfig

adalora_config = AdaLoraConfig(
    target_r=16,
    init_r=12,
    tfa=True,  # TensorFormer compatibility
)
```

### QLoRA with Flash Attention

For efficient training on long sequences:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.pretraining_tp = 1  # Enable TP for memory efficiency
```

## Practical Fine-Tuning Pipeline

```python
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Configure training
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train
trainer.train()

# Save adapter
model.save_pretrained("./lora_adapter")
```

## Deployment Considerations

### Merged Model
For production, merge LoRA weights into the base model:

```python
model = model.merge_and_unload()
model.save_pretrained("./final_model")
```

### Adapter-Only Inference
Keep adapters separate for easier updating:

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
adapter = PeftModel.from_pretrained(base, "./lora_adapter")
output = adapter.generate(input_ids)
```

PEFT methods have democratized fine-tuning. What once required massive GPU clusters now works on a single GPU, enabling rapid experimentation and deployment of specialized models.