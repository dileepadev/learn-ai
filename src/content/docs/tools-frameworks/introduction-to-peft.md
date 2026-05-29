---
title: Introduction to PEFT
description: Get started with HuggingFace PEFT (Parameter-Efficient Fine-Tuning) — the library implementing LoRA, QLoRA, IA³, prefix tuning, and prompt tuning for efficient LLM adaptation — covering installation, configuring adapters, merging weights for zero-overhead inference, and multi-adapter workflows.
---

Fine-tuning a 70-billion-parameter language model by updating every weight requires hundreds of gigabytes of GPU memory and produces a full model checkpoint for every task. **PEFT** (Parameter-Efficient Fine-Tuning) is the HuggingFace library that implements a family of techniques for adapting large models by training only a tiny fraction of parameters while freezing the rest — enabling fine-tuning on a single GPU and producing lightweight adapters instead of full model copies.

## Why PEFT?

| Approach | Trainable Params | Memory (7B model) | Adapter Size |
| --- | --- | --- | --- |
| Full fine-tuning | 100% | ~112 GB | Full checkpoint |
| LoRA (rank 8) | ~0.1% | ~14 GB | ~50 MB |
| QLoRA (4-bit + LoRA) | ~0.1% | ~5 GB | ~50 MB |
| IA³ | ~0.01% | ~14 GB | ~1 MB |
| Prefix tuning | ~0.1% | ~14 GB | ~10 MB |

## Installation

```bash
pip install peft transformers accelerate bitsandbytes
```

## LoRA with PEFT

LoRA (Low-Rank Adaptation) injects trainable low-rank decomposition matrices into the query and value projection layers of each Transformer block. PEFT wraps any HuggingFace model with a `LoraConfig`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                # Rank of the decomposition
    lora_alpha=32,                       # Scaling factor (lora_alpha / r)
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Which layers to adapt
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Trainable params: 41,943,040 || All params: 8,072,192,000 || Trainable%: 0.52%
```

### Training with PEFT + Trainer

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return tokenizer(prompt, truncation=True, max_length=512, padding="max_length")

tokenized = dataset.map(format_example, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./lora-llama3-alpaca",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained("./lora-llama3-alpaca-adapter")
```

## QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) combines 4-bit NormalFloat quantization of the base model with LoRA adapters — reducing memory by ~4× compared to standard LoRA:

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4: optimal for normally distributed weights
    bnb_4bit_compute_dtype="bfloat16",   # Computation in bfloat16
    bnb_4bit_use_double_quant=True,      # Double quantization for additional memory savings
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B",
    quantization_config=quantization_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)  # Enable gradient checkpointing for 4-bit models

qlora_config = LoraConfig(
    r=64, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, qlora_config)
```

QLoRA enables fine-tuning LLaMA-3-70B on a single 48 GB GPU — previously requiring 8× A100s.

## IA³: Infused Adapter by Inhibiting and Amplifying Inner Activations

IA³ (Liu et al., 2022) is even more parameter-efficient than LoRA. It introduces learned scaling vectors that multiply key, value, and feedforward layer activations:

$$h \leftarrow l_k \odot h \text{ (keys/values)}, \quad h \leftarrow l_{ff} \odot \gamma(W_{ff} h) \text{ (FFN)}$$

```python
from peft import IA3Config

ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],  # Where to add scaling vectors
    feedforward_modules=["down_proj"],
)
model = get_peft_model(base_model, ia3_config)
model.print_trainable_parameters()
# Trainable params: 786,432 || Trainable%: 0.009%
```

IA³ trains ~10× fewer parameters than LoRA while achieving similar performance on many tasks — particularly effective for instruction tuning.

## Prefix Tuning and Prompt Tuning

**Prefix tuning** prepends trainable prefix tokens to the key and value sequences of every attention layer, conditioning the model on the prefix without modifying existing weights.

**Prompt tuning** prepends trainable soft token embeddings only to the input sequence — even simpler and fewer parameters.

```python
from peft import PrefixTuningConfig, PromptTuningConfig, PromptTuningInit

# Prefix tuning
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,    # 20 learnable prefix tokens per attention layer
    prefix_projection=True,   # Reparameterize through MLP for stability
)

# Prompt tuning (soft prompts)
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Classify the sentiment of the following text:",
    tokenizer_name_or_path="gpt2",
)
```

## Loading and Merging Adapters

### Load a Saved Adapter

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./lora-llama3-alpaca-adapter")

# Inference with adapter
model.eval()
with torch.no_grad():
    inputs = tokenizer("### Instruction:\nExplain LoRA.\n\n### Response:\n", return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Merge LoRA Weights for Zero-Overhead Inference

After training, merge the LoRA adapter into the base model weights for inference with no additional computation:

```python
# Merge adapter weights into base model (produces a standard model with no PEFT overhead)
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("./merged-llama3-alpaca")
tokenizer.save_pretrained("./merged-llama3-alpaca")

# The merged model is a normal HuggingFace model — no PEFT dependency needed for inference
```

## Multi-Adapter Workflows

PEFT supports attaching and switching multiple adapters on the same base model:

```python
# Train and save multiple adapters for different tasks
model.load_adapter("./adapter-coding", adapter_name="coding")
model.load_adapter("./adapter-medical", adapter_name="medical")
model.load_adapter("./adapter-legal", adapter_name="legal")

# Switch active adapter at inference time
model.set_adapter("coding")
output_code = model.generate(**coding_prompt)

model.set_adapter("medical")
output_med = model.generate(**medical_prompt)

# Disable adapter entirely (use base model)
with model.disable_adapter():
    output_base = model.generate(**prompt)
```

## Choosing the Right PEFT Method

| Task | Recommended Method | Why |
| --- | --- | --- |
| Instruction fine-tuning | LoRA (r=8–64) | Strong performance, flexible rank |
| Limited GPU memory (large model) | QLoRA | 4-bit base + LoRA adapters |
| Many tasks, tiny storage budget | IA³ | 10× fewer params than LoRA |
| Soft-prompt learning | Prompt tuning | No weight modification at all |
| Cross-layer conditioning | Prefix tuning | Conditions all attention layers |
| Production serving (no overhead) | Merge LoRA | Zero runtime cost post-merge |

## Summary

PEFT democratizes LLM fine-tuning by making it feasible on consumer and research hardware:

- **LoRA** injects low-rank adapter matrices into attention projections — 0.1–1% of parameters, full fine-tuning quality
- **QLoRA** adds 4-bit quantization of the base model — enabling 70B+ model fine-tuning on a single GPU
- **IA³** uses learned per-layer scaling vectors — 10× fewer parameters than LoRA, competitive quality
- **Prefix and prompt tuning** condition the model without touching weights — minimal storage, weak for complex tasks
- **Adapter merging** eliminates all runtime overhead — the merged model is indistinguishable from a fully fine-tuned model at inference time
- **Multi-adapter support** enables a single base model to serve many specialized adapters, switching tasks without reloading weights
