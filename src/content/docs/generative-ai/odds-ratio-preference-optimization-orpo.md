---
title: Odds-Ratio Preference Optimization (ORPO)
description: Explore ORPO (Odds-Ratio Preference Optimization), a single-step alignment algorithm that combines supervised fine-tuning and preference alignment into a single loss function.
---

Standard alignment pipelines for Large Language Models (LLMs) are multi-step processes:
1. **Supervised Fine-Tuning (SFT):** Teach the model to follow instructions and format outputs.
2. **Preference Alignment (RLHF or DPO):** Steer the model to prefer desirable outputs over undesirable outputs.

**Odds-Ratio Preference Optimization (ORPO)** is a single-step alignment technique that combines SFT and preference alignment into a single, unified loss function. By eliminating the separate SFT and reference model stages, ORPO reduces training time, memory consumption, and potential model degradation.

---

## The Problem with Two-Step Alignment

In standard SFT, the model learns the joint distribution of prompt and completion. However, SFT does not penalize undesirable responses; it merely increases the likelihood of preferred ones. 

As a result, during SFT, the probability of generating a rejected/bad response often increases alongside the probability of generating the chosen/good response. Subsequent DPO or RLHF stages are then required to suppress the bad responses. This multi-step process:
- Requires loading a separate reference model (e.g., in DPO) to compute KL-divergence, consuming valuable GPU memory.
- Can lead to instability or "catastrophic forgetting" of SFT formatting constraints during the alignment stage.

---

## How ORPO Works

ORPO adds a weak penalty term based on the **odds ratio** directly to the standard cross-entropy loss used in SFT.

For a prompt $x$, a chosen response $y_w$, and a rejected response $y_l$:
1. We compute the **likelihood** of generating the chosen response: $P_\theta(y_w|x)$.
2. We compute the **odds** of generating the chosen response versus other tokens:

$$\text{Odds}_\theta(y_w|x) = \frac{P_\theta(y_w|x)}{1 - P_\theta(y_w|x)}$$

Similarly, the odds of generating the rejected response are:

$$\text{Odds}_\theta(y_l|x) = \frac{P_\theta(y_l|x)}{1 - P_\theta(y_l|x)}$$

The Odds Ratio (OR) measures how much more likely the model is to generate the chosen response compared to the rejected response:

$$\text{OR}_\theta(y_w, y_l | x) = \frac{\text{Odds}_\theta(y_w|x)}{\text{Odds}_\theta(y_l|x)}$$

---

## The ORPO Loss Function

The complete ORPO objective combines SFT loss with a regularizing preference loss:

$$\mathcal{L}_{\text{ORPO}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) + \alpha \cdot \mathcal{L}_{\text{OR}}(\theta)$$

Where:
- $\mathcal{L}_{\text{SFT}}(\theta)$ is the standard negative log-likelihood loss on the chosen response $y_w$:
  
  $$\mathcal{L}_{\text{SFT}}(\theta) = - \log P_\theta(y_w|x)$$

- $\mathcal{L}_{\text{OR}}(\theta)$ is the odds-ratio loss that penalizes the rejected response:
  
  $$\mathcal{L}_{\text{OR}}(\theta) = - \log \sigma \left( \log \text{OR}_\theta(y_w, y_l | x) \right)$$

- $\alpha$ is a scaling factor (typically set between $0.05$ and $0.2$) that balances SFT with preference direction.

As the model trains, the odds-ratio loss pushes the model to actively maximize the gap between the chosen and rejected responses. Because the loss is self-contained, **ORPO does not require a reference model during training**, saving up to 50% of the GPU memory needed for DPO.

---

## Implementing ORPO with Hugging Face TRL

The Hugging Face `trl` library provides support for ORPO via the `ORPOTrainer`.

```python
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load preference dataset (needs columns: "prompt", "chosen", and "rejected")
dataset = load_dataset("json", data_files="preference_data.json")

# 2. Load Model and Tokenizer
# Note: Unlike DPO, we do NOT load a ref_model!
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# 3. Define Configurations
orpo_config = ORPOConfig(
    output_dir="./orpo_aligned_model",
    beta=0.1,             # Equates to α in the paper (controls penalty strength)
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine",
    max_prompt_length=512,
    max_length=1024,
    logging_steps=10,
)

# 4. Initialize Trainer
trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 5. Train
trainer.train()
```

---

## Practical Engineering Insights

1. **Memory Efficiency:** Since there is no reference model, ORPO allows you to train larger models or use larger batch sizes on the same hardware compared to DPO.
2. **Beta Parameter:** If you observe that your model's language capabilities are degrading (e.g., repetitive phrasing or broken grammar), reduce the `beta` parameter to give SFT loss ($\mathcal{L}_{\text{SFT}}$) relatively more weight.
3. **Data Quality:** Since ORPO conducts SFT and alignment concurrently, it is highly sensitive to formatting errors in the chosen responses. Ensure the "chosen" completions are clean and consistent.
