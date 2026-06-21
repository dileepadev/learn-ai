---
title: Contrastive Preference Optimization (CPO)
description: Discover Contrastive Preference Optimization (CPO), an alignment algorithm designed to prevent model degradation in machine translation and structured generation.
---

Standard preference alignment algorithms (like Direct Preference Optimization - DPO) were designed for general chat models. When applied to machine translation or structured formatting tasks, DPO often degrades the model's capabilities, leading to grammatical errors and formatting loss.

**Contrastive Preference Optimization (CPO)** is an alignment algorithm developed to address this limitation. By incorporating a behavior-cloning loss constraint and a length-normalized contrastive objective, CPO aligns models on translation quality and structured formatting without degrading their base generation capabilities.

---

## Why DPO Fails on Machine Translation

DPO works by training a policy $\pi_\theta$ to maximize the likelihood ratio between a chosen response $y_w$ and a rejected response $y_l$ relative to a reference model $\pi_{\text{ref}}$:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

In machine translation:
1. **The reference translation is optimal:** The chosen response $y_w$ is often a near-perfect human translation.
2. **Likelihood Collapse:** DPO's objective only optimizes the *relative* likelihood difference. To satisfy this, the optimizer can decrease the absolute probability of the rejected translation $y_l$ rather than increasing the probability of the chosen translation $y_w$. This degrades the model's overall command of both languages.
3. **KL-Divergence Drift:** Without strict regularization, DPO quickly drifts away from the pre-trained translation distribution, leading to output hallucinations.

---

## The CPO Design

CPO prevents degradation by combining a **contrastive preference loss** with a **behavior-cloning loss** on the chosen responses.

### 1. Behavior-Cloning Regularizer
To prevent the model from decreasing the absolute probability of the optimal chosen response $y_w$, CPO adds a standard supervised fine-tuning (SFT) log-likelihood term:

$$\mathcal{L}_{\text{BC}}(\theta) = - \log \pi_\theta(y_w|x)$$

This acts as an anchor, ensuring the model continues to assign high probability to high-quality translations.

### 2. Contrastive Loss with Length Normalization
The core alignment term uses a length-normalized contrastive loss that directly compares the token-level probabilities of the chosen and rejected sequences without requiring a reference model:

$$\mathcal{L}_{\text{Contrastive}}(\theta) = -\mathbb{E} \left[ \log \sigma \left( \beta \frac{\log \pi_\theta(y_w|x)}{|y_w|} - \beta \frac{\log \pi_\theta(y_l|x)}{|y_l|} \right) \right]$$

This normalization prevents the model from favoring longer translations simply because of token accumulation.

### 3. Unified CPO Loss Function
The complete CPO objective is a weighted combination of these two terms:

$$\mathcal{L}_{\text{CPO}}(\theta) = \mathcal{L}_{\text{Contrastive}}(\theta) + \alpha \cdot \mathcal{L}_{\text{BC}}(\theta)$$

Where $\alpha > 0$ controls the strength of the SFT anchoring loss.

---

## CPO vs. DPO vs. SFT

| Metric | Supervised Fine-Tuning (SFT) | Direct Preference Optimization (DPO) | Contrastive Preference Optimization (CPO) |
|---|---|---|---|
| **Objective** | Maximize chosen likelihood | Maximize relative likelihood ratio | Maximize relative log probability + anchor SFT |
| **Reference Model** | Not Required | Required | Not Required |
| **Memory Footprint**| Low | High | Low |
| **Translation Quality**| Moderate (prone to style drift) | Low (degrades syntax) | High (preserves translation quality) |

---

## Implementing CPO with Hugging Face TRL

Hugging Face's `trl` library provides support for CPO via the `CPOTrainer`.

```python
from datasets import load_dataset
from trl import CPOConfig, CPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load translation preference dataset
# columns: "prompt" (source text), "chosen" (perfect translation), "rejected" (flawed translation)
dataset = load_dataset("json", data_files="translation_pref_data.json")

# 2. Load Model and Tokenizer (No reference model needed!)
model = AutoModelForCausalLM.from_pretrained("CohereForAI/aya-expanse-8b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-expanse-8b")

# 3. Configure CPO
cpo_config = CPOConfig(
    output_dir="./cpo_aligned_translation_model",
    beta=0.1,                 # Contrastive scale parameter
    cpo_alpha=1.0,            # Behavior cloning regularizer weight (α)
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine",
    max_length=512,
    logging_steps=10,
)

# 4. Initialize Trainer
trainer = CPOTrainer(
    model=model,
    args=cpo_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 5. Start training
trainer.train()
```
