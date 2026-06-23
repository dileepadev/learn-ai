---
title: Simple Preference Optimization (SimPO)
description: Explore Simple Preference Optimization (SimPO), a reference-free offline preference alignment algorithm that aligns LLMs using length-normalized log probability and target margins.
---

**Simple Preference Optimization (SimPO)** is a highly efficient, reference-free offline preference alignment algorithm designed as an alternative to Direct Preference Optimization (DPO). 

While DPO relies on a separate reference model to regularize the training policy and prevent model degradation, SimPO completely removes the reference model. By incorporating a length-normalized log probability metric and a target reward margin, SimPO achieves better alignment results with 50% less VRAM and faster training steps.

---

## The Limitation of DPO's Implicit Reward

DPO aligns language models by maximizing the likelihood of the chosen response $y_w$ relative to the rejected response $y_l$, using a reference model $\pi_{\text{ref}}$:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

This formulation suffers from three key issues:
1. **Memory Overhead:** Storing $\pi_{\text{ref}}$ in VRAM alongside $\pi_\theta$ limits the maximum context length or model size you can train.
2. **Length Bias:** The raw log probability $\log \pi_\theta(y|x)$ is not normalized by length. As a result, models aligned with DPO are highly prone to verbosity bias, prioritizing longer answers simply because they contain more token probabilities.
3. **Reference Dependency:** The gradient update depends heavily on the output distribution of the reference model, which may be suboptimal.

---

## The SimPO Design

SimPO replaces the implicit reward difference with a **length-normalized log probability** difference and introduces a target margin $\gamma > 0$ to enforce a separation between chosen and rejected responses.

### 1. Length-Normalized Reward
The reward $R_{\text{SimPO}}(x, y)$ is defined as the average log likelihood of the tokens in $y$ given prompt $x$:

$$p_\theta(y|x) = \frac{1}{|y|} \log \pi_\theta(y|x) = \frac{1}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i | x, y_{<i})$$

Where $|y|$ is the number of tokens in the response. This normalization prevents the model from generating verbose, low-quality responses to inflate its reward.

### 2. Margin-Enhanced Loss
SimPO optimizes the parameters $\theta$ using the following objective function:

$$\mathcal{L}_{\text{SimPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta p_\theta(y_w|x) - \beta p_\theta(y_l|x) - \gamma \right) \right]$$

Where:
- $\beta$ is a scaling factor controlling the magnitude of the rewards.
- $\gamma$ is a target margin that acts as a buffer. It forces the log probability of the chosen response to be at least $\gamma/\beta$ higher than the rejected response, preventing the optimizer from stopping early once they are marginally separated.

---

## SimPO vs. DPO: A Core Comparison

| Feature | Direct Preference Optimization (DPO) | Simple Preference Optimization (SimPO) |
|---|---|---|
| **Reference Model** | Required (increases VRAM footprint) | Not Required (50% VRAM savings) |
| **Reward Metric** | $\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ | $\frac{\beta}{|y|} \log \pi_\theta(y|x)$ |
| **Length Bias Control** | Weak (prone to verbosity) | Strong (explicitly normalized) |
| **Optimization Target** | Likelihood ratio optimization | Margin-based separation |

---

## Implementing SimPO with Hugging Face TRL

The `CPOTrainer` in Hugging Face's `trl` library (or native implementations) can be configured to support SimPO since it shares a similar structure.

```python
from datasets import load_dataset
from trl import CPOConfig, CPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load preference dataset (columns: "prompt", "chosen", "rejected")
dataset = load_dataset("json", data_files="preference_data.json")

# 2. Load Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# 3. Configure SimPO (CPO Config can run SimPO by setting loss_type="simpo")
simpo_config = CPOConfig(
    output_dir="./simpo_aligned_model",
    loss_type="simpo",        # <-- Enables SimPO loss
    beta=2.0,                 # Scale factor
    simpo_gamma=1.0,          # Margin target (γ)
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine",
    max_length=1024,
    logging_steps=10,
)

# 4. Initialize Trainer (Reference Model is omitted!)
trainer = CPOTrainer(
    model=model,
    args=simpo_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 5. Start alignment
trainer.train()
```

---

## Practical Tuning Guide

- **Margin ($\gamma$) Selection:** A value of $1.0$ to $1.5$ is usually the sweet spot. If $\gamma$ is too small, the model fails to differentiate clearly between outputs. If too large, training becomes unstable as gradients explode trying to reach the margin.
- **Beta ($\beta$) Selection:** The recommended value of $\beta$ for SimPO is higher than in DPO, typically ranging between $2.0$ and $2.5$.
- **Length Normalization:** Since SimPO directly normalizes reward by sequence length, it is less prone to generating long-winded answers, making it ideal for chat models where concise answers are desirable.
