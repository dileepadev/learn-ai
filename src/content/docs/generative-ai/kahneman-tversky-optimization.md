---
title: Kahneman-Tversky Optimization (KTO)
description: Explore Kahneman-Tversky Optimization (KTO), a preference alignment algorithm derived from prospect theory that allows optimizing LLMs directly from binary feedback without paired preference data.
---

Kahneman-Tversky Optimization (KTO) is a modern preference alignment method for Large Language Models that directly incorporates insights from behavioral economics—specifically **prospect theory** (pioneered by Daniel Kahneman and Amos Tversky). 

Unlike Direct Preference Optimization (DPO), which requires paired preference datasets ($y_w \succ y_l$ given prompt $x$), KTO aligns models using **unpaired binary feedback** (e.g., simply labeling outputs as "good" or "bad"). This dramatically reduces the cost and complexity of data curation.

---

## The Behavioral Economics Foundation

Standard alignment methods like RLHF and DPO assume human decision-making follows **utility maximization** (specifically, the Bradley-Terry model). Under Bradley-Terry, the probability that a human prefers output $A$ over output $B$ depends purely on the difference in utility:

$$P(A \succ B) = \sigma(R(A) - R(B))$$

However, Kahneman and Tversky's prospect theory showed that humans do not perceive utility in this rational, linear way. Instead:
1. **Reference Dependence:** Humans evaluate outcomes relative to a subjective baseline (reference point) rather than in absolute terms.
2. **Loss Aversion:** Humans are significantly more sensitive to losses than to equivalent gains (losing \$100 hurts more than winning \$100 pleases).
3. **Diminishing Sensitivity:** The marginal utility of both gains and losses decreases as their magnitude increases.

KTO translates these behavioral heuristics directly into a mathematical loss function.

---

## How KTO Works

Instead of maximizing the likelihood that a model prefers a "winning" response over a "losing" response, KTO defines a value function $V$ that mimics human psychology.

Given a prompt $x$ and a model output $y$, let $r_\theta(x, y)$ be the implicit reward:

$$r_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

Where:
- $\pi_\theta$ is the model being trained.
- $\pi_{\text{ref}}$ is the reference policy.
- $\beta$ is a scale parameter controlling divergence.

The KTO utility function $h(r)$ is defined as:

$$h(r) = \begin{cases} 
1 - e^{-r} & \text{if } r > 0 \text{ (Gain)} \\
\lambda (e^r - 1) & \text{if } r \le 0 \text{ (Loss)}
\end{cases}$$

Here, $\lambda > 1$ is the loss aversion coefficient (typically set between $1.33$ and $2.0$), ensuring that negative feedback is weighted more heavily than positive feedback.

### The KTO Loss Function

The KTO objective is split into two components: one for desirable outputs ($y \in Y_+$) and one for undesirable outputs ($y \in Y_-$).

$$\mathcal{L}_{\text{KTO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{x, y \sim Y_+} \left[ \log \sigma \left( r_\theta(x, y) - z_{\text{ref}} \right) \right] - \lambda \mathbb{E}_{x, y \sim Y_-} \left[ \log \sigma \left( z_{\text{ref}} - r_\theta(x, y) \right) \right]$$

Where the reference signal $z_{\text{ref}}$ acts as the subjective baseline, defined as the expected implicit reward over the prompt distribution:

$$z_{\text{ref}} = \mathbb{E}_{x' \sim \mathcal{D}} \left[ r_\theta(x', y') \right]$$

By contrasting the reward of an output against the running average of rewards ($z_{\text{ref}}$), KTO determines whether an output is perceived as a "gain" or a "loss" by the model.

---

## Comparison: KTO vs. DPO

| Feature | Direct Preference Optimization (DPO) | Kahneman-Tversky Optimization (KTO) |
|---|---|---|
| **Data Format** | Paired preferences: $(x, y_w, y_l)$ | Unpaired binary labels: $(x, y, \text{is\_good})$ |
| **Theoretical Basis** | Bradley-Terry Model (Rational Utility) | Prospect Theory (Behavioral Economics) |
| **Data Collection** | Hard (requires rating multiple outputs side-by-side) | Easy (can label single logs or thumbs-up/down) |
| **Sample Efficiency** | High (for paired datasets) | High (can handle highly imbalanced datasets) |
| **Reference Point** | Implicitly relative to the paired sample | Explicitly tracked via $z_{\text{ref}}$ |

---

## Implementing KTO with Hugging Face TRL

The Hugging Face `trl` library provides first-class support for KTO training via the `KTOTrainer`.

```python
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load Dataset (needs columns: "prompt", "completion", and "label")
# label = True (desirable) or False (undesirable)
dataset = load_dataset("json", data_files="kto_data.json")

# 2. Initialize Models
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 3. Define Configurations
kto_config = KTOConfig(
    output_dir="./kto_aligned_model",
    beta=0.1,             # Controls strength of KL penalty
    desirable_weight=1.0, # Weight of positive feedback
    undesirable_weight=1.33, # λ: loss aversion coefficient
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
)

# 4. Initialize Trainer
trainer = KTOTrainer(
    model=model,
    ref_model=ref_model,
    args=kto_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 5. Start Training
trainer.train()
```

---

## Practical Engineering Insights

1. **Setting the Loss Aversion Coefficient ($\lambda$):** In practice, setting $\lambda$ in the range $[1.33, 1.5]$ is ideal. Setting it too high makes the model overly conservative and passive, while setting it too low makes the model prone to hallucination and rule-breaking.
2. **Handling Imbalanced Datasets:** One major advantage of KTO is its robustness to imbalanced feedback. If you have 90% positive examples and 10% negative examples, KTO can still align the model successfully by adjusting the relative weights of the positive and negative terms.
3. **Computing $z_{\text{ref}}$:** Ensure your batch size is large enough to construct a stable estimate of the running average reward baseline $z_{\text{ref}}$. If the batch size is too small, $z_{\text{ref}}$ will fluctuate wildly, causing unstable updates.
