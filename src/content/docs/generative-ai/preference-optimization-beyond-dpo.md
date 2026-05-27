---
title: Preference Optimization Beyond DPO
description: Explore the growing family of DPO alternatives for aligning language models from preference data — covering KTO, IPO, CPO, ORPO, SimPO, and RLOO — understanding the theoretical motivations, practical tradeoffs, and when each approach outperforms standard DPO.
---

Direct Preference Optimization (DPO) transformed LLM alignment by eliminating the need for a separate reward model, but its implicit reward formulation has well-documented failure modes: it overfits to distribution shifts, is sensitive to reference model choice, and can reward length over quality. A wave of follow-on work has produced a rich family of alignment objectives — each addressing specific limitations of DPO while retaining its simplicity relative to full RLHF. Understanding this landscape is essential for practitioners selecting alignment strategies.

## DPO as Baseline

DPO (Rafailov et al., 2023) derives a closed-form optimal policy given a Bradley-Terry preference model. For chosen response $y_w$ and rejected response $y_l$ given prompt $x$:

$$\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right)\right]$$

**Limitations:**

- Requires a reference model ($\pi_{\mathrm{ref}}$) at every training step — memory overhead
- Assumes pairwise comparisons with a strong Bradley-Terry model — fails on noisy or ordinal preferences
- Implicit reward can become miscalibrated when policy drifts far from the reference
- Prone to reward hacking via length bias

## IPO: Identity Preference Optimization

**IPO** (Azar et al., 2024) addresses DPO's theoretical fragility: DPO's derivation assumes infinite data and exact preference probabilities. With finite noisy data, the Bradley-Terry assumption breaks down.

IPO replaces the log-sigmoid loss with a squared difference directly on preference margins, without assuming any functional form for how humans generate preferences:

$$\mathcal{L}_{\mathrm{IPO}} = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} - \frac{1}{2\beta}\right)^2\right]$$

The target margin $\frac{1}{2\beta}$ prevents the policy from growing the log-ratio margin arbitrarily, adding an implicit regularization that DPO lacks. IPO is more robust to noise in preference labels.

## KTO: Kahneman-Tversky Optimization

**KTO** (Ethayarajh et al., 2024) is motivated by Kahneman-Tversky prospect theory — humans are more sensitive to losses than to gains of equal magnitude. Critically, KTO does not require paired preferences: it works with unpaired data where each response is labeled simply as "good" or "bad".

The KTO objective separately maximizes good-response likelihood and minimizes bad-response likelihood, weighted by a reference point:

$$\mathcal{L}_{\mathrm{KTO}} = \mathbb{E}\left[w(y) \cdot \left(1 - \sigma\!\left(\beta\!\left(r_\theta(x, y) - z_{\mathrm{ref}}\right)\right)\right)\right]$$

where $r_\theta(x, y) = \log\frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$ is the implicit reward, $w(y)$ is $\lambda_D$ for desirable (good) responses and $-\lambda_U$ for undesirable (bad) responses, and $z_{\mathrm{ref}} = \mathrm{KL}[\pi_\theta \| \pi_{\mathrm{ref}}]$ is a reference-anchored baseline.

```python
import torch
import torch.nn.functional as F


def kto_loss(
    policy_chosen_logps,     # log p_θ(y_good | x)
    policy_rejected_logps,   # log p_θ(y_bad | x)
    ref_chosen_logps,        # log p_ref(y_good | x)
    ref_rejected_logps,      # log p_ref(y_bad | x)
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
):
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)

    # KL divergence estimate as reference baseline
    kl = (policy_chosen_logps - ref_chosen_logps).mean().detach()

    # Desirable (chosen) loss
    chosen_loss = desirable_weight * (1 - F.sigmoid(chosen_reward - kl))
    # Undesirable (rejected) loss
    rejected_loss = undesirable_weight * (1 - F.sigmoid(kl - rejected_reward))

    return (chosen_loss + rejected_loss).mean()
```

**Key advantage:** KTO works on unpaired data — you only need a binary quality label, not a pair of responses where one is preferred over the other. This makes it practical for datasets where pairwise comparisons are unavailable.

## CPO: Contrastive Preference Optimization

**CPO** (Xu et al., 2024) adds a supervised fine-tuning (SFT) term for the chosen response directly into the DPO objective, preventing forgetting of the chosen response distribution:

$$\mathcal{L}_{\mathrm{CPO}} = \mathcal{L}_{\mathrm{DPO}} - \mathbb{E}\left[\log \pi_\theta(y_w \mid x)\right]$$

The SFT term ensures the model increases probability of good responses regardless of the rejected comparison, eliminating the reference model dependence while guarding against length-exploiting shortcuts.

## ORPO: Odds Ratio Preference Optimization

**ORPO** (Hong et al., 2024) eliminates the reference model entirely by using the **odds ratio** between chosen and rejected responses:

$$\mathcal{L}_{\mathrm{ORPO}} = -\mathbb{E}\left[\log \sigma\!\left(\log \frac{\pi_\theta(y_w \mid x)}{1 - \pi_\theta(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{1 - \pi_\theta(y_l \mid x)}\right)\right] - \mathbb{E}\left[\log \pi_\theta(y_w \mid x)\right]$$

The odds ratio $\frac{p}{1-p}$ naturally stays bounded even when response probabilities are near 0 or 1, providing better training stability than log-ratio differences. The SFT term is built in. ORPO requires 50% less GPU memory than DPO because there is no reference model:

```python
def orpo_loss(policy_chosen_logps, policy_rejected_logps):
    """ORPO: no reference model needed."""
    # Convert log-probabilities to odds
    log_odds_chosen = policy_chosen_logps - torch.log1p(-policy_chosen_logps.exp())
    log_odds_rejected = policy_rejected_logps - torch.log1p(-policy_rejected_logps.exp())

    # Preference loss (contrastive)
    preference_loss = -F.logsigmoid(log_odds_chosen - log_odds_rejected)

    # SFT loss (maximize chosen)
    sft_loss = -policy_chosen_logps

    return (preference_loss + sft_loss).mean()
```

## SimPO: Simple Preference Optimization

**SimPO** (Meng et al., 2024) also removes the reference model but takes a different approach: normalize log-probabilities by sequence length and introduce a reward margin $\gamma$:

$$\mathcal{L}_{\mathrm{SimPO}} = -\mathbb{E}\left[\log \sigma\!\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l \mid x) - \gamma\right)\right]$$

Length normalization directly targets DPO's length bias — longer responses have higher raw log-probability, so DPO inadvertently rewards verbosity. By normalizing by sequence length, SimPO ensures the preference signal is about quality not quantity. The margin $\gamma$ prevents the optimization from collapsing when margins are near zero.

SimPO consistently outperforms DPO on AlpacaEval and Arena-Hard while requiring less compute.

## RLOO: REINFORCE Leave-One-Out

**RLOO** (Ahmadian et al., 2024) returns to explicit RL but avoids training a separate reward model by using the policy's own outputs as a baseline. For each prompt, sample $k$ responses and use the mean reward of the other $k-1$ as the baseline for each:

$$\mathcal{L}_{\mathrm{RLOO}} = -\mathbb{E}\left[\sum_{i=1}^k \left(r(y_i) - \bar{r}_{-i}\right) \cdot \log \pi_\theta(y_i \mid x)\right]$$

where $\bar{r}_{-i} = \frac{1}{k-1}\sum_{j \neq i} r(y_j)$ is the leave-one-out baseline. RLOO is particularly effective when a verifiable reward signal is available (math accuracy, code execution results) — aligning with the trend toward RL from verifiable rewards.

## Choosing the Right Method

| Scenario | Recommended Method |
| --- | --- |
| Paired preference data, reference model available | DPO or IPO |
| Noisy preference labels | IPO |
| Unpaired binary quality labels | KTO |
| No reference model (memory-constrained) | ORPO or SimPO |
| Length bias is a problem | SimPO |
| Verifiable rewards (math, code) | RLOO or GRPO |
| Standard supervised preference | CPO |

## Practical Implementation with TRL

The HuggingFace TRL library implements most of these:

```python
from trl import DPOTrainer, KTOTrainer, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# ORPO (no reference model needed)
trainer = ORPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,       # needs "prompt", "chosen", "rejected" columns
    tokenizer=tokenizer,
    beta=0.1,
)
trainer.train()

# KTO (unpaired data — only "prompt", "completion", "label" columns)
kto_trainer = KTOTrainer(
    model=model,
    args=training_args,
    train_dataset=unpaired_dataset,
    tokenizer=tokenizer,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)
kto_trainer.train()
```

## Summary

The DPO family has rapidly diversified to address specific failure modes:

- **IPO** adds a squared margin target to prevent reward hacking under finite noisy data
- **KTO** removes the pairwise comparison requirement — binary good/bad labels suffice
- **CPO** adds an SFT term to prevent forgetting the chosen distribution
- **ORPO** and **SimPO** eliminate the reference model entirely — reducing memory by ~50%
- **SimPO** further adds length normalization to prevent verbosity reward hacking
- **RLOO** enables explicit RL from verifiable rewards without a trained reward model

No single method dominates across all settings. The key variables are data format (paired vs. unpaired), memory constraints (reference model cost), and the nature of the reward signal (human preferences vs. verifiable correctness).
