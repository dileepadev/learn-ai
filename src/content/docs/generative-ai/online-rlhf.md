---
title: Online RLHF and Preference Learning
description: Understand online reinforcement learning from human feedback — covering PPO-based LLM alignment, iterative preference collection, online DPO, REST-EM, self-play fine-tuning, and how continuous feedback loops overcome the distributional shift limitations of static offline preference datasets.
---

The dominant paradigm for aligning language models — **Reinforcement Learning from Human Feedback (RLHF)** — encompasses two fundamentally different training strategies: *offline* methods that optimize over a fixed preference dataset collected before training, and *online* methods that continuously collect new preferences from the model's current behavior. The distinction has profound consequences for alignment quality, reward hacking susceptibility, and training stability.

## The Offline vs Online Distinction

**Offline preference learning** (e.g., standard DPO) collects a static dataset of preference pairs $(x, y_w, y_l)$ — where $y_w$ is preferred over $y_l$ for prompt $x$ — and optimizes a fixed objective:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

The problem: as $\pi_\theta$ improves, the fixed dataset $\mathcal{D}$ becomes increasingly **off-policy** — the model learns to avoid the specific losing responses in $\mathcal{D}$ but may produce new failure modes never observed during data collection.

**Online preference learning** generates new responses from the *current* policy $\pi_\theta$ at each training step, collecting feedback on these on-policy samples. This keeps data distribution aligned with the current model, enabling more targeted improvement.

## PPO-Based RLHF

The original RLHF pipeline (Christiano et al., 2017; InstructGPT, 2022) uses Proximal Policy Optimization with a learned reward model.

### The Four-Component System

1. **SFT model** $\pi_{\text{SFT}}$: base model fine-tuned on instruction-following demonstrations
1. **Reward model** $r_\phi(x, y)$: trained on human preference pairs to predict which response is preferred
1. **Policy** $\pi_\theta$: the model being aligned, initialized from SFT
1. **Value model** $V_\psi(x, y_{<t})$: estimates expected return for PPO's advantage estimation

### PPO Objective for LLMs

The policy maximizes reward while staying close to the SFT reference:

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y) - \beta \cdot \text{KL}[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)]\right]$$

The KL penalty prevents the policy from exploiting reward model blind spots (reward hacking). PPO clips the probability ratio to stabilize updates:

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t\left[\min\!\left(\hat{A}_t \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}, \hat{A}_t \cdot \text{clip}\!\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}, 1-\varepsilon, 1+\varepsilon\right)\right)\right]$$

### Implementation with TRL

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# Load models
model = AutoModelForCausalLMWithValueHead.from_pretrained("sft-model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft-model")
tokenizer = AutoTokenizer.from_pretrained("sft-model")
reward_model = load_reward_model("reward-model-checkpoint")

config = PPOConfig(
    model_name="sft-model",
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=16,
    ppo_epochs=4,
    kl_penalty="kl",
    target_kl=6.0,       # Adaptive KL coefficient target
    init_kl_coef=0.2,
)

trainer = PPOTrainer(config, model, ref_model, tokenizer)

for batch in dataloader:
    queries = batch["input_ids"]
    # Generate on-policy responses
    responses = trainer.generate(queries, max_new_tokens=256)

    # Score with reward model
    rewards = [reward_model(q, r) for q, r in zip(queries, responses)]

    # PPO update
    stats = trainer.step(queries, responses, rewards)
```

### Challenges with PPO

PPO for LLMs is notoriously **training-unstable** and requires careful tuning:

- Four models loaded simultaneously (policy, reference, reward, value) → high memory
- Reward hacking: policy learns to exploit reward model errors, producing high-scoring but poor-quality outputs
- KL budget management: too-low KL allows hacking; too-high KL undoes SFT alignment
- Advantage estimation is noisy at the token level

## REINFORCE with Leave-One-Out (RLOO)

A simpler online alternative to PPO uses the REINFORCE policy gradient with a leave-one-out baseline:

$$\nabla_\theta \mathcal{J} = \mathbb{E}\left[(r(y) - b_{-i}) \nabla_\theta \log \pi_\theta(y|x)\right]$$

where $b_{-i} = \frac{1}{K-1}\sum_{j \neq i} r(y_j)$ is the mean reward of the other $K-1$ responses sampled for the same prompt. RLOO requires no value model and is significantly simpler than PPO while achieving comparable performance at scale.

## Online DPO

**Online DPO** (Guo et al., 2024) addresses the off-policy problem by generating preference pairs from the current policy at each training step:

```python
def online_dpo_step(model, ref_model, reward_model, prompt_batch):
    # Generate two responses from current policy
    response_a = model.generate(prompt_batch, do_sample=True, temperature=0.9)
    response_b = model.generate(prompt_batch, do_sample=True, temperature=0.9)

    # Score with reward model to create preference pairs
    score_a = reward_model(prompt_batch, response_a)
    score_b = reward_model(prompt_batch, response_b)

    # Assign winner/loser based on scores
    y_w = torch.where(score_a > score_b, response_a, response_b)
    y_l = torch.where(score_a > score_b, response_b, response_a)

    # DPO loss on freshly generated on-policy pairs
    loss = dpo_loss(model, ref_model, prompt_batch, y_w, y_l, beta=0.1)
    return loss
```

Online DPO maintains the simplicity of DPO (no PPO complexity, no value model) while providing on-policy data. The reward model acts as a preference oracle rather than an RL reward signal.

## Iterative DPO

**Iterative DPO** (or Self-Rewarding Language Models, Yuan et al., 2024) alternates between:

1. **Data generation**: sample responses from current $\pi_\theta$, score them with a reward model or LLM-as-judge
1. **DPO update**: train on the new preference pairs, producing $\pi_{\theta'}$
1. **Repeat**: use the improved $\pi_{\theta'}$ to generate the next iteration's data

Each iteration uses the current policy as the reference model, gradually shifting the alignment target:

$$\pi_{\theta_0} \xrightarrow{\text{DPO}} \pi_{\theta_1} \xrightarrow{\text{DPO}} \pi_{\theta_2} \xrightarrow{\text{DPO}} \cdots$$

The key insight: using the model itself as the judge (LLM-as-judge) removes the need for a separate reward model, enabling self-improvement without additional labeled data.

## REST-EM: Reward-ranked Self-Improvement

**REST-EM** (Gulcehre et al., 2023) frames online preference learning as Expectation-Maximization:

- **E-step**: generate $K$ candidate responses per prompt, rank them by reward, keep the top $p\%$
- **M-step**: supervised fine-tuning on the reward-selected responses

$$\pi_{\theta}^{(t+1)} = \arg\max_\theta \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_{\theta}^{(t)}(\cdot|x)} [r(x,y) \cdot \log \pi_\theta(y|x)]$$

This is equivalent to weighted maximum likelihood where high-reward responses receive higher weight. REST-EM avoids reward model training entirely — a ground-truth reward function (e.g., code execution for programming tasks) serves as the E-step oracle.

## SPIN: Self-Play Fine-Tuning

**SPIN** (Chen et al., 2024) frames alignment as a two-player game: the current policy plays against the previous iteration as the opponent. At iteration $t$:

- **Winning responses**: drawn from the reference data distribution $p_{\text{data}}$
- **Losing responses**: generated by the policy from the previous iteration $\pi_\theta^{(t-1)}$

The DPO objective trains $\pi_\theta^{(t)}$ to prefer reference data over its previous self:

$$\mathcal{L}_{\text{SPIN}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_{\text{data}}|x)}{\pi_\theta^{(t-1)}(y_{\text{data}}|x)} - \beta \log \frac{\pi_\theta(y_{\text{gen}}|x)}{\pi_\theta^{(t-1)}(y_{\text{gen}}|x)}\right)\right]$$

SPIN requires no reward model and no additional human labels beyond the original SFT dataset. It drives improvement until the model's generations are indistinguishable from reference data.

## SPPO: Self-Play Preference Optimization

**SPPO** (Wu et al., 2024) models the alignment problem as finding the Nash equilibrium of a preference game. At each iteration, the current policy generates responses that are compared against a mixture of its own samples:

1. Sample multiple responses for each prompt
1. Compute pairwise preferences using a reward model or judge
1. Update via a Nash-equilibrium-convergent update rule

SPPO converges to the optimal policy under the Bradley-Terry preference model and avoids the reward hacking dynamics of standard RL.

## Comparison of Online Preference Methods

| Method | Reward Model Needed? | Value Model? | Key Strength | Key Weakness |
| --- | --- | --- | --- | --- |
| PPO-RLHF | Yes | Yes | Strongest alignment | Complex, unstable, memory-heavy |
| RLOO | Yes | No | Simple policy gradient | Still needs reward model |
| Online DPO | Yes (as oracle) | No | DPO simplicity + on-policy | Reward model still needed |
| Iterative DPO | Optional (LLM-judge) | No | Self-improving without extra data | LLM-judge noise |
| REST-EM | Ground-truth reward | No | Strong for verifiable tasks | Needs oracle reward function |
| SPIN | No | No | No reward model at all | Convergence to data distribution |
| SPPO | Yes | No | Nash equilibrium guarantee | Reward model dependence |

## When to Use Online Methods

Online RLHF provides the largest gains when:

- **Reward hacking is observed**: static datasets lead to exploitation of known preference patterns
- **Iterative improvement is possible**: multiple rounds of generation and feedback are feasible
- **Verifiable rewards exist**: code execution, math solvers, or other automatic oracles provide ground-truth reward (enabling REST-EM without a learned reward model)
- **The policy distribution has shifted significantly** from the offline data collection policy

Offline DPO remains the default for most fine-tuning tasks due to its simplicity, but online methods close the performance gap on tasks where the evaluation distribution is well-defined.

## Summary

Online RLHF methods keep preference data aligned with the evolving model policy, overcoming the distributional shift inherent in static offline datasets:

- **PPO-RLHF** provides the strongest alignment signal at the cost of training complexity and memory
- **Online DPO** retains DPO's simplicity while solving the off-policy data problem
- **Iterative DPO / SPIN** enable self-improvement without a separate reward model
- **REST-EM** is the method of choice when automatic reward oracles are available (code, math, factual QA)

The field is converging toward hybrid approaches: simpler objectives (DPO-style losses) combined with online data collection (iterative generation), eliminating the PPO complexity while maintaining the on-policy distribution benefit.
