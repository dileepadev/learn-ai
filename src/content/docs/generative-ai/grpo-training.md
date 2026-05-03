---
title: GRPO — Group Relative Policy Optimization
description: Understand Group Relative Policy Optimization (GRPO), the RL training algorithm behind DeepSeek-R1 and DeepSeek-R1-Zero. Covers the GRPO objective, how it eliminates the value model required by PPO, group-based advantage estimation, reward modeling for math and code, and training recipes for reasoning models.
---

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm for fine-tuning large language models that was developed by the DeepSeek team and made famous through its use in **DeepSeek-R1-Zero** — a model that learned sophisticated chain-of-thought reasoning purely through RL, without any supervised fine-tuning on reasoning traces.

GRPO's key contribution is eliminating the **value model (critic)** required by PPO-based RLHF. Training a separate value model at LLM scale is expensive and unstable. GRPO replaces it with a group-based baseline: sample a group of outputs for each prompt, score them all, and compute advantages relative to the group mean. This self-critic approach is simpler, more memory-efficient, and empirically comparable or superior to PPO for reasoning tasks.

## Why Not PPO?

Standard **Proximal Policy Optimization (PPO)** for LLM fine-tuning requires four models in memory simultaneously:

1. **Policy** $\pi_\theta$ (the model being trained)
2. **Reference policy** $\pi_{ref}$ (frozen base model, for KL penalty)
3. **Reward model** $r_\phi$ (trained to score outputs)
4. **Value model** $V_\psi$ (critic that estimates expected future reward)

The value model must be the same size as the policy to provide good baselines — doubling GPU memory requirements and adding a second training loop. Value function estimation is also notoriously unstable at LLM scale: the model must generalize from sparse terminal rewards to dense token-level value estimates.

## The GRPO Objective

GRPO samples a **group** of $G$ outputs $\{o_1, o_2, \ldots, o_G\}$ for each prompt $q$ from the old policy $\pi_{\theta_{old}}$, then computes rewards $\{r_1, r_2, \ldots, r_G\}$ using a reward function. The **group-normalized advantage** for output $i$ is:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

The GRPO policy gradient objective (with clipping and KL regularization) is:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D},\, \{o_i\} \sim \pi_{\theta_{old}}}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\!\left(\rho_{i,t} \hat{A}_i,\; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i\right) - \beta\, D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

where $\rho_{i,t} = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}$ is the per-token importance ratio, $\epsilon$ is the clip ratio, and $\beta$ is the KL penalty coefficient.

The KL divergence term prevents the policy from drifting too far from the reference model, preserving general capabilities while acquiring new reasoning behaviors.

## Implementation

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable

@dataclass
class GRPOConfig:
    group_size: int = 8           # G: outputs sampled per prompt
    clip_ratio: float = 0.2       # epsilon: PPO-style probability ratio clipping
    kl_coef: float = 0.04         # beta: KL penalty coefficient
    max_new_tokens: int = 2048
    temperature: float = 1.0      # sampling temperature for group rollouts
    top_p: float = 0.95


class GRPOTrainer:
    """
    GRPO trainer for LLM reasoning fine-tuning.
    
    Key differences from PPO:
    - No value model: advantage estimated from group reward statistics
    - Group sampling: G completions per prompt, scored independently
    - Normalized advantage: z-score within the group
    - Token-level averaging: loss averaged over tokens, then outputs, then batch
    """
    
    def __init__(
        self,
        policy_model,           # model being trained
        ref_model,              # frozen reference model
        reward_fn: Callable,    # reward(prompt, completion) -> float
        tokenizer,
        config: GRPOConfig
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.config = config

    @torch.no_grad()
    def rollout(self, prompts: list[str]) -> dict:
        """
        Sample G completions per prompt from current policy.
        Returns completions, token log-probs, and rewards.
        """
        cfg = self.config
        all_completions = []
        all_logprobs = []
        all_rewards = []
        
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            
            # Sample G completions
            outputs = self.policy.generate(
                input_ids.repeat(cfg.group_size, 1),
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            completions = self.tokenizer.batch_decode(
                outputs.sequences[:, input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Log-probabilities of each generated token under old policy
            # outputs.scores: list of (G, vocab_size) tensors per step
            token_logprobs = []
            for step, scores in enumerate(outputs.scores):
                generated_ids = outputs.sequences[:, input_ids.shape[1] + step]
                step_logprobs = F.log_softmax(scores, dim=-1)
                token_logprobs.append(
                    step_logprobs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
                )
            # (G, max_new_tokens) — padded positions will be masked later
            logprobs_tensor = torch.stack(token_logprobs, dim=1)
            
            # Score each completion with reward function
            rewards = [
                self.reward_fn(prompt, completion) for completion in completions
            ]
            
            all_completions.append(completions)
            all_logprobs.append(logprobs_tensor)
            all_rewards.append(torch.tensor(rewards, dtype=torch.float32))
        
        return {
            "completions": all_completions,
            "logprobs_old": all_logprobs,       # token log-probs under π_old
            "rewards": all_rewards
        }

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Group-normalized advantage: z-score within each prompt's group.
        
        rewards: (G,) tensor of scalar rewards for one prompt's group
        Returns: (G,) normalized advantages
        """
        mean = rewards.mean()
        std = rewards.std() + 1e-8   # avoid division by zero
        return (rewards - mean) / std

    def compute_grpo_loss(
        self,
        policy_logprobs: torch.Tensor,   # (G, T) current policy log-probs
        old_logprobs: torch.Tensor,       # (G, T) old policy log-probs  
        ref_logprobs: torch.Tensor,       # (G, T) reference policy log-probs
        advantages: torch.Tensor,         # (G,) normalized group advantages
        attention_mask: torch.Tensor      # (G, T) mask for non-padding tokens
    ) -> torch.Tensor:
        """
        GRPO loss: clipped importance-ratio loss + KL penalty.
        Averaged per token, then per output, then across group.
        """
        cfg = self.config
        
        # Per-token importance ratios
        log_ratio = policy_logprobs - old_logprobs   # (G, T)
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate objective (PPO-style)
        A = advantages.unsqueeze(1)   # (G, 1) → broadcasts to (G, T)
        surrogate_unclipped = ratio * A
        surrogate_clipped = ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * A
        policy_loss = -torch.min(surrogate_unclipped, surrogate_clipped)
        
        # KL penalty: D_KL(π_θ || π_ref) ≈ log(π_θ/π_ref) (forward KL approximation)
        kl_penalty = policy_logprobs - ref_logprobs   # (G, T)
        
        # Combined per-token loss
        token_loss = policy_loss + cfg.kl_coef * kl_penalty   # (G, T)
        
        # Mask padding tokens and average over valid tokens
        masked_loss = (token_loss * attention_mask).sum(dim=1)
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        per_output_loss = masked_loss / token_counts   # (G,)
        
        return per_output_loss.mean()   # scalar
```

## Reward Functions for Reasoning

The reward function is the only supervision signal in GRPO training. For **mathematical reasoning**, two reward signals are commonly used:

```python
import re
import sympy

def math_accuracy_reward(prompt: str, completion: str) -> float:
    """
    Outcome-based reward: 1.0 if final numerical answer matches ground truth.
    
    DeepSeek-R1-Zero uses this as the primary reward for math tasks.
    The model learns to show work (chain-of-thought) as an instrumental
    strategy to reach the correct final answer — it is NOT supervised on CoT.
    """
    # Extract ground truth from prompt metadata (in practice, from dataset)
    ground_truth = extract_ground_truth(prompt)
    
    # Extract answer from completion (model should use <answer>...</answer> tags)
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if not answer_match:
        return 0.0
    
    predicted = answer_match.group(1).strip()
    
    try:
        # Symbolic comparison: "3/4" == "0.75" → True
        return 1.0 if sympy.simplify(predicted + " - " + ground_truth) == 0 else 0.0
    except Exception:
        return 1.0 if predicted == ground_truth else 0.0


def format_compliance_reward(completion: str) -> float:
    """
    Format reward: small positive signal for using expected output structure.
    Encourages the model to separate reasoning from final answer.
    
    DeepSeek-R1-Zero learns to produce <think>...</think><answer>...</answer>
    structure without being explicitly trained on examples using this format.
    """
    has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
    return 0.1 * (int(has_think) + int(has_answer))


def combined_reward(prompt: str, completion: str) -> float:
    """Total reward = accuracy + format bonus."""
    return math_accuracy_reward(prompt, completion) + format_compliance_reward(completion)


def code_execution_reward(prompt: str, completion: str, test_cases: list[dict]) -> float:
    """
    Reward based on passing unit tests for code generation tasks.
    Binary: passes all tests → 1.0, else fraction passed.
    
    Advantages over human evaluation:
    - Fully automated, scalable
    - Objective (pass/fail is unambiguous)
    - Resistant to reward hacking (can't game test execution)
    """
    code = extract_code_block(completion)
    if not code:
        return 0.0
    
    passed = 0
    for test in test_cases:
        try:
            exec_globals = {}
            exec(code, exec_globals)
            result = exec_globals["solution"](*test["inputs"])
            if result == test["expected"]:
                passed += 1
        except Exception:
            pass
    
    return passed / len(test_cases)
```

## GRPO vs. PPO vs. DPO

| Property | PPO | DPO | GRPO |
| --- | --- | --- | --- |
| Value model needed | Yes (same size as policy) | No | No |
| Online sampling | Yes | No (offline data) | Yes |
| Reward model needed | Yes | Implicit (in preference data) | Can use rule-based |
| KL control | Explicit penalty | Implicit via loss formulation | Explicit penalty |
| Suitable for sparse rewards | Difficult | N/A | Yes (group normalization) |
| Memory cost | ~4× policy | ~2× policy | ~2× policy + G rollouts |

## The DeepSeek-R1-Zero Phenomenon

DeepSeek-R1-Zero was trained from a base (pre-trained, non-instruct-tuned) model using GRPO with only outcome-based math rewards — no supervised fine-tuning on reasoning traces whatsoever. The model spontaneously developed:

- **Extended thinking**: allocating more tokens to harder problems
- **Self-correction**: revising answers mid-reasoning when detecting errors
- **Reflection markers**: phrases like "Wait, let me reconsider..." appearing naturally
- **Structured output**: adopting `<think>...</think>` format from format rewards alone

This demonstrated that sophisticated reasoning behaviors can emerge from RL alone, without imitation of human-authored reasoning chains — a significant finding for the field of AI reasoning and the viability of purely outcome-supervised training.
