---
title: "RLHF: Reinforcement Learning from Human Feedback"
description: "Learn how RLHF aligns language models with human preferences through preference modeling and policy optimization."
date: "2026-03-20"
tags: ["deep-learning", "alignment", "reinforcement-learning", "llms"]
---

Reinforcement Learning from Human Feedback (RLHF) is the technique used to align large language models with human values and preferences. It powered models like ChatGPT, Claude, and GPT-4, enabling them to follow instructions and respond helpfully.

## The Three-Stage RLHF Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Collect demonstrations of desired behavior and fine-tune the base model:

```python
# After SFT, the model can follow instructions
# but may still produce problematic outputs
```

### Stage 2: Reward Model Training

Collect human comparisons of model outputs, train a reward model:

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Replace final layer with a scalar head
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids):
        # Get hidden states
        outputs = self.base_model(input_ids)
        # Use [CLS] token representation
        reward = self.reward_head(outputs.last_hidden_state[:, 0])
        return reward.squeeze(-1)

# Training: maximize margin between preferred and rejected responses
def compute_reward_loss(chosen_rewards, rejected_rewards):
    """Pairwise ranking loss for reward model."""
    loss = torch.nn.functional.relu(
        rejected_rewards - chosen_rewards + margin
    )
    return loss.mean()
```

### Stage 3: Policy Optimization (PPO)

Use the reward model to fine-tune the SFT model with RL:

```python
import torch.optim as optim
from collections import deque

class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, clip_epsilon=0.2):
        self.policy = policy_model
        self.ref = ref_model  # Reference model (SFT) to prevent drift
        self.reward = reward_model
        self.clip = clip_epsilon
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-5)
    
    def compute_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def ppo_update(self, states, actions, old_log_probs, rewards, dones):
        """PPO clip objective update."""
        # Get new policy outputs
        logits = self.policy(states)
        new_log_probs = log_softmax(logits).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clip objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL penalty (stay close to reference model)
        with torch.no_grad():
            ref_logits = self.ref(states)
            kl = kl_divergence(logits, ref_logits)
        
        total_loss = policy_loss + 0.01 * kl
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
```

## Complete RLHF Training Loop

```python
def train_rlhf(policy, ref_model, reward_model, dataloader, epochs=3):
    """Full RLHF training loop."""
    trainer = PPOTrainer(policy, ref_model, reward_model)
    
    for epoch in range(epochs):
        for batch in dataloader:
            # Generate responses from current policy
            responses = generate_responses(policy, batch["prompts"])
            
            # Score with reward model
            rewards = reward_model(responses)
            
            # PPO update
            loss = trainer.ppo_update(...)
            
        print(f"Epoch {epoch}: Policy optimized")
```

## KL Divergence Constraints

A critical component is preventing the policy from drifting too far from the reference model:

```python
def kl_divergence(logits1, logits2):
    """Compute KL divergence between two policy distributions."""
    probs1 = torch.softmax(logits1, dim=-1)
    probs2 = torch.softmax(logits2, dim=-1)
    return (probs1 * (torch.log(probs1) - torch.log(probs2))).sum(-1)
```

The KL penalty weight balances:
- Too low: policy may exploit reward model with unrealistic outputs
- Too high: policy update is too conservative, learning slows

## Challenges and Considerations

**Reward hacking:** Models find ways to game the reward model rather than produce genuinely good outputs.

**Human preference saturation:** As models improve, human raters may struggle to distinguish better outputs.

**Sample efficiency:** RLHF requires significant human annotation, typically 10K-100K comparisons per model.

RLHF represents a paradigm shift in how we train AI systems — moving from imitating data to optimizing for human judgment.