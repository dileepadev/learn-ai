---
title: "Reinforcement Learning from Human Feedback (RLHF)"
description: "Understand how RLHF aligns LLMs with human preferences — from preference dataset collection to reward modeling and PPO optimization, the technique behind ChatGPT and Claude."
---

Reinforcement Learning from Human Feedback (RLHF) is the key technique that transformed language models from autocomplete systems into helpful, harmless, and honest assistants. Understanding RLHF is essential for anyone building aligned AI systems.

## Why RLHF Matters

Pretrained language models learn to predict the next token on web text. This objective doesn't distinguish between helpful and harmful outputs, honest and deceptive outputs, clear and confusing outputs. RLHF injects human values into the model by training it to maximize a learned "preference" rather than just next-token probability.

## The Three Stages of RLHF

### Stage 1: Supervised Fine-Tuning (SFT)

Before RLHF, the model is fine-tuned on high-quality demonstrations of desired behavior:

```python
sft_data = [
    {
        "prompt": "How do I bake bread?",
        "response": """Here's a simple bread recipe:\n\n
        Ingredients:\n
        - 500g bread flour\n
        - 7g instant yeast\n
        - 10g salt\n
        - 325ml warm water\n\n
        Steps:\n
        1. Mix flour, yeast, and salt in a bowl.\n
        2. Add water and mix until a shaggy dough forms.\n
        3. Knead for 10 minutes until smooth.\n
        4. Let rise for 1 hour.\n
        5. Shape and bake at 220°C for 25 minutes.\n\n
        [continues with detailed instructions...]"""
    },
    # ... thousands of high-quality examples
]
```

This stage teaches the model the *format* and *style* of good responses. The model learns to write helpful, well-structured answers.

### Stage 2: Reward Model Training

A separate **reward model (RM)** is trained to predict human preferences:

1. **Collect preference annotations**: Humans rank multiple model responses to the same prompt.
2. **Train the RM**: Given a prompt and two responses, predict which humans preferred.

```python
# Preference dataset format
preference_data = [
    {
        "prompt": "Explain quantum computing",
        "responses": [
            "Quantum computing uses qubits that can be 0, 1, or both at once. This lets it solve certain problems faster.",
            "Quantum computers process information using quantum bits, which leverage superposition and entanglement to perform calculations in parallel, enabling exponential speedups for specific algorithms."
        ],
        "choice": 1,  # Humans preferred response 2
        "rankings": [2, 1]  # Response 2 > Response 1
    },
    # ... tens to hundreds of thousands of examples
]

# Reward model training
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "EleutherAI/gpt-neo-1.3B",
    num_labels=1,
)
# Train to predict: RM(response) > RM(rejected_response)
```

The reward model learns to score responses according to human preferences. It's typically a smaller model (1B–6B parameters) trained on the preference data.

### Stage 3: RL Optimization

The language model is optimized to maximize the reward model score using reinforcement learning:

1. **Generate responses**: The LLM produces outputs for given prompts.
2. **Score with RM**: The reward model gives each response a score.
3. **Update with PPO**: Proximal Policy Optimization updates the LLM to increase high-scoring responses.

```python
from trl import PPOTrainer, PPOConfig

config = PPOConfig(
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=reference_model,  # KL penalty against reference
    reward_model=reward_model,
    tokenizer=tokenizer,
)

# Generate and score
query_tensor = tokenizer(prompt, return_tensors="pt").input_ids
response = model.generate(query_tensor, **generation_kwargs)
reward = reward_model(response).score

# PPO update
ppo_trainer.step([query_tensor], [response], [reward])
```

The **KL divergence penalty** between the policy and the original SFT model prevents the model from drifting too far from its helpful responses to just maximize the reward model.

## Common Challenges

### Reward Hacking
The model finds ways to get high reward without actually being helpful:
- Producing short, vague responses that the reward model can't penalize.
- Exploiting biases in the reward model.

Mitigation: Diverse preference data, RL from AI feedback (RLAIF), and careful reward model evaluation.

### Training Instability
PPO is notoriously finicky. Common issues:
- **Exploding gradients**: Clip or normalize gradients.
- **Reward model exploitation**: Use multiple reward models or adversarial evaluation.
- **KL collapse**: The KL penalty is too strong, preventing learning.

### Preference Data Quality
Human annotator agreement can be low, especially for nuanced questions. Strategies:
- Clear annotation guidelines.
- Multiple annotators per sample with agreement checks.
- Expert annotators for specialized domains.

## Alternatives to RLHF

### Direct Preference Optimization (DPO)
DPO directly optimizes on preference pairs without training a separate reward model:

```python
# DPO loss — no reward model needed
def dpo_loss(policy_chosen, policy_rejected, beta=0.1):
    log_probs_chosen = log_prob(policy_chosen)
    log_probs_rejected = log_prob(policy_rejected)
    
    logits = beta * (log_probs_chosen - log_probs_rejected)
    return -F.log_sigmoid(logits).mean()
```

Simpler and more stable than PPO, and often achieves comparable results.

### Odds Ratio Policy Optimization (ORPO)
Combines SFT and preference optimization in a single stage, avoiding the multi-stage complexity of RLHF.

## The Future of RLHF

- **RLAIF (RL from AI Feedback)**: Use LLMs to generate preference data instead of humans, reducing annotation costs.
- **Constitutional AI**: Train models to self-improve according to a set of principles, reducing human annotation needs.
- **Online RLHF**: Continuously update the model based on user feedback signals (thumbs up/down) rather than static datasets.

RLHF remains the foundation of modern AI alignment. Understanding its mechanics — preference collection, reward modeling, and RL optimization — is essential for building models that are not just capable, but aligned with human values.