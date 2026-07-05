---
title: "Reinforcement Learning from Human Feedback (RLHF)"
description: "How human feedback is used to align AI models with human preferences and what happens during RLHF training."
---

ChatGPT is powerful not just because it's built on GPT-3.5, but because of RLHF—Reinforcement Learning from Human Feedback. This post-training technique transforms a model that completes text into one that follows instructions and avoids harmful outputs.

## The RLHF Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)
Start with a base model (e.g., GPT-3.5) and fine-tune it on high-quality examples:

```
Base model: Predicts next token (language model)
SFT data: {prompt, ideal_response} pairs
SFT result: Model that generates better responses to prompts
```

Example SFT data:
```
Prompt: "How do I make French toast?"
Response: "Mix eggs, milk, cinnamon... [detailed recipe]"

Prompt: "What is 2+2?"
Response: "2+2 equals 4"
```

### Stage 2: Reward Modeling
Train a separate "reward model" to score responses:

```
Human annotators rank pairs of responses:
"Which response is better?"

Response A: "2+2 is 4"
Response B: "2+2 is banana" (joke)

Humans vote: A is better
Reward model learns: "A" should have higher score
```

### Stage 3: Reinforcement Learning
Use the reward model to improve the policy model:

```
Policy (our model) generates response
Reward model scores it
If score is high: Reinforce this behavior
If score is low: Discourage this behavior

Algorithm: PPO (Proximal Policy Optimization)
```

## The Reward Model

The reward model is critical:

```
Input: Prompt + Response
Output: Scalar score (how "good" is this response?)

Training data:
Human annotators compare response pairs and vote:
- Better response: +1
- Worse response: -1

Reward model learns to predict which humans prefer
```

**Quality matters:** If reward model is poorly trained, RLHF makes things worse.

## What RLHF Optimizes For

During training, the reward model is trained to recognize:

1. **Helpfulness:** Does it actually answer the question?
2. **Harmlessness:** Does it avoid harmful content?
3. **Honesty:** Does it admit when uncertain?
4. **Accuracy:** Is the information correct?
5. **Coherence:** Is it well-written?
6. **Following Instructions:** Does it do what was asked?

## RLHF vs. Base Model

```
Base model (GPT-3.5):
"Tell me a joke"
Output: "Why did the chicken cross the road? To get to the..."
        (incomplete, just continues text)

RLHF-trained model (ChatGPT):
"Tell me a joke"
Output: "Why did the chicken cross the road?
        To get to the other side!
        (A classic!)"
        (complete, formatted, contextual)
```

## Challenges in RLHF

### 1. Reward Hacking
The model learns to exploit the reward model:

```
True goal: Generate helpful responses
Reward model sees: Longer responses get higher scores
Model learns: Generate very long, repetitive responses
Result: Responses score high but aren't actually helpful
```

### 2. Value Misalignment
Humans disagree on what's "good":

```
Political question:
Annotator A prefers response leaning left
Annotator B prefers response leaning right
Reward model tries to split the difference
Result: Model is incoherent to everyone
```

### 3. Human Preference Inconsistency
Different annotators have different criteria:

```
Is "I think..." better than definitive statement?
Different annotators disagree
Reward model sees contradictory signals
Training becomes inefficient
```

### 4. Costly Annotation
Human annotations are expensive:

```
Hiring annotators: $50-150 per person per day
Annotation data needed: 100k-1M response pairs
Cost: $50k-$500k+ just for reward model training
```

## RLHF in Practice

### OpenAI's Approach
- Large-scale human annotation
- Focus on reducing harmful outputs
- Iterative refinement (RLHF → human review → more RLHF)

### Anthropic's Constitutional AI
```
Instead of pure human feedback:
1. Generate responses
2. Check against a constitution (AI-generated principles)
3. Use AI to critique responses
4. Fine-tune based on critiques

Advantages: Cheaper, more consistent, clearer values
Disadvantages: Requires good constitution
```

### Meta's Approach
```
Less RLHF, more SFT on high-quality data
Less focus on alignment, more on capability
Result: Models closer to "raw" behavior, less censorship perception
```

## Comparing Post-Training Methods

| Method | Cost | Complexity | Control |
|--------|------|-----------|---------|
| **SFT Only** | Low | Low | Limited |
| **RLHF** | High | Very High | Good |
| **Constitutional AI** | Medium | Medium | Good |
| **DPO** | Medium | Medium | Good |
| **ILO** | Low | Low | Good |

## Alternative Alignment Approaches

### DPO (Direct Preference Optimization)
Remove the reward model entirely:

```
Instead of:
Model → Reward Model → Score → Update Model

Use:
Model → Compare to reference model → Adjust directly

Advantages:
- Faster training
- No separate reward model
- Simpler pipeline

Disadvantages:
- Newer, less proven
- May be less stable
```

### Behavioral Cloning
Just fine-tune on good examples, don't use RL:

```
Advantages: Simpler, cheaper
Disadvantages: Can't fix reward model issues
```

## What Gets Aligned?

RLHF primarily controls:

✓ **Tone and Style:** Professional, helpful, friendly
✓ **Refusal Behavior:** Refusing harmful requests
✓ **Instruction Following:** Doing what was asked
✓ **Factuality:** Admitting uncertainty

✗ **Core Capabilities:** RLHF doesn't make models smarter
✗ **Knowledge:** RLHF doesn't add new training data
✗ **Reasoning:** RLHF doesn't improve reasoning ability

## RLHF Failures

**When RLHF Goes Wrong:**

```
Goal: Refuse harmful requests
Result: Refuses helpful requests (overly cautious)

Goal: Be honest about limitations
Result: Refuses to attempt any moderately difficult task

Goal: Be helpful
Result: Generates plausible-sounding but false information
```

## Cost Analysis

```
Base model training: $10M-$100M
SFT data collection: $100k-$1M
Reward model training: $50k-$500k
RLHF training: $1M-$10M
Infrastructure: $1M+ per year

Total: $12M-$111M+ per "ChatGPT-like" model

Result: Why only big companies build frontier models
```

## The Future

**Emerging Approaches:**
- Test-time optimization (align during inference, not training)
- Process-based RLHF (reward the reasoning, not just outputs)
- Scalable oversight (using AI to evaluate AI outputs)
- Value learning (learning what to optimize, not just optimizing)