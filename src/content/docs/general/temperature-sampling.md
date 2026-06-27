---
title: "Temperature and Sampling: Controlling AI Model Creativity"
description: "How temperature, top-k, and top-p sampling parameters control the randomness and diversity of AI outputs."
---

Set temperature to 0 and your model becomes robotic and deterministic. Set it to 2.0 and it becomes chaotic and unpredictable. Temperature is a critical parameter for controlling model behavior, but most developers don't understand it.

## What Is Temperature?

Temperature controls how "confident" the model is when choosing the next token. Technically, it scales the logits (raw model outputs) before converting them to probabilities.

**Formula:**
```
probability = softmax(logits / temperature)
```

## Visual Explanation

At each step, the model calculates probabilities for possible next tokens:

```
Token options: ["the" (50%), "a" (30%), "this" (20%)]

Temperature 0.1 (very cold):
→ ["the" (99.9%), "a" (0.05%), "this" (0.05%)]
→ Always picks "the"

Temperature 1.0 (normal):
→ ["the" (50%), "a" (30%), "this" (20%)]
→ Follows original probabilities

Temperature 2.0 (very hot):
→ ["the" (20%), "a" (25%), "this" (27%), other (28%)]
→ Much more randomness
```

## Temperature Guidelines

| Temperature | Use Case | Behavior |
|-------------|----------|----------|
| 0.0 | Deterministic output | Always picks highest probability token; not truly random |
| 0.3-0.5 | Fact-based tasks | Very reliable; minimal variation |
| 0.7-0.9 | Default/balanced | Good mix of consistency and creativity |
| 1.0 | General purpose | Standard LLM behavior |
| 1.2-1.5 | Creative tasks | More variety while staying coherent |
| 2.0+ | Highly experimental | Often produces nonsense; rarely useful |

## Real-World Examples

**Q&A System (Temperature 0.2):**
```
Input: "What is 2+2?"
Output 1: "2+2 equals 4"
Output 2: "2+2 equals 4" (identical—good for factual tasks)
```

**Creative Writing (Temperature 1.2):**
```
Input: "Write a short poem about rain"
Output 1: "Soft drops fall from cloudy skies..."
Output 2: "Pitter-patter on the window, nature's gentle song..."
(Different each time—good for creativity)
```

## Beyond Temperature: Top-K and Top-P

Temperature alone isn't enough. You also need **Top-K** and **Top-P sampling**.

### Top-K Sampling
Only sample from the K most likely next tokens.

```
Possible tokens: ["the", "a", "this", "dog", "cat", ...]
With Top-K=5: Only consider ["the" (50%), "a" (30%), "this" (15%), "dog" (3%), "cat" (2%)]
With Top-K=1: Only "the" is possible (same as temperature 0)
```

**Effect:** Removes the "long tail" of unlikely tokens, reducing weird outputs.

### Top-P (Nucleus Sampling)
Only sample from tokens whose cumulative probability exceeds P.

```
Tokens sorted by probability:
- "the" (50%) → cumulative 50%
- "a" (30%) → cumulative 80%
- "this" (15%) → cumulative 95%
- "dog" (3%) → cumulative 98%

Top-P=0.95:
→ Include tokens until 95% cumulative probability
→ Use ["the", "a", "this"] but exclude "dog" and beyond
```

**Effect:** More dynamic than Top-K (adapts to the distribution).

## Practical Configuration

**For deterministic output:**
```
temperature = 0.0
```

**For balanced quality:**
```
temperature = 0.7
top_p = 0.9
```

**For creative output:**
```
temperature = 1.0
top_p = 0.95
top_k = 40
```

**For controlled randomness (recommended):**
```
temperature = 0.7
top_p = 0.95
(Top-K usually not needed if using Top-P)
```

## Common Mistakes

1. **Using temperature 1.0 and thinking you have no randomness** — There's still randomness; try temperature 0 to see the difference
2. **Setting temperature too high for production** — Temperature 1.5+ produces unreliable outputs
3. **Not testing different settings** — What works for one task might fail for another
4. **Forgetting about reproducibility** — Temperature 0 is required for reproducible outputs (but other factors still affect it)