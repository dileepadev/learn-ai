---
title: "In-Context Learning: Learning from Demonstrations"
description: "Understand how large language models learn from examples in the prompt without gradient updates."
date: "2026-03-20"
tags: ["deep-learning", "llms", "prompting", "few-shot"]
---

In-context learning (ICL) is the ability of large language models to learn from examples provided in the prompt. The model adapts its behavior based on these demonstrations without any weight updates — making it a form of "learning without learning."

## What Is In-Context Learning?

Given a prompt containing examples and a new query, the model produces the correct output:

```python
# Demonstration examples
examples = [
    ("The cat sat on the mat", "cat: noun, sat: verb, mat: noun"),
    ("She runs quickly", "she: pronoun, runs: verb, quickly: adverb"),
]

# New query
query = "The bird flew high"

# Model completes the pattern
output = model.generate(f"{examples}\n\nQuery: {query}\nOutput:")
# Expected: "bird: noun, flew: verb, high: adverb"
```

The key observation: model weights remain unchanged throughout.

## Mechanisms Behind In-Context Learning

### Gradient-Based Perspective

Recent work suggests ICL performs implicit gradient descent. Each attention head effectively computes:

```
∇θ L(θ, examples) → the model "computes gradients" through attention
```

This means the model's forward pass implements a form of meta-learning.

### Demonstration as Conditioning

The examples modify the model's internal state, which conditions generation:

```python
def in_context_forward(model, examples, query):
    # Encode examples
    example_embeddings = [encode(example) for example in examples]
    
    # Attention over examples creates weighted representation
    # This representation influences how query is processed
    context = weighted_attention(query, example_embeddings)
    
    return model.generate(context + query)
```

## Types of In-Context Learning

### Zero-Shot Learning

No examples provided, just instructions:

```python
prompt = """Classify the sentiment of this review as positive or negative.

Review: This movie was fantastic!
Sentiment:"""
```

### One-Shot Learning

Single example provided:

```python
prompt = """Classify the sentiment.

Example:
Review: This movie was terrible.
Sentiment: negative

Review: This movie was fantastic!
Sentiment:"""
```

### Few-Shot Learning

Multiple examples (typically 2-32):

```python
prompt = """Classify the sentiment.

Review: The plot was confusing.
Sentiment: negative
Review: Great acting and direction.
Sentiment: positive
Review: I fell asleep halfway through.
Sentiment: negative
Review: An absolute masterpiece.
Sentiment:"""
```

## Factors Affecting ICL Performance

### Example Quality

```python
# Good examples: diverse, correct, representative
good_examples = [
    ("Simple sentence", "Correct parsing"),
    ("Complex sentence", "Correct parsing"),
    ("Edge case", "Correct parsing"),
]

# Bad examples: noisy, inconsistent, incorrect
bad_examples = [
    ("Sentence", "Wrong parsing"),  # Inconsistent format
    ("Sentence", "Correct"),         # Different output format
]
```

### Example Order

Recent work shows examples at the end of the prompt have more influence. Also, the relative order matters — mixing easy and hard examples can help.

### Model Size Effect

ICL capability strongly scales with model size:
- Models below ~6B parameters show limited ICL
- Capability emerges gradually, then sharply around 100B+ parameters

## Improving ICL Performance

### Example Selection

```python
def select_best_examples(query, candidate_examples, embedding_model, k=4):
    """Select examples most similar to the query."""
    query_emb = embedding_model.encode(query)
    
    similarities = []
    for ex in candidate_examples:
        ex_emb = embedding_model.encode(ex["input"])
        sim = cosine_similarity(query_emb, ex_emb)
        similarities.append((ex, sim))
    
    # Select top-k by similarity
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
```

### Example Ordering

```python
def order_examples_by_difficulty(examples, model, difficulty_metric):
    """Order examples from easy to hard or hard to easy."""
    # Score each example by "hardness"
    scored = [(ex, difficulty_metric(ex, model)) for ex in examples]
    
    # Order appropriately
    return [ex for ex, _ in sorted(scored, key=lambda x: x[1])]
```

### Calibrating ICL Outputs

```python
def calibrate_predictions(logits, example_outputs):
    """Adjust predictions based on example outputs."""
    # Compute bias from examples
    example_logits = []
    for example in examples:
        logits = model(example.input)
        example_logits.append(logits[example.output_position])
    
    # Subtract average bias
    bias = torch.mean(torch.stack(example_logits), dim=0)
    calibrated = logits - bias
    
    return calibrated
```

## Theoretical Understanding

Recent research views ICL through several lenses:

1. **Gradient Descent View:** The attention mechanism computes implicit gradients over the in-context examples.

2. **Bayesian Inference View:** The model performs Bayesian inference over a latent concept that explains the examples.

3. **Learning to Learn View:** ICL is an emergent form of meta-learning where the model has learned to learn from examples during pretraining.

In-context learning remains an active research area, with ongoing work to understand when it works, when it fails, and how to improve it systematically.