---
title: "In-Context Learning: LLMs Learn from Examples Without Weight Updates"
description: "Understand how LLMs can learn new tasks from examples in the prompt — from few-shot prompting to demonstration selection and the theoretical foundations of in-context learning."
---

In-context learning (ICL) is one of the most remarkable properties of large language models: they can perform new tasks by simply seeing examples in their context, without any gradient updates or fine-tuning. This capability makes LLMs remarkably flexible.

## What Is In-Context Learning?

In-context learning is the ability of an LLM to adapt its behavior based on examples provided in the prompt:

```python
# Zero-shot: No examples
prompt = "Translate to French: Hello, how are you?"
# → "Bonjour, comment ça va?"

# One-shot: One example
prompt = """Translate to French: Hello, how are you? → Bonjour, comment ça va?
Translate to French: Good morning! → """

# Few-shot: Multiple examples
prompt = """Translate to French:
Hello, how are you? → Bonjour, comment ça va?
Good morning! → Bonjour!
Where is the train station? → Où est la gare?
Nice to meet you → """
```

The model performs the task based purely on the pattern in the examples — no weights are updated.

## Why In-Context Learning Works

### Surface Learning vs. Gradient-Based Learning

Traditional ML requires updating weights to learn. In-context learning appears to work differently:

1. **Gradient-based learning**: Adjust weights to minimize loss on training data.
2. **In-context learning**: Use the forward pass to "condition" the model on examples.

### Theoretical Perspectives

**Bayesian Inference View**: The model infers a latent concept from examples and uses that concept to generate outputs. This explains why more examples generally improve performance — more evidence for the Bayesian posterior.

**Meta-Learning View**: The model has learned, during pretraining, a broad meta-learning capability. It can "learn to learn" from examples because it has seen millions of task-example-output triplets.

**Linear Representation View**: Recent research suggests that examples are linearly encoded in the model's representations, allowing the model to extract task vectors that modify its behavior.

## Designing Effective Demonstrations

### Example Selection

Not all examples are equally useful. Key considerations:

```python
# Good examples for sentiment classification
demonstrations = [
    ("The movie was absolutely fantastic!", "Positive"),
    ("I didn't enjoy this at all. Boring and slow.", "Negative"),
    ("Decent film, worth watching once.", "Neutral"),
]
```

**Principles for example selection:**

1. **Diversity**: Cover the range of input/output patterns.
2. **Accuracy**: Examples should be correct — the model will learn wrong patterns too.
3. **Representativeness**: Examples should represent typical inputs the model will see.
4. **Length**: Match the typical length of real queries.

### Example Ordering

The order of examples can significantly impact performance:

- **Voting patterns**: Some tasks are sensitive to example order.
- **Hallucination propagation**: Incorrect examples early in the context can lead to errors.
- **Recency effect**: The model often pays more attention to recent examples.

### Label Selection

For classification tasks, the choice of labels matters:

```python
# Binary sentiment — good
labels = ["Positive", "Negative"]

# Too granular — may confuse the model
# labels = ["Very Positive", "Positive", "Slightly Positive", 
#           "Neutral", "Slightly Negative", "Negative", "Very Negative"]

# Arbitrary labels — works fine
labels = ["A", "B"]
```

## Types of In-Context Learning

### Zero-Shot Learning
No examples provided. The model must infer the task from the instruction.

```python
prompt = """Extract the names of all people mentioned in the text below.
Text: John and Mary went to the store. They met Sarah there.
People:"""
```

### One-Shot Learning
One example demonstrates the task.

```python
prompt = """Extract people names:
Text: John went to the store.
People: John

Text: Sarah bought an apple.
People:"""
```

### Few-Shot Learning
Multiple examples (typically 2–20) demonstrate the task.

### K-Nearest Neighbors In-Context (KNN-ICL)
Retrieve the most similar examples from a database based on the current query:

```python
def knn_icl(query, example_database, k=8):
    # Embed query and examples
    query_emb = embed(query)
    example_embs = [embed(ex) for ex in example_database]
    
    # Find nearest neighbors
    neighbors = find_nearest(query_emb, example_embs, k=k)
    
    # Construct prompt with retrieved examples
    prompt = format_demonstrations(neighbors) + "\n" + query
    return generate(prompt)
```

## In-Context Learning Failures

### Sensitivity to Formatting
Small changes in how examples are formatted can drastically change performance:

```python
# Works well
"Q: What is 2+2? A: 4\nQ: What is 3+3? A:"

# May fail
"Q: What is 2+2? Answer: 4\nQ: What is 3+3? Answer:"
```

### Label Noise
ICL learns from examples regardless of whether they're correct:

```python
# Model will learn the wrong pattern
demonstrations = [
    ("Input: The food was good.", "Label: Negative"),  # Wrong!
    ("Input: The food was bad.", "Label: Positive"),   # Wrong!
]
```

### Compositional Complexity
ICL struggles with tasks that require combining multiple concepts:

```python
# Easy for ICL
"Sentiment: The food was great. → Positive"

# Harder (compositional)
"First sentiment: good. Second sentiment: bad. Overall: neutral"
# The model may not understand the composition rule.
```

## Optimizing In-Context Learning

### Demonstration Engineering
Carefully designing the set and format of demonstrations:

```python
def engineer_demonstrations(task, candidate_examples, metric='accuracy'):
    """Select optimal subset of demonstrations."""
    from itertools import combinations
    
    best_score = 0
    best_subset = None
    
    # Try all combinations up to size k
    for subset in combinations(candidate_examples, k=8):
        score = evaluate_icp(task, subset, metric)
        if score > best_score:
            best_score = score
            best_subset = subset
    
    return best_subset
```

### Calibrating ICL Outputs
ICL outputs can be biased by label distribution in examples:

```python
def calibrate_predictions(outputs, example_labels):
    """Adjust for label bias in demonstrations."""
    label_counts = Counter(example_labels)
    label_prior = {l: c / len(example_labels) for l, c in label_counts.items()}
    
    # Adjust output probabilities by inverse prior
    calibrated = {
        label: prob / (label_prior.get(label, 0.01) + 1e-8)
        for label, prob in outputs.items()
    }
    return normalize(calibrated)
```

## In-Context Learning vs. Fine-Tuning

| Aspect | In-Context Learning | Fine-Tuning |
|--------|---------------------|-------------|
| **Data needs** | A few examples | Thousands of examples |
| **Speed** | Instant | Requires training |
| **Memory** | None | Stores weights |
| **Flexibility** | Switch tasks by changing prompt | Requires retraining |
| **Performance** | Limited by context | Can exceed ICL |
| **Catastrophic forgetting** | None | Risk during fine-tuning |

## The Future of In-Context Learning

Research is pushing the boundaries of ICL:

- **Longer contexts**: 1M+ token contexts enable thousands of examples.
- **Retrieval-augmented ICL**: Retrieve relevant examples from large databases.
- **Adaptive ICL**: Dynamically select examples based on the query.
- **Theoretical understanding**: New frameworks explain when and why ICL works.

In-context learning is what makes LLMs so versatile. Understanding its strengths and limitations is essential for building effective LLM applications.