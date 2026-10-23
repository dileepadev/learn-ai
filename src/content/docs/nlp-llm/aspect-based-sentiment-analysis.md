---
title: Aspect-Based Sentiment Analysis
description: Learn how aspect-based sentiment analysis identifies the specific features people discuss and the sentiment toward each, and how LLMs and fine-tuned models approach the task.
---

Standard sentiment analysis assigns one label to a whole text. Aspect-based sentiment analysis (ABSA) is finer-grained: it finds *what* is being discussed and the sentiment toward each target.

> "The battery life is excellent but the screen is dim."

Overall sentiment is mixed. ABSA outputs: battery life -> positive, screen -> negative.

## Subtasks

| Subtask | Output |
|---------|--------|
| Aspect term extraction (ATE) | Spans such as "battery life" |
| Aspect category detection | Category such as BATTERY, DISPLAY |
| Aspect sentiment classification (ASC) | Polarity for a given aspect |
| Opinion term extraction | Words such as "excellent" |
| Joint tuples | (aspect, category, opinion, sentiment) quadruples |

## Approaches

**Pipeline**: extract aspects, then classify sentiment per aspect. Errors in the first stage propagate.

**Sentence-pair classification**: feed a transformer the sentence plus the aspect as a second segment and predict polarity from the [CLS] representation.

**Generative formulation**: use a sequence-to-sequence model or LLM to output structured tuples directly, handling several subtasks at once.

## LLM Prompting Example

```python
prompt = """Extract every (aspect, sentiment) pair from the review.
Sentiment must be positive, negative, or neutral. Return JSON.

Review: "Battery life is excellent but the screen is dim."
"""
# Expected:
# [{"aspect": "battery life", "sentiment": "positive"},
#  {"aspect": "screen", "sentiment": "negative"}]
```

Using structured output constraints makes parsing reliable.

## Evaluation

Common metrics are span-level F1 for extraction and accuracy and macro-F1 for polarity. Benchmarks include SemEval-2014 restaurant and laptop datasets.

## Practical Uses

- Product review analytics by feature
- Customer feedback triage for support teams
- Brand and hospitality monitoring

## Challenges

Implicit aspects ("It died after a week" refers to durability without naming it), sarcasm, comparison across entities, and domain-specific vocabulary all complicate the task. Define the aspect taxonomy carefully before labeling, since annotation consistency strongly affects results.
