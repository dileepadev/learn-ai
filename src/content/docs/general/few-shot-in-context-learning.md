---
title: "Few-Shot Learning and In-Context Learning: Teaching AI Through Examples"
description: "How providing examples in the prompt allows models to learn new tasks without retraining."
---

You don't need to fine-tune a model or modify weights to teach it a new skill. Show it a few examples in the prompt and it learns through in-context learning. This is one of the most underutilized techniques in prompt engineering.

## In-Context Learning Basics

**The surprising fact:** LLMs can learn new patterns from examples in a single prompt.

```
Few-shot prompt (no training required):
"Here are examples of product reviews classified as helpful or not:

Example 1:
Review: 'This product changed my life!'
Label: Helpful

Example 2:
Review: 'Pretty good, does what it says'
Label: Helpful

Example 3:
Review: 'Waste of money'
Label: Not Helpful

Now classify this review:
Review: 'Great quality for the price'
Label: ?"

Model outputs: Helpful
```

No retraining. No fine-tuning. Just examples in the prompt.

## Zero-Shot vs. Few-Shot vs. Many-Shot

### Zero-Shot
No examples, just a description:

```
Classify this review as Helpful or Not Helpful: "Great quality"
Expected accuracy: 65-75%
```

### Few-Shot (2-8 examples)
Include a few examples:

```
[Show 2-4 examples]
Classify this review: "Great quality"
Expected accuracy: 75-85%
```

### Many-Shot (16-32+ examples)
Include many examples:

```
[Show 20 examples]
Classify this review: "Great quality"
Expected accuracy: 85-92%
```

## Performance vs. Examples

```
Accuracy
    ↑
    |     ╱╱
 90%|    ╱╱
    |   ╱╱
 80%|  ╱╱
    | ╱╱
 70%|╱╱
    |
    └─────────────────→
      0   4    8   16  Examples
```

More examples generally improve performance, but with diminishing returns.

## When In-Context Learning Works Best

✓ **Pattern Recognition Tasks:**
- Classification
- Extraction
- Formatting

✓ **Consistent Examples:**
- Clear patterns in examples
- High-quality labels
- Representative of real data

✓ **Well-Defined Tasks:**
- Specific output format
- Clear decision boundaries
- Examples that remove ambiguity

## When In-Context Learning Fails

✗ **Complex Reasoning:** Model struggles to learn deep logic from examples
✗ **Rare Patterns:** Only a few examples of edge case (model might not learn)
✗ **Domain-Specific Knowledge:** Requires expertise not in training data
✗ **Multi-Step Tasks:** Complex workflows with many steps

## Example Quality Matters

**Bad Examples:**
```
Example 1: input="hello", output="goodbye"
Example 2: input="good morning", output="thank you"

Pattern unclear; model can't reliably learn
```

**Good Examples:**
```
Example 1:
Sentiment: "I love this!"
Label: Positive

Example 2:
Sentiment: "Not great, disappointed"
Label: Negative

Clear pattern; model learns easily
```

## Strategies to Improve In-Context Learning

### 1. Example Diversity
Include examples that cover different cases:

```
Good:
- Positive example (clear signal)
- Negative example (clear counter-signal)
- Edge case (borderline decision)
- Extreme case (very positive or very negative)
```

### 2. Example Ordering
Order matters:

```
Psychology effect: Models are influenced by recent examples
Put good examples last (recency bias helps you)
Put difficult examples in the middle
Put diverse examples throughout
```

### 3. Explanation in Examples
Include reasoning:

```
Without explanation:
Input: "Good product" → Output: Positive

With explanation:
Input: "Good product"
Reasoning: Adjective "good" is positive
Output: Positive
```

The explanation helps model learn the underlying pattern.

### 4. Correct Mistakes Explicitly
If model makes errors, show examples of corrections:

```
"The model incorrectly classified 'decent' as negative.
'Decent' is positive. Here's an example:

Input: "Decent quality, works as described"
Correct classification: Positive
Reason: Even mild positive language indicates satisfaction
"
```

## Many-Shot Learning Breakthrough

Recent research shows many-shot learning (50-100+ examples) is surprisingly effective:

```
Zero-shot: Model uses training knowledge
Few-shot: Model combines training knowledge + examples
Many-shot: Model mostly follows examples, forgets some training knowledge

Effect: Sometimes many-shot outperforms fine-tuning for specific domains
```

## Implementation Tips

### Prompt Structure
```
[Task Description]

[Examples with clear formatting]

Now solve:
[User query]
```

### Format Consistency
```
BAD (inconsistent):
Example 1: Input: X Output: Y
Example 2: Input=A Output=B

GOOD (consistent):
Example 1:
Input: X
Output: Y

Example 2:
Input: A
Output: B
```

### Delimiter Clarity
```
Use clear delimiters:
---
Input: X
Output: Y
---
```

Helps model parse examples reliably.

## Cost-Performance Tradeoff

```
Zero-shot: 100 tokens + prompt
Accuracy: 70%
Cost: Low

Few-shot (4 examples): 500 tokens + prompt
Accuracy: 85%
Cost: 5x higher

Many-shot (50 examples): 3000 tokens + prompt
Accuracy: 92%
Cost: 30x higher
```

## When to Use In-Context Learning vs. Fine-Tuning

| Factor | In-Context | Fine-Tuning |
|--------|-----------|-------------|
| **Setup Time** | Minutes | Hours/Days |
| **Data Required** | 2-50 examples | 100s-1000s |
| **Cost Per Task** | Higher | Lower at scale |
| **Flexibility** | Easy to change | Hard to change |
| **Performance** | 85-92% | 90-95%+ |
| **When to Use** | Prototyping, testing | Production, stable tasks |

## Real-World Example

**Task:** Extract key information from customer emails

```
Few-shot prompt:

Extract the following from each customer email:
- Issue Type
- Urgency
- Requested Action

Example 1:
Email: "My order #123 never arrived! I need it for tomorrow's event!"
Issue Type: Missing Order
Urgency: High
Requested Action: Expedite replacement

Example 2:
Email: "Just wondering if you have blue in size L?"
Issue Type: Product Inquiry
Urgency: Low
Requested Action: Provide product availability

Example 3:
Email: "The product broke after one week. Not impressed."
Issue Type: Defective Product
Urgency: Medium
Requested Action: Replacement or refund

Now extract from this email:
Email: "I've been trying to reach support for a week with no response!"
[Model generates: Issue Type: Support Access, Urgency: High, Action: Escalate to manager]
```

## Limitations

- Models sometimes ignore examples if training data conflicts
- Performance plateau (can't improve beyond ~92-95% with examples alone)
- Large prompt size increases cost and latency
- Examples must fit in context window