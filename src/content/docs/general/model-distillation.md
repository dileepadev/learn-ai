---
title: "Model Distillation: Creating Smaller, Faster Models"
description: "How to train smaller models to mimic larger ones, enabling deployment on resource-constrained devices."
---

A large model performs excellently but is too slow and expensive for production. A small model is fast but inaccurate. Knowledge distillation teaches the small model to behave like the large one, getting the best of both worlds.

## How Distillation Works

**Standard Training:**
```
Large, expensive data → Train model → Good performance
```

**Knowledge Distillation:**
```
Large, expensive data → Train large "teacher" model → Good performance
                             ↓
                        Use teacher to generate labels
                             ↓
Cheaper, unlabeled data + teacher labels → Train small "student" model
                             ↓
Result: Small model that mimics teacher
```

## Why It Works

The teacher model learns rich representations. The student doesn't need to rediscover everything; it learns to copy the teacher's behavior.

**Example:**
- Teacher sees image of cat, identifies it as "cat" with 95% confidence
- Student learns not just "it's a cat" but also the probability distribution (cat 95%, tiger 3%, leopard 2%)
- This "soft target" provides more learning signal than hard labels

## Temperature in Distillation

Remember temperature from earlier? It's critical for distillation:

```
Without distillation temperature:
- Teacher: cat 95%, tiger 3%, leopard 2%
- Student: try to output same, but loses nuance

With high distillation temperature (4-5):
- Teacher becomes "softer": cat 75%, tiger 15%, leopard 10%
- Student learns the relative importance of wrong answers
- More forgiving training
```

## Real-World Results

| Scenario | Teacher | Student | Student Size | Student Speed | Quality Drop |
|----------|---------|---------|---|---|---|
| **Image Classification** | ResNet-152 | MobileNet | 5% | 10x faster | 2-4% accuracy drop |
| **NLP Classification** | BERT-large | BERT-small | 20% | 4x faster | 1-3% accuracy drop |
| **Object Detection** | YOLO-large | YOLO-small | 8% | 15x faster | 3-5% mAP drop |
| **LLM** | GPT-4 | GPT-3.5 | 10% | 2-3x faster | 10-15% capability drop |

## Distillation Strategies

### 1. Response-Based
Distill the final output layer:

```
Teacher input → Label predictions
Student input → Try to match teacher's label distribution
Loss = KL_divergence(teacher_output, student_output)
```

Simplest approach, works well for classification.

### 2. Feature-Based
Distill intermediate representations:

```
Teacher layer 12 output → Rich features
Student layer 6 output → Should match teacher's intermediate features
```

Requires alignment between teacher and student architecture.

### 3. Relation-Based
Distill relationships between data points:

```
If teacher says "A is more similar to B than to C"
Student should learn the same relationships
```

Good for ranking and retrieval tasks.

## Practical Implementation

**Step 1: Train Teacher**
```python
# Train a large model normally
teacher = train_large_model(data, epochs=100)
```

**Step 2: Generate Soft Labels**
```python
# Use teacher to label unlabeled data
soft_labels = teacher.predict(unlabeled_data, temperature=5)
# soft_labels = probability distributions, not hard decisions
```

**Step 3: Train Student**
```python
# Train small model on soft labels
student = train_small_model(
    unlabeled_data,
    soft_labels,
    temperature=5,  # Must match teacher temperature
    alpha=0.7  # Balance between soft and hard labels
)
```

**Step 4: Fine-tune Student**
```python
# Optional: fine-tune on original hard labels
student = fine_tune(student, original_data, epochs=10)
```

## Temperature Parameter Matters

```
Temperature = 1 (normal):
- Distillation: 60% as effective
- Student struggles to match nuanced teacher

Temperature = 3-5 (recommended):
- Distillation: 100% effective
- Student learns rich behavior from teacher

Temperature = 10+ (too high):
- Information loss
- Student becomes too similar to random model
```

## When to Use Distillation

✓ **Need Speed:** Model is too slow for production
✓ **Resource Constraints:** Deploy on edge devices, mobile
✓ **Cost:** Large model is expensive to run
✓ **Have Good Teacher:** Willing to train large model first
✓ **Task is Well-Defined:** Distillation works better for well-understood tasks

✗ **Creative Tasks:** Distilling for generation/creativity often fails
✗ **No Good Teacher:** Expensive to train teacher first
✗ **Rapid Iteration:** Teacher training is expensive per change

## Distillation in LLMs

LLM distillation is more complex but increasingly common:

- **GPT-3.5 is distilled from GPT-4:** Smaller, cheaper, 85% of GPT-4 capabilities
- **Llama 2-7B was distilled:** From larger Llama 2 model
- **Phi-3 mini:** Distilled from Phi-3 models, optimized for specific tasks

LLM distillation challenges:
- Teachers are expensive (billions of tokens)
- Quality drops more dramatically (20-30%)
- Requires very careful temperature tuning

## Advanced Techniques

### Multi-Teacher Distillation
Learn from multiple teacher models simultaneously:

```
Student ← Teacher A (good at reasoning)
       ← Teacher B (good at knowledge)
       ← Teacher C (good at instruction-following)
```

Result: Student that combines strengths.

### Self-Distillation
Use the same model at different temperatures:

```
Large model + high temperature (soft)
          ↓
Train small model on soft targets
          ↓
Student that's easier to deploy
```

## Cost-Benefit Analysis

```
Scenario: Need to classify 1 billion images/year

Option A: Use large model
- Cost: $300,000/year
- Latency: 500ms per image

Option B: Distill and use small model
- Teacher training: 1 week, $500
- Student training: 2 days, $1,000
- Inference cost: $50,000/year (90% reduction)
- Latency: 50ms per image (10x faster)

Savings: $250,000 in year 1, $300,000 every year after