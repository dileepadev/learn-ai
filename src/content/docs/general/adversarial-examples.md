---
title: "Adversarial Examples and AI Robustness"
description: "Understanding how small perturbations can fool AI models and techniques to build more robust systems."
---

Add a few pixels to an image and a confident classifier becomes completely wrong. Alter 1% of a text and sentiment analysis fails. These adversarial examples reveal a critical gap between human and AI robustness.

## The Adversarial Example Problem

### Image Example
```
Original: Stop sign (model confidence: 99%)
Add imperceptible noise:
  (pixel adjustments invisible to humans)
Modified: Speed limit sign (model confidence: 95%)

Human sees: Stop sign
AI sees: Speed limit sign
```

### Text Example
```
Original: "I love this movie!" (sentiment: positive, 99%)
Modified: "I love this movie!" (with homoglyph: I͏ love this movie!)
       [character replacement, invisible to most)
Modified sentiment: negative, 87%
```

## Why Models Are Vulnerable

1. **High-Dimensional Space:** Small perturbations in high dimensions can have outsized effects
2. **Linear Behavior:** Models sometimes rely on simple linear relationships attackers can exploit
3. **Sharp Decision Boundaries:** A model moves from "definitely class A" to "definitely class B" abruptly
4. **Lack of Robustness Training:** Models trained normally are vulnerable by default

## Threat Levels

| Threat | Impact | Example |
|--------|--------|---------|
| **Imperceptible** | High | Stop sign → Speed limit (imperceptible pixels) |
| **Semantic** | Very High | Stop sign vandalized but still recognizable as stop sign to humans |
| **Targeted** | Critical | Attacker chooses specific misclassification |
| **Untargeted** | High | Attacker just wants wrong answer (any wrong answer) |

## Attack Methods

### 1. FGSM (Fast Gradient Sign Method)
Fast, simple attack:

```
1. Feed image to model, get prediction
2. Calculate gradient of loss with respect to image pixels
3. Add small perturbation in the direction of increasing loss
4. Result: Adversarial example

Time: <1 second per image
Success rate: 70-90%
```

### 2. PGD (Projected Gradient Descent)
Iterative attack (stronger):

```
for i in range(iterations):
    1. Calculate gradient
    2. Add small perturbation
    3. Project back to valid input space
    4. Repeat

Result: Very effective but slower
```

### 3. Carlini-Wagner Attack
Optimization-based, most effective:

```
Find minimum perturbation that causes misclassification
Uses optimization algorithms to search for adversarial example

Success rate: 95-100%
Time: Minutes per image
```

### 4. Text Attacks
For NLP models:

- **Typos:** "teh" instead of "the"
- **Synonyms:** Replace words with similar-meaning synonyms
- **Character-level:** Add accents, use homoglyphs
- **Adversarial examples:** Specific word sequences that flip sentiment

## Defense Strategies

### 1. Adversarial Training
Include adversarial examples in training:

```
Normal training: Train on natural examples
Adversarial training: Train on mix of natural + adversarial examples

Result: Model learns to classify correctly even with perturbations
Cost: 2-3x longer training time
Robustness gain: Good, but not perfect
```

### 2. Defensive Distillation
Use distillation to increase robustness:

```
Teacher trained normally
Teacher predictions with high temperature (soft labels)
Student trained on soft labels

Result: Student is more robust to adversarial examples
Why: Soft labels act as regularization
```

### 3. Input Preprocessing
Clean adversarial perturbations before feeding to model:

```
Techniques:
- JPEG compression (removes pixel-level noise)
- Gaussian blurring (smooths adversarial patterns)
- Feature squeezing (reduces input precision)

Drawback: Can reduce accuracy on clean examples
```

### 4. Ensemble Methods
Use multiple models:

```
Prediction = majority vote of 10 models
Attacker would need to fool all 10 simultaneously
Much harder than fooling one model
```

### 5. Certified Defenses
Mathematically prove robustness bounds:

```
Randomized smoothing:
- Add Gaussian noise to input
- Run model multiple times
- Aggregate predictions
- Mathematically guarantee robustness within a radius

Trade-off: Slower, slight accuracy drop
Benefit: Provable guarantees
```

## Real-World Impact

**Critical Systems:**
- **Autonomous Vehicles:** Stop sign → Speed limit could cause accidents
- **Medical AI:** Slightly altered image changes diagnosis
- **Security:** Face recognition spoofed by adversarial eyeglasses
- **Content Moderation:** Adversarial examples bypass hate speech filters

**Less Critical:**
- Product recommendations
- Entertainment suggestions
- Text categorization (usually)

## Defensive Checklist

For high-stakes applications:

- [ ] Test with adversarial examples
- [ ] Use adversarial training
- [ ] Monitor for distribution shifts
- [ ] Implement ensemble predictions
- [ ] Add input validation
- [ ] Log suspicious inputs for analysis
- [ ] Have human review for edge cases

## Detection vs. Prevention

**Prevention (Ideal):**
- Make model robust to adversarial examples
- Hard and expensive

**Detection:**
- Detect when input looks adversarial
- Reject or escalate to human review
- Easier and more practical

**Example Combined Approach:**
```
Input → Adversarial detector
        → If suspicious: human review
        → If normal: use fast model
        
Result: Trade security for speed intelligently
```

## Future Directions

- **Verified Robustness:** Mathematical guarantees, not empirical
- **Certified Training:** Training methods that provably maintain robustness
- **Transparent Models:** Models where robustness is easier to understand
- **Adversarial Robustness as Standard:** Build robustness in from the start, not as an afterthought