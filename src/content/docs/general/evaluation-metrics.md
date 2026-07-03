---
title: "Evaluation Metrics for AI Systems: Beyond Accuracy"
description: "Understanding precision, recall, F1, AUC, and when each metric matters for evaluating AI models."
---

Your model is 95% accurate. Is it good? Maybe. If 99% of examples are negative and your model always predicts negative, you'd have 99% accuracy and zero value. Accuracy alone is meaningless. You need the right metrics.

## Classification Metrics

### Confusion Matrix Foundation
All classification metrics come from this:

```
           Predicted
           Positive  Negative
Actual   ┌─────────┬──────────┐
Positive │   TP    │   FN     │  (True Positive, False Negative)
         ├─────────┼──────────┤
Negative │   FP    │   TN     │  (False Positive, True Negative)
         └─────────┴──────────┘
```

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

"Out of all predictions, how many were correct?"

Problem: Useless for imbalanced data
Example: 99% accuracy when 99% of data is negative (just predict negative always)
```

### Precision
```
Precision = TP / (TP + FP)

"Of the predictions I made as positive, how many were actually positive?"

Use when: False positives are costly
Example: Email spam detection (wrong flagged emails are bad)
```

### Recall (Sensitivity)
```
Recall = TP / (TP + FN)

"Of all actual positives, how many did I find?"

Use when: False negatives are costly  
Example: Disease detection (missing actual diseases is bad)
```

### F1 Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

"Balanced measure of precision and recall"

Use when: You care about both false positives and false negatives
```

## When to Use Each Metric

### Scenario 1: Medical Diagnosis
```
Disease is rare (0.1% of population have it)

Metric: Recall >> Precision
Goal: Find all actual cases (miss 0% of sick people)
Reasoning: False negatives (missing disease) = potential death
          False positives (wrong diagnosis) = just more tests

Example threshold: 90% recall (find 90% of cases), even if precision is only 50%
(This means: of predicted positives, only half are actually positive)
```

### Scenario 2: Email Spam Filter
```
Most emails are legitimate (95%)

Metric: Precision >> Recall
Goal: Never mark legitimate emails as spam
Reasoning: False positives (blocking good email) = user misses important messages
          False negatives (spam gets through) = minor inconvenience

Example: 99% precision (almost never block real emails), even if recall is 70% (some spam gets through)
```

### Scenario 3: Product Recommendation
```
Metric: Balanced (F1 or weighted metric)
Goal: Recommend products users actually want
Reasoning: Too many bad recommendations (low precision) = poor UX
          Missing good recommendations (low recall) = missed revenue
```

## Beyond Binary Classification

### Multiclass Problems
When you have 3+ classes:

```
Macro-average: Average metric across all classes equally
  - Use when: Classes are equally important

Weighted average: Average metric weighted by class frequency
  - Use when: Class frequencies matter

Example:
- Class A: 80% samples, 90% precision
- Class B: 10% samples, 70% precision
- Class C: 10% samples, 50% precision

Macro: (90 + 70 + 50) / 3 = 70%
Weighted: 0.8*90 + 0.1*70 + 0.1*50 = 82%
```

### AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
```
"How good is the model across all classification thresholds?"

Standard metric: Plots True Positive Rate vs. False Positive Rate
Interpretation: 
- 0.5 = Random (useless)
- 0.7-0.8 = Fair
- 0.8-0.9 = Good
- 0.9+ = Excellent

Use when: You don't have a fixed decision threshold
```

### AUC-PR (Area Under Precision-Recall Curve)
```
"How good is the model when you care about precision vs. recall?"

Use when: Dealing with imbalanced data or you care more about precision/recall than TPR/FPR
Better for imbalanced problems than AUC-ROC
```

## Regression Metrics

### MAE (Mean Absolute Error)
```
MAE = Average of |actual - predicted|

Example: Predicting house prices
Actual: $500k, Predicted: $480k
Error: $20k

Use when: You want to know average error in original units
Robust to outliers
```

### RMSE (Root Mean Squared Error)
```
RMSE = √(Average of (actual - predicted)²)

Example: Same $20k error becomes more significant
Penalizes large errors more than small ones

Use when: Large errors are particularly bad
Sensitive to outliers
```

### R² (Coefficient of Determination)
```
R² = 1 - (SSres / SStot)

Interpretation:
- 0.9+ = Excellent (explains 90%+ of variance)
- 0.7-0.9 = Good
- 0.5-0.7 = Fair
- <0.5 = Poor

Use when: You want to know "what percentage of variance does my model explain"
```

## Ranking and Retrieval Metrics

### NDCG (Normalized Discounted Cumulative Gain)
```
Penalizes getting good results in wrong order

Example: Find top 5 documents
- Rank 1: Highly relevant (good position)
- Rank 2: Moderately relevant (okay position)
- Rank 3: Irrelevant (bad position)
- Rank 4: Highly relevant (should be rank 1, bad position)
- Rank 5: Moderately relevant

NDCG penalizes #4 (good result in wrong spot)
```

### MRR (Mean Reciprocal Rank)
```
"How far down the results is the first correct answer?"

MRR = 1 / (Average rank of first correct answer)

Example:
- Query 1: Correct answer at rank 1 → 1/1 = 1.0
- Query 2: Correct answer at rank 3 → 1/3 = 0.33
- Query 3: No correct answer → 0
- MRR = (1.0 + 0.33 + 0) / 3 = 0.44

Use when: You care about finding correct answer quickly (search engines)
```

## Fairness Metrics

### Demographic Parity
```
Model prediction rates should be same across groups

Example: Loan approval
- Group A approval rate: 40%
- Group B approval rate: 42%

Good: Close rates (demographic parity)
Bad: 40% vs 20% (disparate impact)
```

### Equalized Odds
```
True positive rates should be same across groups

Example: Cancer detection
- Correctly detects 90% of cancer in Group A
- Correctly detects 80% of cancer in Group B

Bad: Different recall across groups (equalized odds violated)
```

## Choosing Your Metrics: Framework

1. **What's the cost of each type of error?**
   - False positive costly? → Optimize precision
   - False negative costly? → Optimize recall

2. **Is my data balanced?**
   - Balanced: Accuracy is fine
   - Imbalanced: Use precision/recall or AUC

3. **Do I have a deployment threshold?**
   - Yes, fixed threshold: Precision/Recall/F1
   - No, flexible: AUC-ROC or AUC-PR

4. **What does success look like?**
   - Revenue? → Track business impact, not just model metrics
   - Safety? → Track failure modes, not averages
   - User satisfaction? → A/B test actual impact

## Common Mistakes

❌ **Using only accuracy on imbalanced data**
✓ Use precision/recall or AUC

❌ **Ignoring class imbalance in weighted metrics**
✓ Calculate both macro and weighted averages

❌ **Not considering business impact**
✓ Tie metrics to revenue, safety, or user satisfaction

❌ **Optimizing for test set metrics, ignoring production performance**
✓ Monitor model performance in production with real data

## Metric Selection by Task

| Task | Primary Metric | Secondary | Notes |
|------|---|---|---|
| **Binary Classification** | F1 or AUC-ROC | Precision if > Recall needed | Consider class imbalance |
| **Multiclass** | Weighted F1 or Macro F1 | Per-class precision/recall | Choose weighted if classes imbalanced |
| **Regression** | RMSE or MAE | R² | RMSE if outliers matter |
| **Ranking** | NDCG@5 or NDCG@10 | MRR | Use @K for top-k results |
| **Recommendation** | Recall@K + Precision@K | Coverage | K depends on use case |
| **Fairness** | Demographic Parity + Equalized Odds | Calibration | Use multiple fairness metrics |

## Real-World Example

```
Building email spam filter

Business context:
- Missing spam (FN) is annoying but not catastrophic
- Blocking good emails (FP) is very bad (users miss important messages)
- We expect 10% of emails to be spam

Metrics to track:
- Precision: Focus on not blocking good email (minimize FP)
  Target: 99% (1% of flagged emails are false positives)
- Recall: Catch most spam
  Target: 80% (catch 80% of actual spam)
- User complaints: Metric that matters (track A/B test with real users)

Don't optimize:
- Accuracy: Useless here (90% baseline = just mark all as legit)
- AUC-ROC: Less relevant than precision when FP is the issue

Set threshold:
- Default threshold makes 50% precision, 90% recall
- Adjust threshold to increase precision to 99% (decreases recall to 70%)
- Users prefer missing spam to losing important emails
```