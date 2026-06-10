---
title: Bias and Fairness in Machine Learning
description: Understanding and mitigating algorithmic bias — fairness metrics, bias sources, and techniques for building equitable AI systems.
---

**Bias and fairness** in machine learning addresses a critical challenge: AI systems trained on historical data can perpetuate or amplify existing societal inequities. A hiring algorithm trained on past hiring decisions may systematically discriminate against underrepresented groups.

Fairness requires intentional design, measurement, and mitigation.

## Sources of Bias

### Data Bias

**Historical bias**: Training data reflects past discrimination. If women were historically underrepresented in certain roles, historical hiring data reflects that bias. A model trained on such data learns biased patterns.

**Measurement bias**: What we measure in data may not reflect reality. For credit scoring, available data (past loans) excludes those denied credit — surviving bias is unobservable.

**Sampling bias**: Training data may not represent the population. A model trained on predominantly white faces may fail for people of color (demonstrated in gender classification benchmarks).

### Algorithmic Bias

**Model choice**: Some algorithms have inherent biases. Decision trees may be biased toward features appearing early; linear models can't learn complex, non-linear protections.

**Hyperparameter tuning**: Optimizing for overall accuracy can hurt minority groups. A classifier achieving 95% accuracy might perform at 70% for a minority subgroup.

### Bias Amplification

Models can amplify biases present in training data. Example:

- Training data: 60% positive outcomes for group A, 40% for group B.
- Trained model: 65% positive predictions for group A, 35% for group B (increased gap).

## Fairness Definitions

Defining "fairness" is contentious; multiple mathematical definitions exist, often in tension:

### Demographic Parity

Equal decision rates across groups:

$$P(\hat{Y} = 1 | A = 0) = P(\hat{Y} = 1 | A = 1)$$

where $A$ is a protected attribute (race, gender) and $\hat{Y}$ is the model's prediction.

**Intuition**: No group should be systematically disadvantaged.

**Limitation**: Ignores true labels; can force unfair outcomes (denying qualified individuals to achieve balance).

### Equalized Odds

Equal true positive and false positive rates across groups:

$$P(\hat{Y} = 1 | Y = 1, A = 0) = P(\hat{Y} = 1 | Y = 1, A = 1)$$
$$P(\hat{Y} = 1 | Y = 0, A = 0) = P(\hat{Y} = 1 | Y = 0, A = 1)$$

The model's accuracy should be the same across groups (true positives) and false alarm rates should be equal.

**Advantage**: Respects true labels; ensures accuracy fairness.

**Limitation**: May be impossible to satisfy with other constraints (e.g., if base rates differ).

### Equality of Opportunity

Equal true positive rates (focus on those who should be accepted):

$$P(\hat{Y} = 1 | Y = 1, A = 0) = P(\hat{Y} = 1 | Y = 1, A = 1)$$

Weaker than equalized odds; focuses on not harming positives.

### Predictive Parity

Equal positive predictive value across groups:

$$P(Y = 1 | \hat{Y} = 1, A = 0) = P(Y = 1 | \hat{Y} = 1, A = 1)$$

When the model predicts positive, the true positive rate should be the same across groups.

**Trade-off**: Conflicts with equalized odds when base rates differ.

### Calibration

Model's predicted probabilities should match empirical frequencies:

$$P(Y = 1 | \hat{P} = p, A = a) \approx p \quad \forall a$$

where $\hat{P}$ is the predicted probability.

A calibrated model's predictions are trustworthy across groups.

## Fairness-Accuracy Trade-offs

Fairness constraints often reduce overall accuracy:

- **Demographic parity**: Can significantly lower accuracy if enforced strictly.
- **Equalized odds**: Smaller trade-off; respects ground truth.
- **Calibration**: Little accuracy cost if well-calibrated models exist.

Decision-makers must choose based on the application:

- **High-stakes (criminal justice)**: Prioritize fairness; accept accuracy loss.
- **Accuracy-critical (medical diagnosis)**: Accept fairness imperfections for higher overall accuracy.

## Mitigation Strategies

### Pre-Processing: Fair Data

Balance training data across groups or reweight samples:

$$w_i = \frac{1}{n} \cdot \frac{P(A = a_i)}{n_{a_i}}$$

Upweight minority samples; downweight majority samples. Reduces but doesn't eliminate bias.

### In-Processing: Fair Learning

Modify the loss function to include fairness constraints:

$$\min_\theta \mathcal{L}(Y, \hat{Y}) + \lambda \cdot \mathcal{L}_{\text{fairness}}(A, \hat{Y})$$

where $\mathcal{L}_{\text{fairness}}$ penalizes violations of chosen fairness definition.

**Fairness-aware learners**: Train models subject to fairness constraints using Lagrangian relaxation or other optimization techniques.

### Post-Processing: Fair Predictions

Adjust model outputs to satisfy fairness constraints without retraining:

- **Threshold adjustment**: Use different decision thresholds for different groups.
- **Output calibration**: Adjust predicted probabilities to be calibrated.

### Causal Approaches

Use causal reasoning (remove effects of protected attributes):

1. **Identify confounders**: What variables mediate the protected attribute's effect?
2. **Adjust**: Control for confounders; direct effects of protected attribute become unbiased.

## Evaluation and Monitoring

### Fairness Audits

Regularly evaluate model performance across demographic groups:

- **Subgroup accuracy**: Measure accuracy for each group separately.
- **Fairness metrics**: Compute demographic parity, equalized odds, etc.
- **Intersectionality**: Evaluate at intersections of multiple attributes (e.g., Black women vs. white men).

### Continuous Monitoring

In production, monitor performance drift across groups:

$$\Delta_{\text{fairness}} = |\text{FairnessMetric}_{\text{train}} - \text{FairnessMetric}_{\text{deployment}}|$$

If drift exceeds threshold, retrain or adjust.

## Challenges

### Causality and Counterfactuals

Fairness definitions often rely on counterfactuals: "What would have happened if the person were in a different group?" Counterfactuals are unobservable; causal assumptions are required and often controversial.

### Multiple Fairness Objectives

Different stakeholders want different fairness properties. Optimizing for one fairness metric often violates others. No universally fair algorithm exists.

### Protected vs. Proxy Attributes

Even if a model doesn't use protected attributes directly, it can infer them from proxies. Example: zip code correlates with race. Removing proxies may remove legitimate information.

### Fairness vs. Transparency

Enforcing fairness constraints can reduce interpretability. Fairness-aware models may be harder to explain than simpler models.

## Applications and Case Studies

### Criminal Justice

**COMPAS** (Correctional Offender Management Profiling for Alternative Sanctions) predicts recidivism risk. Studies found it was more likely to rate Black defendants as high-risk despite similar histories. Illustrates challenges: fairness definitions disagree on whether the system is unfair.

### Hiring and Recruitment

Amazon scrapped an AI hiring tool that discriminated against women. The model was trained on past hiring decisions reflecting male-dominated tech industry. Lesson: fairness requires intentional efforts; merely removing sensitive attributes is insufficient.

### Credit Lending

Fair lending requires equal access to credit regardless of protected attributes. ML models must be designed to avoid discriminatory lending practices (which may violate fair lending laws).

## Regulatory Landscape

**EU AI Act**: Requires fairness and non-discrimination for high-risk AI systems.

**Fair Lending Laws**: US Fair Housing Act and Equal Credit Opportunity Act restrict discrimination in lending.

**Algorithmic Accountability**: Emerging laws require companies to audit AI systems for bias.

## Best Practices

1. **Measure fairness**: Choose fairness metrics aligned with your values; measure regularly.
2. **Diverse teams**: Include perspectives from affected communities in algorithm design.
3. **Transparency**: Explain how fairness is defined and how the system works.
4. **Redress**: Provide mechanisms for individuals to appeal unfavorable decisions.
5. **Tradeoff awareness**: Communicate fairness-accuracy trade-offs to stakeholders.
6. **Continuous monitoring**: Fairness is not one-time; monitor in production.

## Research Directions

- **Intersectionality**: Fairness across combinations of attributes (not just individual attributes).
- **Multi-objective fairness**: Optimizing multiple fairness objectives simultaneously.
- **Federated fairness**: Fair learning on decentralized data.
- **Causal fairness**: Incorporating causal reasoning into fairness definitions.

Fairness in machine learning is an evolving field, balancing technical rigor with societal values. Building equitable AI systems requires both technical skill and ethical commitment.
