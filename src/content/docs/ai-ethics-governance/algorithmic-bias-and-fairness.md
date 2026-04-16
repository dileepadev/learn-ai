---
title: Algorithmic Bias and Fairness in AI
description: Understand the sources of algorithmic bias in machine learning systems, the competing mathematical definitions of fairness, and practical techniques for building more equitable AI.
---

Algorithmic bias occurs when an AI system produces systematically unfair outcomes for certain groups of people. As machine learning models are deployed in consequential domains — hiring, credit, healthcare, criminal justice, and content moderation — the impacts of bias are no longer academic. They affect real people's access to opportunities and resources.

Understanding algorithmic bias requires confronting both technical and social dimensions: bias is often a reflection of society's historical inequities encoded into data, magnified and automated at scale.

## Sources of Bias

### Historical Bias

Training data reflects past human decisions, which themselves encode discrimination. A hiring model trained on historical hiring decisions learns to prefer candidates similar to those historically hired — perpetuating past exclusions regardless of actual ability.

### Representation Bias

If certain groups are underrepresented in training data, the model learns poorer representations for them. Face recognition systems famously perform significantly worse on darker-skinned women (Buolamwini & Gebru, 2018) because early datasets were predominantly light-skinned males.

### Measurement Bias

The proxy measure used as a target may not capture what we actually care about for all groups equally. Using arrest records as a proxy for criminal behavior encodes policing patterns, not underlying behavior — communities with heavier policing are measured differently.

### Feedback Loops

Deployed models influence the data they collect. A predictive policing system directs more officers to flagged neighborhoods → more arrests there → the model reinforces the pattern. This **runaway feedback** amplifies initial biases over time.

### Aggregation Bias

A model trained on all groups together may not generalize well to subgroups with distinct patterns. A single diabetes model may perform well on average but poorly on populations with different disease manifestations.

### Deployment Shift

A model trained and tested on one population may be deployed on a different one. The performance gap may be systematic across demographic groups.

## Mathematical Definitions of Fairness

There is no single universally agreed definition of algorithmic fairness. The choice among definitions reflects value judgments about what fairness means in a given context.

### Demographic Parity (Statistical Parity)

The proportion of positive predictions is equal across groups:

$$P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)$$

**Limitation:** Ignores true outcome rates. Achieving demographic parity may require a model to apply different quality standards to different groups.

### Equal Opportunity

True positive rates are equal across groups:

$$P(\hat{Y} = 1 \mid Y = 1, A = 0) = P(\hat{Y} = 1 \mid Y = 1, A = 1)$$

Ensures that qualified individuals from each group are equally likely to receive a positive prediction. Preferred in hiring and lending contexts.

### Equalized Odds

Both true positive rates *and* false positive rates are equal across groups:

$$P(\hat{Y} = 1 \mid Y = y, A = 0) = P(\hat{Y} = 1 \mid Y = y, A = 1) \quad \forall y \in \{0, 1\}$$

### Calibration (Predictive Parity)

For a given predicted score $s$, the actual probability of the outcome is the same across groups:

$$P(Y = 1 \mid \hat{p} = s, A = 0) = P(Y = 1 \mid \hat{p} = s, A = 1)$$

Used in recidivism prediction (COMPAS controversy) — the argument that a score of 0.7 means the same thing regardless of race.

### Impossibility Results

Chouldechova (2017) and Kleinberg et al. (2017) proved that when base rates differ between groups, **demographic parity, equal opportunity, and calibration cannot all be satisfied simultaneously**. This is not a technical limitation — it reflects a genuine tension between different conceptions of fairness that must be resolved through values, not mathematics.

## Detecting Bias

### Disaggregated Evaluation

The foundational step: compute and report performance metrics *separately* for each demographic group. An overall accuracy of 92% may hide 98% for the majority group and 70% for the minority.

### Disparate Impact Analysis

Borrowed from U.S. employment law: if the selection rate for a protected group is less than 80% of the highest group's rate, adverse impact is indicated.

$$\text{Disparate Impact Ratio} = \frac{P(\hat{Y}=1 \mid A=\text{minority})}{P(\hat{Y}=1 \mid A=\text{majority})}$$

### Counterfactual Fairness

Ask: would the decision change if the individual's protected attribute were different, holding everything else equal? If yes, the model is using the attribute's causal influence.

### Audit Tools

- **Fairlearn** — Microsoft's open-source toolkit for bias assessment and mitigation.
- **AI Fairness 360 (AIF360)** — IBM's toolbox with 70+ bias metrics and 10+ mitigation algorithms.
- **SHAP** — Explain model decisions at the individual level to surface differential treatment.

## Mitigating Bias

### Pre-processing Techniques

Modify the training data before model training:

- **Reweighting** — Assign higher weights to underrepresented or disadvantaged group samples.
- **Disparate Impact Remover** — Transform features to reduce correlation with protected attributes while preserving rank.
- **Relabeling / Data augmentation** — Correct mislabeled samples for minority groups.

### In-processing Techniques

Modify the learning algorithm:

- **Adversarial debiasing** — Train a classifier jointly with an adversary that tries to predict the protected attribute from the predictions. The classifier is penalized for being distinguishable.
- **Fairness constraints** — Add fairness conditions (equal opportunity, equalized odds) as explicit constraints in the optimization problem.
- **Fair representations** — Learn a representation from which the protected attribute cannot be predicted (IBM FairRep, variational fair autoencoders).

### Post-processing Techniques

Adjust model outputs after training:

- **Threshold adjustment** — Use different decision thresholds per group to equalize a fairness metric (e.g., TPR).
- **Calibrated equalized odds** — Optimize a mix of true and false positive rates to achieve equalized odds.

Post-processing is the most flexible approach — it can be applied to any model without retraining.

## Beyond Technical Fixes

Technical fairness interventions are necessary but not sufficient. Key broader considerations:

- **Stakeholder inclusion** — Involve affected communities in defining what fairness means for the application.
- **Purpose limitation** — Some high-stakes decisions may be inappropriate for algorithmic systems regardless of fairness optimizations.
- **Transparency and contestability** — Individuals should be able to understand why a decision was made and appeal it.
- **Ongoing monitoring** — Fairness must be re-evaluated continuously as deployment contexts and populations shift.
- **Intersectionality** — Bias compounds across multiple attributes (race + gender + age). Analyzing single attributes in isolation misses complex disparities.

## Regulatory Landscape

- **EU AI Act** — Classifies AI systems into risk tiers; high-risk systems (credit, hiring, law enforcement) face strict fairness and transparency requirements.
- **US Equal Credit Opportunity Act (ECOA)** — Prohibits credit discrimination based on protected characteristics; increasingly applied to ML-based lending.
- **NYC Local Law 144** — Requires bias audits for AI tools used in employment decisions; one of the first municipal AI fairness laws.

Algorithmic fairness is an evolving field at the intersection of computer science, statistics, law, philosophy, and social science. Building fairer AI systems requires technical skill — but also the humility to recognize that fairness is ultimately a values question that no algorithm can resolve on its own.
