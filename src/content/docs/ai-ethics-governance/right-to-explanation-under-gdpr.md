---
title: The Right to Explanation Under GDPR
description: Understand what GDPR actually requires when automated decisions affect individuals, and what this means in practice for AI system design.
---

The GDPR's provisions on automated decision-making, primarily Article 22, are often summarized as a "right to explanation," though the actual legal text is narrower and more contested than that phrase suggests.

## What Article 22 Actually Says

Article 22 gives individuals the right not to be subject to a decision based solely on automated processing, including profiling, that produces legal or similarly significant effects — with exceptions where the decision is necessary for a contract, authorized by law, or based on explicit consent. When one of those exceptions applies, the data subject still retains the right to obtain human intervention, express their point of view, and contest the decision.

```text
Automated decision with legal/significant effect
  -> generally prohibited unless: contract necessity, legal authorization, or explicit consent
  -> if permitted: subject can demand human review, express their view, and contest the outcome
```

## The "Meaningful Information" Requirement

Separately, Articles 13-15 require controllers to provide "meaningful information about the logic involved" in automated decision-making when data is collected or on request. Legal scholars have debated for years whether this constitutes a genuine right to a specific, individualized explanation of a decision or only a more general right to understand the decision-making process at a system level — the regulation's text does not resolve this cleanly, and interpretations vary across EU member state enforcement.

## Practical Implications for AI System Design

Regardless of the precise legal scope, organizations deploying automated decisions with significant effects on individuals in the EU (credit decisions, hiring, insurance pricing) need a practical way to provide meaningful, non-technical information about how a decision was reached, and a functioning human review process for contested decisions. This has pushed many organizations toward more interpretable model choices, or toward maintaining an explanation layer (feature importance summaries, decision rules) alongside a complex model that is not inherently interpretable on its own.

## Practical Guidance

Build the human review and contestability pathway into the product from the start rather than treating it as a compliance afterthought — Article 22 rights are about process and recourse as much as about the model's technical explainability. Don't conflate having a SHAP or LIME explanation available with legal compliance; the meaningful-information requirement is about what's communicated to the affected individual, not merely what's technically computable about the model internally.
