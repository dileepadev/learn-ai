---
title: AI Liability and Legal Accountability - Who's Responsible When AI Causes Harm
description: Understand the legal frameworks emerging to assign liability for AI-caused harm, and the key open questions that existing law struggles to answer.
---

When an AI system causes harm — a self-driving car causes an accident, a hiring algorithm discriminates, a medical AI misses a diagnosis — existing liability law was largely written before autonomous decision-making systems existed, and courts and legislators are still working out how it applies.

## Why Traditional Liability Frameworks Struggle

Product liability law traditionally distinguishes manufacturing defects, design defects, and failure-to-warn claims, all of which assume a fairly static, predictable product. AI systems complicate this: a model's behavior can change after deployment through fine-tuning or drift, harm can result from an emergent interaction between the model and unusual inputs never seen in testing, and the causal chain often runs through several parties — the model developer, the company that fine-tuned it, the company that deployed it, and sometimes the end user who prompted it.

```text
Potentially liable parties for one harmful AI output:
  base model developer -> fine-tuning company -> deploying application -> end user
```

## Approaches Being Explored

Some jurisdictions are exploring strict liability for certain high-risk AI applications, which would hold a developer or deployer liable for harm regardless of fault or negligence, shifting the burden away from proving the harm was foreseeable or the result of careless design. Others favor a negligence-based approach that asks whether the developer took reasonable care given the known risks, similar to existing product liability standards, which requires courts to develop a working definition of what "reasonable care" looks like for AI development. The EU's approach under the AI Liability Directive proposal and the revised Product Liability Directive leans toward easier burden-shifting for claimants, making it easier for a harmed party to obtain evidence about how a system was built and trained.

## The Insurance and Contract Layer

In practice, much AI liability today is negotiated contractually between vendors and deploying organizations through indemnification clauses, well before any court weighs in with a general legal standard, and specialized AI liability insurance products are emerging to price this risk. This means many disputes never establish public legal precedent, which slows the development of clear case law even as AI deployment accelerates.

## Practical Guidance

Organizations deploying AI in consequential domains should document model evaluation, known limitations, and deployment safeguards proactively — this evidentiary record matters enormously if liability is ever contested, since "we tested for this failure mode and mitigated it" is a materially different legal position than having no record of due diligence at all. Review indemnification terms in vendor contracts carefully rather than assuming liability defaults to the model provider.
