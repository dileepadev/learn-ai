---
title: Dual-Use AI Research Risks
description: Understand what makes AI research dual-use, how the field is adapting biosecurity-style norms, and the tradeoffs in restricting open publication.
---

Dual-use research is research that has both clearly beneficial applications and a plausible path to causing serious harm if misused. AI research increasingly grapples with this tension in ways the field's historically open publication culture wasn't built to handle.

## Where Dual-Use Concerns Arise in AI

Models trained to design proteins or novel molecules for drug discovery can, in principle, be repurposed to help design harmful biological or chemical agents. Models with strong cybersecurity capabilities — finding vulnerabilities, writing exploit code — accelerate both defensive patching and offensive attack development. Highly capable general-purpose models raise dual-use concerns more diffusely: the same reasoning and planning capabilities that help with legitimate research can also lower the barrier to sophisticated misuse across many domains at once, which is part of why frontier model evaluations increasingly include specific tests for dangerous capability uplift.

## Tension with Open Publication Norms

Academic AI research has traditionally prized open publication of methods, code, and sometimes model weights, which accelerates progress and enables independent verification, but a fully open release also gives anyone, including bad actors, the same capabilities. This has produced real disagreement within the field over practices like staged release (publishing findings before releasing a fully capable model, or releasing progressively larger versions over time) and withholding certain technical details that would meaningfully lower the barrier to misuse without providing much additional safety or scientific benefit.

## Structured Access as a Middle Ground

Rather than a binary choice between fully open and fully closed, many labs use structured access: releasing a model through an API with usage monitoring and misuse detection, or granting research access under a data use agreement, rather than releasing weights or full technical details unconditionally. This preserves the ability to revoke access if misuse is detected, at the cost of restricting the broader research community's ability to independently audit or build on the work as freely as full open release would allow.

## Practical Guidance

Before publishing dual-use-adjacent AI research, consider a structured risk assessment: who specifically benefits from full disclosure, what does withholding a component actually cost in terms of lost scientific value, and does a redacted or delayed release meaningfully reduce misuse risk without eliminating the legitimate research contribution. Organizations building models with plausible dual-use capability should have a defined internal review process for release decisions rather than leaving this judgment to individual researchers or teams on an ad hoc basis.
