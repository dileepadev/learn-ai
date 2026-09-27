---
title: AI Transparency Reporting
description: Learn what organizations disclose in AI transparency reports, including system documentation, usage and incident data, and how to design reports that are useful to the public and regulators.
---

AI transparency reporting is the practice of regularly publishing information about how AI systems are built, evaluated, deployed, and governed. It complements per-model documentation such as model cards by describing organizational practice and real-world outcomes over time.

## Why Publish

- **Accountability**: lets regulators, researchers, and the public check claims
- **Trust**: users can understand limits and risks
- **Regulatory alignment**: obligations in laws like the EU AI Act, and voluntary frameworks, increasingly call for disclosure of capabilities, training data summaries, and incidents
- **Internal discipline**: collecting the information forces consistent evaluation and record-keeping

## Typical Contents

| Section | Examples |
|---------|----------|
| System description | Purpose, intended and prohibited uses, model type, versioning |
| Data | Sources, collection methods, high-level summary, privacy handling |
| Evaluation | Benchmarks, fairness and robustness tests, red-teaming results, known limitations |
| Safety measures | Content policies, guardrails, human oversight |
| Operations | Usage statistics, enforcement actions, takedown and appeal numbers |
| Incidents | Significant failures, causes, remediation |
| Governance | Responsible owners, review boards, change management |

## Layers of Disclosure

Different audiences need different detail:

1. **Public summary**: plain language, what it does and its risks
2. **Technical documentation**: model and system cards with evaluation details
3. **Regulator or auditor access**: confidential detail under controlled access

## Example Machine-Readable Entry

```json
{
  "system": "support-assistant",
  "version": "2.3",
  "reporting_period": "2026-Q3",
  "monthly_active_users": 410000,
  "evaluations": {
    "toxicity_rate": 0.002,
    "factual_error_rate_sampled": 0.041
  },
  "incidents": [
    {"id": "INC-118", "severity": "medium", "summary": "Incorrect refund policy shown", "resolved": true}
  ],
  "human_escalation_rate": 0.083
}
```

## Good Practice

- Report negative results and limitations, not only strengths
- Use consistent metrics across periods so trends are visible
- Define terms and methods, so numbers can be interpreted and reproduced
- Avoid disclosing sensitive details that would enable misuse (for example, exact jailbreak recipes) or violate privacy
- Provide a channel for feedback and corrections

## Challenges

Disclosure can conflict with trade secrets, security, and competitive pressure. Selective reporting ("transparency washing") undermines the purpose; independent verification and standardized formats help keep reports meaningful.
