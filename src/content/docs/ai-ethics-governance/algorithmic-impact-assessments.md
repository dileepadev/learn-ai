---
title: Algorithmic Impact Assessments
description: Learn how algorithmic impact assessments systematically evaluate an automated system's risks to people before and during deployment, and how to run one in practice.
---

An algorithmic impact assessment (AIA) is a structured process for identifying, evaluating, and documenting the potential effects of an automated decision system on individuals and communities, and for deciding what safeguards are needed. It is analogous to environmental or data protection impact assessments.

## Why They Exist

Automated systems used in hiring, lending, benefits, policing, and healthcare can cause harm at scale. Assessing risks *before* deployment is cheaper and safer than responding to harm afterward, and many regulations now require something similar: Canada's Directive on Automated Decision-Making mandates AIAs for federal systems, the EU AI Act requires fundamental rights impact assessments for certain deployers, and GDPR requires data protection impact assessments for high-risk processing.

## Typical Stages

1. **Scope and purpose**: what decision is automated, who is affected, what human oversight exists
2. **Stakeholder mapping**: identify affected groups, including indirectly affected people
3. **Risk identification**: consider bias, privacy, safety, transparency, security, and effects on rights
4. **Impact analysis**: estimate likelihood, severity, reversibility, and who bears the harm
5. **Mitigation planning**: technical, procedural, and governance controls
6. **Consultation**: involve affected communities and domain experts
7. **Documentation and sign-off**: record findings, residual risks, and accountable owners
8. **Monitoring and review**: reassess after launch and when the system changes

## Questions to Ask

- What happens when the system is wrong, and who can contest the outcome?
- Do error rates differ across demographic groups?
- What data is used, and was it collected for this purpose?
- Can affected people understand and challenge decisions?
- Is automation necessary, or would a simpler or non-automated approach suffice?

## Example Risk Register Entry

```yaml
risk: Higher false-rejection rate for applicants with non-standard employment history
affected_group: gig workers, career changers
likelihood: medium
severity: high
mitigation:
  - Add human review for rejections near the decision threshold
  - Monitor false-rejection rate by group monthly
  - Provide an appeal process with a 10-day SLA
owner: Head of Lending Operations
```

## Making Assessments Effective

- Run them early, when design choices are still open, and update them as the system evolves
- Involve people with domain, legal, and lived-experience perspectives, not only engineers
- Tie outcomes to real decisions, including the option to not deploy
- Publish summaries where possible to enable external scrutiny

## Limitations

AIAs can become checkbox exercises if no one has authority to halt deployment, and self-assessments may understate risks. Independent review and clear accountability improve their value.
