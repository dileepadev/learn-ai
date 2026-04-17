---
title: AI Governance Frameworks
description: A detailed overview of leading AI governance frameworks including the EU AI Act, NIST AI Risk Management Framework, ISO/IEC 42001, and global governance initiatives, covering risk classification, compliance obligations, audit requirements, and organisational implementation.
---

AI governance frameworks provide structured approaches for organisations to develop, deploy, and maintain AI systems responsibly. As regulators, standards bodies, and industry groups converge on formalised requirements, understanding these frameworks has become essential for technology leaders, compliance officers, and AI practitioners.

## Why Governance Frameworks Matter

Without structured governance, AI systems can:

- Produce discriminatory or harmful outcomes at scale
- Operate opaquely, preventing affected parties from understanding decisions
- Create unacceptable liability exposure for deploying organisations
- Undermine public trust in AI technologies more broadly

Governance frameworks address these risks through risk classification, transparency obligations, technical safeguards, and accountability mechanisms.

## The EU AI Act

The EU AI Act, which entered into force in August 2024, is the world's first comprehensive horizontal regulation on AI. It applies to any AI system placed on the EU market or affecting EU residents, regardless of where the developer is located.

### Risk Tiers

The Act defines four risk categories:

| Tier | Description | Requirements |
| --- | --- | --- |
| **Unacceptable Risk** | Systems posing clear threats to fundamental rights (e.g., social scoring by governments, real-time biometric surveillance in public spaces) | **Prohibited** — cannot be developed or deployed |
| **High Risk** | Systems in critical domains (healthcare, education, employment, border control, critical infrastructure, law enforcement, justice) | Mandatory conformity assessment, registration, logging, human oversight, accuracy/robustness requirements |
| **Limited Risk** | Chatbots and synthetic content generators | Transparency obligations — users must be informed they are interacting with AI |
| **Minimal Risk** | Spam filters, AI-enabled video games, most B2B tools | No specific obligations beyond general product safety |

### General-Purpose AI Models

The Act introduces specific rules for GPAI models (such as large language models):

- Models exceeding 10^25 FLOPs of training compute are designated **systemic risk models** and face enhanced obligations including adversarial testing (red-teaming), incident reporting, and cybersecurity assessments
- All GPAI providers must publish training data summaries and comply with copyright law

### Key Compliance Timelines

| Date | Requirement |
| --- | --- |
| February 2025 | Prohibited practices rules apply |
| August 2025 | GPAI model obligations apply |
| August 2026 | High-risk system requirements fully applicable |
| August 2027 | Extended transition period ends |

## NIST AI Risk Management Framework

The NIST AI RMF (released January 2023) is a voluntary framework providing organisations with a structured approach to managing AI risks. It is increasingly referenced in US federal procurement and sector-specific guidance.

### The AI RMF Core — GOVERN, MAP, MEASURE, MANAGE

**GOVERN** — Establishes the organisational culture, policies, roles, and accountability structures for responsible AI:

- Define AI risk tolerance and ownership
- Establish policies for responsible AI development
- Create feedback channels and escalation paths

**MAP** — Identifies and categorises the AI system's context, intended use, and potential harms:

- Document system purpose, stakeholders, and use environment
- Identify potential negative impacts across user groups
- Assess third-party AI components in the supply chain

**MEASURE** — Applies methods to assess, analyse, and track AI risks:

- Evaluate performance metrics for fairness, reliability, and accuracy
- Apply bias and vulnerability testing
- Track metrics over time as system or deployment context evolves

**MANAGE** — Prioritises and addresses identified risks:

- Implement controls proportionate to assessed risk level
- Establish incident response procedures
- Maintain documentation for auditability

The NIST AI RMF is technology-neutral and can be applied to any AI system type or deployment context.

## ISO/IEC 42001 — AI Management Systems

ISO/IEC 42001 (published November 2023) is the first international standard specifying requirements for an AI Management System (AIMS). It follows the high-level structure common to ISO 9001 (quality) and ISO 27001 (information security), enabling integration into existing management system programmes.

Key requirements include:

- **Leadership commitment** — top management must establish AI policy and assign accountability
- **Risk and impact assessment** — identify and address harms from AI system lifecycle
- **Operational controls** — document AI system objectives, design, data governance, and testing procedures
- **Monitoring and auditing** — internal audits and management reviews at defined intervals

Organisations can pursue third-party certification against ISO 42001, providing an auditable signal of governance maturity to customers, regulators, and investors.

## Sector-Specific Frameworks

Many industry sectors have developed or adapted AI governance guidance for their specific risk profiles:

| Sector | Framework | Focus Areas |
| --- | --- | --- |
| Financial services | Basel AI guidance, EBA guidelines | Model risk management, explainability for credit decisions |
| Healthcare | FDA SaMD guidance, WHO AI ethics | Clinical validation, post-market surveillance, bias |
| Employment | EEOC AI guidance (US) | Non-discrimination in hiring, promotions, compensation |
| Education | UNESCO AI competency | Age-appropriate AI, data protection for minors |
| Public sector | UK AI Procurement guidance | Transparency, fairness, human oversight |

## Shared Principles Across Frameworks

Although frameworks differ in legal force and scope, they converge on a common set of principles:

- **Transparency and explainability** — affected parties should understand how AI decisions are made
- **Fairness and non-discrimination** — systems should not produce unjustified disparate impacts across demographic groups
- **Human oversight and control** — humans must be able to review, override, and shut down AI systems
- **Accuracy and reliability** — systems should perform consistently across diverse populations and edge cases
- **Safety** — AI must not cause physical, psychological, financial, or societal harm
- **Privacy and data governance** — personal data used in AI must be collected and processed lawfully
- **Accountability** — clear ownership of AI outcomes within deploying organisations

## Implementing Governance in Organisations

A governance implementation roadmap typically follows these stages:

1. **Inventory** — document all AI systems in use, including third-party models and embedded AI features
2. **Risk classification** — apply a risk taxonomy aligned to the relevant regulatory frameworks
3. **Gap analysis** — compare current practices against required controls for each risk tier
4. **Policy development** — publish AI use policies, procurement standards, and data governance rules
5. **Technical controls** — implement logging, explainability tooling, bias monitoring, and access controls
6. **Training and culture** — equip teams with AI literacy and embed responsible AI review into development processes
7. **Audit and continuous review** — establish periodic compliance reviews and incident reporting mechanisms

### Governance Tooling

| Category | Tools |
| --- | --- |
| Model cards and documentation | Hugging Face Model Cards, Google Model Cards |
| Bias and fairness testing | IBM AI Fairness 360, Fairlearn, What-If Tool |
| Explainability | SHAP, LIME, InterpretML, Captum |
| Risk assessment | NIST AI RMF Playbook, Microsoft Responsible AI Impact Assessment |
| Audit logging | MLflow, Weights & Biases, custom lineage tracking |

## Emerging Global Governance Landscape

| Jurisdiction | Initiative | Status |
| --- | --- | --- |
| European Union | EU AI Act | In force — phased enforcement 2024–2027 |
| United States | EO 14110 on AI Safety; NIST RMF | Executive order; voluntary frameworks |
| United Kingdom | Pro-innovation AI principles | Sector-led, non-binding |
| China | Generative AI Regulations | In force since August 2023 |
| Canada | Artificial Intelligence and Data Act (AIDA) | Proposed legislation |
| Singapore | Model AI Governance Framework | Voluntary; widely adopted in APAC |
| Brazil | AI Bill | Parliamentary review |

International alignment is advancing through mechanisms such as the Global Partnership on AI (GPAI), OECD AI Principles, Hiroshima AI Process, and bilateral regulatory dialogues between the EU and US.

The cumulative effect of these frameworks is shifting AI governance from a voluntary ethics exercise to a legal and regulatory compliance obligation. Organisations that treat governance as integral to AI system design — rather than a post-hoc compliance layer — will be better positioned to build trustworthy systems, meet regulatory requirements efficiently, and maintain the confidence of users and affected communities.
