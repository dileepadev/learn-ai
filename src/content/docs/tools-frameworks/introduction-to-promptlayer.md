---
title: Introduction to PromptLayer - Prompt Management and Versioning
description: Learn how PromptLayer separates prompt content from application code, enabling versioning, collaboration, and evaluation without redeploys.
---

PromptLayer is a prompt management platform that treats prompts as versioned, collaboratively editable assets rather than hardcoded strings buried inside application code.

## Separating Prompts from Code

Instead of embedding prompt text directly in a codebase, prompts are stored and versioned in PromptLayer and fetched by a stable identifier at runtime:

```python
from promptlayer import PromptLayer

pl_client = PromptLayer(api_key="PROMPTLAYER_API_KEY")

prompt = pl_client.templates.get("customer-support-triage", version="production")
response = pl_client.run(
    prompt_name="customer-support-triage",
    input_variables={"ticket_text": ticket_text}
)
```

This means a non-engineer (a prompt engineer, product manager, or domain expert) can edit and test prompt wording through a UI and promote a new version to production without requiring a code deployment, and every prompt change is tracked with full version history.

## Request Logging and Evaluation

Every LLM request made through PromptLayer is logged with its exact prompt version, inputs, outputs, latency, and cost, creating a searchable history that's essential for debugging why a specific output was produced or comparing behavior across prompt versions. PromptLayer also supports running evaluation sets against multiple prompt versions to compare quality before promoting a change to production, rather than deploying a prompt edit and hoping it performs as well as the previous version.

## Prompt Registries and Collaboration

Teams often maintain many prompts across different features, and PromptLayer's registry gives a central, searchable place to see what prompts exist, who owns them, and their current production version — addressing the common failure mode where prompt wording drifts across a codebase with no single source of truth and no visibility into who changed what.

## Practical Guidance

Adopt a prompt management tool once prompt iteration speed becomes a bottleneck tied to code deployment cycles, or once multiple people (engineers and non-engineers) need to collaborate on prompt wording. Treat prompt versions with the same rigor as code changes — evaluate before promoting, and keep a rollback path to the previous version readily available.
