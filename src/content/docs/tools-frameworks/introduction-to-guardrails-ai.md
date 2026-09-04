---
title: Introduction to Guardrails AI
description: Understand how Guardrails AI enforces structural integrity, regex guarantees, PII redaction, and hallucination guardrails on large language model outputs using RAIL specs.
---

Large Language Models (LLMs) are probabilistic systems: they do not guarantee that output strings will conform to strict JSON schemas, adhere to mathematical bounds, or avoid leaking sensitive Personally Identifiable Information (PII). In production environments, unvalidated LLM outputs lead to backend crashes, SQL errors, regulatory fines, and brand reputational damage.

**Guardrails AI** is an open-source framework that adds structural and semantic **execution guardrails** to LLM applications. By wrapping LLM API calls with composable, programmable validators, Guardrails AI guarantees that generated outputs satisfy predefined structural, statistical, and security constraints before they are consumed by downstream systems.

---

## The Guardrails Execution Lifecycle

Instead of treating LLM output as a blind black box, Guardrails executes a continuous verification loop:

```
[ User Prompt / Context ]
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Guard Execution Layer                                                       │
│                                                                             │
│ 1. Formats Prompt with Structural Specification Schema (Pydantic / RAIL)    │
│ 2. Dispatches Request to LLM Provider (OpenAI, Anthropic, Ollama, etc.)     │
│ 3. Intercepts Raw Text Output                                               │
│                                                                             │
│ 4. Executes Validation Pipeline (Guardrails Hub):                           │
│    • Schema Structure Validation: Are all required JSON keys present?       │
│    • Semantic Validation: Are bounds respected? Any toxic speech or PII?    │
│    • Factuality Validation: Is output grounded in provided context?         │
│                                                                             │
│ 5. Action on Failure:                                                       │
│    ┌───────────────┬───────────────────────────────┬──────────────────────┐ │
│    ▼               ▼                               ▼                      │ │
│   [ Fix / Filter ] [ Re-ask LLM with Correction ]   [ Raise Exception ]    │ │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
                       Guaranteed Validated Output!
```

---

## On-Fail Actions: Handling Validation Failures

When an output fails validation, Guardrails provides configurable recovery strategies:

| On-Fail Action | Behavior | Best Used For |
| :--- | :--- | :--- |
| **`noop`** | Logs validation error, takes no action, returns raw output | Non-critical warnings, auditing |
| **`filter`** | Drops the violating key/attribute from the returned dictionary | Optional metadata fields |
| **`fix`** | Programmatically repairs the output (e.g., regex trimming, clipping numbers)| Deterministic formatting errors |
| **`reask`** | Automatically sends the validation traceback back to the LLM to self-correct | Formatting slips, JSON syntax issues |
| **`exception`**| Immediately halts execution and raises a Python runtime error | Critical security, PII, or safety breaches |

---

## The Guardrails Hub

The **Guardrails Hub** is a modular repository of community-curated and enterprise-tested validators that can be mixed and matched:

```bash
# Install individual modular validators via the Guardrails CLI
guardrails hub install hub/detect_pii
guardrails hub install hub/toxic_language
guardrails hub install hub/provenance_v1
guardrails hub install hub/regex_match
```

Popular Hub Validators:
- **`detect_pii`:** Scans for phone numbers, credit cards, Social Security numbers, and email addresses.
- **`toxic_language`:** Evaluates text against hate speech, harassment, and profanity classifiers.
- **`provenance_v1`:** Uses natural language inference to verify that every claim in the response is supported by source reference passages.
- **`valid_sql`:** Parses SQL queries into an Abstract Syntax Tree (AST) to verify syntactic legality before database execution.

---

## Hands-On Python Walkthrough

### 1. Installation

```bash
pip install guardrails-ai
guardrails hub install hub/detect_pii
guardrails hub install hub/two_words
```

### 2. Defining Guardrails with Pydantic

```python
from pydantic import BaseModel, Field
from guardrails.hub import DetectPII, TwoWords
from guardrails import Guard
from openai import OpenAI

# 1. Define target output schema with attached validators
class UserAccountProfile(BaseModel):
    username: str = Field(
        description="A short two-word username",
        validators=[TwoWords(on_fail="reask")]
    )
    email: str = Field(
        description="User contact email address"
    )
    biography: str = Field(
        description="User profile biography",
        validators=[DetectPII(pii_entities=["PHONE_NUMBER", "SSN"], on_fail="fix")]
    )

# 2. Create the Guard
guard = Guard.from_pydantic(output_class=UserAccountProfile)

# 3. Wrap model invocation
prompt = "Generate a user profile for Alice who loves robotics. Include phone number 555-0199 in bio."

raw_llm_response, validated_output, *rest = guard(
    llm_api=OpenAI().chat.completions.create,
    model="gpt-4o-mini",
    prompt=prompt
)

print("Validated Output Dictionary:")
print(validated_output)
# Result: PII phone number is automatically redacted by the DetectPII 'fix' action!
```

---

## Guardrails vs. Outlines / Instructor

| Dimension | Guardrails AI | Outlines | Instructor |
| :--- | :--- | :--- | :--- |
| **Primary Focus** | Safety, Semantic Validation, PII, Hallucinations | Grammar-constrained Logit Masking | Clean Pydantic Function Calling |
| **Validation Layer** | Post-generation & Interleaved Validation | Pre-generation FSM / Regex Masking | Schema Parsing via Tool Calls |
| **Corrective Action** | Automatic Multi-turn Re-asking & Repair | Mathematical impossibility to emit invalid tokens | Retry loops on ValidationError |
| **Ecosystem** | Modular Guardrails Hub | Local Open-Source Runtimes | Any LLM with Tool Calling |

---

## Key Takeaways

- Guardrails AI adds structural, statistical, and safety guarantees to probabilistic language model outputs.
- Configurable On-Fail policies (`reask`, `fix`, `filter`, `exception`) allow graceful degradation instead of unhandled production crashes.
- The Guardrails Hub provides modular, plug-and-play validators spanning PII scrubbing, SQL verification, and factuality grounding.
