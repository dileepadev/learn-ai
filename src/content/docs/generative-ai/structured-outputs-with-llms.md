---
title: Structured Outputs with LLMs
description: How to reliably extract structured data from large language models using JSON mode, function calling, and schema enforcement.
---

Large language models generate free-form text by default. For many real-world applications — APIs, data pipelines, user interfaces — you need predictable, machine-readable output rather than prose. Structured outputs are the set of techniques used to constrain LLM responses into a defined format such as JSON, XML, or a typed schema.

## Why Structured Outputs Matter

When an LLM is embedded in a larger system, downstream components depend on consistent formatting. A response that sometimes returns a JSON object and sometimes wraps it in a markdown code block will break parsers and raise reliability issues. Structured outputs eliminate this class of failure.

Common scenarios where structured outputs are essential:

- Extracting entities (names, dates, amounts) from documents
- Filling form fields from natural language descriptions
- Generating API responses that conform to a schema
- Building agentic tools that consume model output programmatically

## Approaches

### JSON Mode

Several model providers offer a JSON mode that instructs the model to always output valid JSON. The model is constrained at the sampling level to never produce tokens that would result in malformed JSON.

**OpenAI example:**

```json
{
  "response_format": { "type": "json_object" }
}
```

JSON mode guarantees valid JSON but does not enforce a specific schema — the keys and values are still determined by the prompt.

### Function Calling / Tool Use

Function calling (also referred to as tool use) lets you define a schema for the model's output. The model selects a function and provides arguments that match the declared parameter types, enforced by the API layer.

```json
{
  "name": "extract_invoice",
  "description": "Extract invoice details from text",
  "parameters": {
    "type": "object",
    "properties": {
      "vendor": { "type": "string" },
      "amount": { "type": "number" },
      "due_date": { "type": "string", "format": "date" }
    },
    "required": ["vendor", "amount", "due_date"]
  }
}
```

The model returns structured arguments that are guaranteed to match the schema, not prose.

### Structured Outputs with Schema Enforcement

OpenAI's Structured Outputs feature (introduced in 2024) extends function calling with strict schema validation. Every field in the schema is respected exactly, including required fields, type constraints, and nesting depth. The API rejects responses that do not conform before returning them to the caller.

### Prompt-Based Constraints

Without API-level schema enforcement, prompts alone can guide structure. Instructing the model to respond only with a JSON object and providing a clear template reduces formatting errors, though it does not provide the hard guarantees of schema enforcement.

### Libraries and Frameworks

Several open-source libraries provide schema-based structured output on top of any model:

- **Instructor** (Python): Wraps OpenAI and other clients to validate responses against Pydantic models.
- **Outlines**: Uses constrained token sampling to enforce a grammar or regular expression at generation time.
- **Guidance**: A language for interleaving generation and structure constraints.
- **LangChain output parsers**: Parse and retry LLM output to match a defined schema.

## Best Practices

- **Use the most constrained method available.** Prefer schema enforcement over JSON mode over prompt-only approaches, in that order.
- **Keep schemas shallow.** Deeply nested schemas increase the chance of subtle conformance errors.
- **Always validate downstream.** Even with schema enforcement, validate the parsed output in your application code before using it.
- **Provide examples in the prompt.** Few-shot examples showing expected input-output pairs significantly improve accuracy when schema enforcement is unavailable.
- **Handle retries gracefully.** Build retry logic that re-submits the prompt with the invalid output and an explanation of what was wrong.

## Tradeoffs

| Approach | Schema Guaranteed | Requires API Support | Flexibility |
|---|---|---|---|
| Schema enforcement | Yes | Yes | Low |
| Function calling | Yes (with strict mode) | Yes | Medium |
| JSON mode | Format only | Yes | High |
| Prompt-based | No | No | High |
| Constrained sampling (Outlines) | Yes | No (local) | Medium |

## Summary

Structured outputs are a prerequisite for using LLMs reliably in production systems. The right approach depends on your model provider, deployment environment, and tolerance for unpredictability. As API-level schema support becomes standard, prompt-only approaches are increasingly a fallback for legacy or open-weight models.
