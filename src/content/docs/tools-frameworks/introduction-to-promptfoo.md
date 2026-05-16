---
title: Introduction to Promptfoo
description: Get started with Promptfoo — an open-source CLI and library for testing, evaluating, and red-teaming LLM prompts and AI applications — covering configuration, test cases, assertions, model comparison, and CI/CD integration.
---

Promptfoo is an open-source **LLM prompt testing and evaluation framework** that helps engineers systematically test AI applications across models, prompts, and datasets. It treats prompt engineering as software engineering: test cases, assertions, continuous integration, and reproducible benchmarks replace ad-hoc manual testing. Promptfoo is model-agnostic, supports dozens of providers, and can run locally or in CI pipelines.

## Why Prompt Testing Matters

Prompts are code. Like any software component, they need to be tested against:

- **Regressions**: a prompt change that improves one scenario may break another
- **Model switches**: migrating from GPT-4o to Claude 3.5 may change behavior in non-obvious ways
- **Edge cases**: unusual inputs that expose prompt failures not apparent in happy-path testing
- **Safety**: systematic red-teaming to find jailbreaks, bias, and harmful outputs before deployment

Manual testing is insufficient at scale. Promptfoo provides a systematic test harness analogous to unit and integration testing for traditional software.

## Installation

```bash
# Via npm
npm install -g promptfoo

# Or run directly without installing
npx promptfoo@latest
```

Verify the installation:

```bash
promptfoo --version
```

## Core Concepts

### Providers

**Providers** are the LLM APIs or model backends being evaluated. Promptfoo supports:

- OpenAI (`openai:gpt-4o`, `openai:gpt-4o-mini`)
- Anthropic (`anthropic:claude-3-5-sonnet-20241022`)
- Google (`google:gemini-2.0-flash`)
- Ollama (`ollama:llama3.2`)
- Azure OpenAI, AWS Bedrock, Vertex AI, HuggingFace, custom HTTP endpoints

### Prompts

**Prompts** are the templates under evaluation. They can be:

- Plain text strings with `{{variable}}` interpolation
- Structured chat message arrays (system + user turns)
- Files (`.txt`, `.json`, `.yaml`)
- Nunjucks or Handlebars templates for complex logic

### Test Cases

**Test cases** provide concrete inputs (variables) and expected behaviors (assertions) for each prompt.

### Assertions

**Assertions** define what constitutes a passing test. They can check:

- Exact string matches
- Contains / not-contains patterns
- Regular expressions
- LLM-graded criteria (semantic correctness)
- JSON schema validity
- Custom JavaScript/Python functions

## Configuration File

Promptfoo is configured via a `promptfooconfig.yaml` file:

```yaml
# promptfooconfig.yaml
description: "RAG Q&A prompt evaluation"

providers:
  - openai:gpt-4o
  - anthropic:claude-3-5-sonnet-20241022

prompts:
  - "Answer the following question using only the provided context.\n\nContext: {{context}}\n\nQuestion: {{question}}"
  - file://prompts/rag-v2.txt

tests:
  - vars:
      context: "The Eiffel Tower is located in Paris, France. It was built in 1889."
      question: "Where is the Eiffel Tower?"
    assert:
      - type: contains
        value: "Paris"
      - type: llm-rubric
        value: "Response correctly identifies the location without hallucinating additional information"

  - vars:
      context: "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen."
      question: "What are the inputs to photosynthesis?"
    assert:
      - type: contains-all
        value: ["sunlight", "water", "CO2"]
      - type: not-contains
        value: "nitrogen"

  - vars:
      context: "The company was founded in 2010 by Alice Chen."
      question: "Who founded the company?"
    assert:
      - type: regex
        value: "Alice Chen"
      - type: javascript
        value: "output.length < 200"  # response is concise
```

## Running Evaluations

```bash
# Run with default config file
promptfoo eval

# Run with a specific config
promptfoo eval --config promptfooconfig.yaml

# Run with verbose output
promptfoo eval --verbose

# Output results as JSON
promptfoo eval --output results.json
```

The output shows a results table with pass/fail status for each test × prompt × provider combination.

## Viewing Results

Promptfoo includes a web UI for exploring results:

```bash
# Open the results viewer
promptfoo view
```

The UI shows:

- A comparison grid: rows are test cases, columns are prompt × provider combinations
- Color-coded pass/fail indicators per assertion
- Full prompt/response for each cell
- Diff view when comparing prompt versions

## Assertion Types

### String Assertions

```yaml
assert:
  - type: contains
    value: "expected substring"

  - type: not-contains
    value: "unexpected text"

  - type: contains-all
    value: ["term1", "term2", "term3"]

  - type: starts-with
    value: "Response:"

  - type: regex
    value: "\\d{4}-\\d{2}-\\d{2}"  # date pattern
```

### LLM-as-Judge

```yaml
assert:
  - type: llm-rubric
    value: "The response is factually accurate, concise, and does not include information not present in the context"
    provider: openai:gpt-4o  # judge model (defaults to config provider)

  - type: answer-relevance
    threshold: 0.8  # semantic similarity score 0-1

  - type: context-faithfulness
    threshold: 0.9  # checks response is grounded in provided context
```

### JSON and Structured Output

```yaml
assert:
  - type: is-json

  - type: json-schema
    value:
      type: object
      required: ["answer", "confidence"]
      properties:
        answer:
          type: string
        confidence:
          type: number
          minimum: 0
          maximum: 1
```

### Custom Function Assertions

```yaml
assert:
  - type: javascript
    value: |
      // output is the model response string
      const words = output.trim().split(/\s+/);
      return words.length <= 100;  // max 100 words
```

```yaml
assert:
  - type: python
    value: |
      import json
      data = json.loads(output)
      return data.get("sentiment") in ["positive", "negative", "neutral"]
```

## Comparing Multiple Models

Promptfoo shines at **model comparison** — evaluating the same prompt suite across different providers or model versions:

```yaml
providers:
  - openai:gpt-4o
  - openai:gpt-4o-mini
  - anthropic:claude-3-5-haiku-20241022
  - ollama:llama3.2

prompts:
  - file://prompts/customer-support.txt

tests:
  - file://tests/customer-support-cases.yaml
```

Running `promptfoo eval` produces a side-by-side comparison of all model × prompt combinations, enabling data-driven model selection decisions.

## Red-Teaming

Promptfoo includes built-in red-teaming capabilities for systematic safety testing:

```bash
# Generate adversarial test cases automatically
promptfoo redteam init

# Run the red-team evaluation
promptfoo redteam run
```

The red-teaming module:

- Generates jailbreak attempts targeting the configured prompt
- Tests for harmful outputs across categories (violence, hate speech, PII leakage, prompt injection)
- Scores model safety across attack categories
- Reports a safety score with detailed breakdowns

### Custom Red-Team Plugins

```yaml
# redteam config section in promptfooconfig.yaml
redteam:
  plugins:
    - prompt-injection
    - jailbreak
    - harmful:violence
    - pii:direct
    - overreliance
  strategies:
    - jailbreak:composite
    - multilingual
```

## CI/CD Integration

Promptfoo integrates with CI pipelines to catch prompt regressions before deployment:

```yaml
# .github/workflows/prompt-eval.yml
name: Prompt Evaluation

on:
  pull_request:
    paths:
      - "prompts/**"
      - "promptfooconfig.yaml"

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Promptfoo Evaluation
        run: npx promptfoo@latest eval --ci
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: output.json
```

The `--ci` flag fails the pipeline if any assertions fail, treating prompt regressions as build failures.

## Dataset-Driven Testing

For large test suites, define test cases in separate files:

```yaml
# tests/qa-cases.yaml
- vars:
    question: "What year was the Eiffel Tower built?"
    context: "The Eiffel Tower was built between 1887 and 1889..."
  assert:
    - type: contains
      value: "1889"

- vars:
    question: "Who designed the Eiffel Tower?"
    context: "Gustave Eiffel designed the tower..."
  assert:
    - type: llm-rubric
      value: "Correctly identifies Gustave Eiffel as the designer"
```

Reference in config:

```yaml
tests:
  - file://tests/qa-cases.yaml
```

## Caching

Promptfoo caches LLM responses to avoid redundant API calls during iterative development:

```bash
# Clear cache
promptfoo cache clear

# Disable cache for a run
promptfoo eval --no-cache
```

Cache is stored locally and keyed by provider + prompt + inputs hash, so identical requests reuse cached responses.

## Comparison with Alternatives

| Feature | Promptfoo | RAGAS | Langfuse | BrainTrust |
| --- | --- | --- | --- | --- |
| Open-source | Yes | Yes | Yes | Partial |
| CLI-first | Yes | No | No | No |
| Model comparison | Yes | Limited | No | Yes |
| Red-teaming | Yes | No | No | Partial |
| CI/CD integration | Yes | Limited | Yes | Yes |
| RAG-specific evals | Partial | Yes | Yes | Yes |
| Trace-based eval | No | No | Yes | Yes |

## Summary

Promptfoo brings software engineering discipline to prompt development. Its key strengths are:

- **Configuration-driven**: declarative YAML configs version-control alongside prompts
- **Multi-provider comparison**: evaluate any number of models in a single run
- **Rich assertion types**: from simple string checks to LLM-graded semantic criteria
- **Built-in red-teaming**: systematic safety evaluation without custom tooling
- **CI/CD native**: treats prompt regressions as build failures

For teams iterating on prompts at scale or evaluating model upgrades, Promptfoo provides the systematic testing infrastructure that ad-hoc manual evaluation cannot match.
