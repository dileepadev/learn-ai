---
title: Constrained Decoding
description: Explore constrained decoding — techniques for forcing language model outputs to conform to structural constraints such as JSON schemas, context-free grammars, regular expressions, or logical predicates. Covers finite-state machine decoding, logit masking, LMQL, Outlines, Guidance, XGrammar, and applications in structured data extraction, code generation, and tool-use agents.
---

**Constrained decoding** refers to methods that modify the token generation process of a language model to guarantee that outputs conform to a specified structure — a JSON schema, a programming language grammar, a regular expression, or an arbitrary predicate. Rather than sampling freely and post-hoc filtering or retrying until the format is correct (which wastes computation and may never terminate for complex schemas), constrained decoding enforces validity at every generation step by masking or reweighting the token distribution to exclude tokens that would make the partial output impossible to complete validly.

This is crucial for reliable LLM deployment: applications that require structured outputs (tool calling, database query generation, form filling, code synthesis) cannot tolerate free-form generation that violates the expected format.

## The Core Problem

A language model outputs a probability distribution over vocabulary tokens at each step:

$$P(x_t \mid x_{<t}) \propto \exp(\text{logits}_t / T)$$

In unconstrained decoding, any token can be sampled. For structured outputs, most tokens at a given position are invalid — for example, after `{"name": "` in JSON, only string characters are valid; a number or closing brace would produce malformed JSON.

Post-hoc approaches (prompt engineering, retry loops) are fragile: models fail to follow instructions reliably, and retrying is expensive and adds latency. Constrained decoding **eliminates invalid tokens before sampling** by setting their logits to $-\infty$, ensuring:

$$P_\text{constrained}(x_t \mid x_{<t}) \propto \exp(\text{logits}_t) \cdot \mathbb{1}[x_t \in \mathcal{V}_\text{valid}(x_{<t})}]$$

where $\mathcal{V}_\text{valid}(x_{<t})$ is the set of tokens that can appear at position $t$ while keeping the partial sequence completable to a valid output.

## Finite-State Machine Decoding

For **regular languages** (describable by regular expressions, JSON schemas over simple types), constrained decoding is implemented via a **finite-state machine (FSM)**:

1. Compile the constraint (regex, JSON schema) into a deterministic FSM with states and transitions.
1. Map each FSM state to the set of valid next tokens (transitions from that state).
1. During generation, maintain the current FSM state; after each generated token, advance the FSM state along the corresponding transition.
1. Mask all tokens that don't correspond to valid transitions from the current state.

FSM compilation is done once per schema at initialization and cached; the per-token overhead is $O(|\text{vocab}|)$ for the masking step.

**Outlines** (Willard & Louf, 2023) is the leading open-source implementation of FSM-based constrained decoding. It compiles regular expressions, JSON schemas (including nested objects, arrays, enums, and pattern constraints), and Pydantic models to FSMs, providing a Python API for structured generation.

## Context-Free Grammar Decoding

Regular expressions cannot express **context-free constraints** such as balanced parentheses, nested data structures with recursive types, or full programming language grammars. These require **pushdown automata** (equivalent to CFGs).

CFG-constrained decoding maintains a **parsing stack** alongside the LM state:

1. The CFG is converted to a format that supports incremental parsing (Earley parser, CYK, LALR).
1. At each token position, the parser determines which tokens are valid given the current parsing state (stack + parse history).
1. Invalid tokens are masked.

The key challenge: standard LLM tokenization uses subword tokens (BPE, SentencePiece) that may span multiple grammar symbols, making per-token CFG state tracking non-trivial. **XGrammar** (Dong et al., 2024) addresses this by precomputing token masks per grammar state, storing them in an optimized lookup structure that allows $O(1)$ mask retrieval at generation time.

**Guidance** (Microsoft) supports CFG decoding with a Python DSL for expressing grammars inline with generation code, enabling interleaved constrained and free-form generation.

## Logit Processors and Token Healing

In token-based systems, generated tokens may partially overlap with the constraint boundary. For example, if the model generates `{"k` as two tokens `{"` and `k`, but the JSON schema requires a specific key, the partial `k` token must be "healed" — the model backtracks one token and remasks the vocabulary to only allow tokens that complete the required key.

**Token healing** (a technique in Guidance) addresses tokenization boundary artifacts by regenerating the last token with a restricted prefix to ensure the output matches the schema exactly, compensating for tokenization granularity mismatches.

## LMQL: A Query Language for LLMs

**LMQL** (Language Model Query Language, Beurer-Kellner et al., 2022) is a programming language that embeds LM generation within a SQL-like query syntax with first-class support for constraints and control flow:

```lmql
argmax
  "Name: [NAME]"
  "Age: [AGE]"
  "Occupation: [OCCUPATION]"
from
  "openai/gpt-4"
where
  len(TOKENS(NAME)) < 5 and
  INT(AGE) in range(18, 100)
```

LMQL compiles constraints into token-level masks during generation, handling both simple predicates (`INT(AGE) in range(...)`) and complex multi-variable constraints through a combination of FSM decoding and constraint propagation.

## Structured Prediction with Constrained Decoding

### JSON and Tool Calling

Modern LLM APIs (OpenAI, Anthropic, Mistral) implement **structured outputs** or **tool use** by constraining generation to valid JSON matching a specified schema. At each token step, the API's constrained decoder ensures only schema-valid tokens are produced — guaranteeing that tool call arguments, function parameters, and structured responses are always parseable.

### SQL Generation

Text-to-SQL systems use CFG-based constrained decoding to ensure all generated SQL queries are syntactically valid for a given database schema — constraining column names to those in the target table, operator types to valid SQL operators, and query structure to the supported SQL dialect.

### Code Generation

Constrained code generation restricts output to syntactically valid programs in a target language. Combined with type-directed constraints (variables must be in scope, function arguments must match expected types), this reduces compilation errors in LLM-generated code.

## Semantic Constraints via Logit Reweighting

Beyond syntactic constraints (grammar/schema validity), **semantic constraints** can be approximated via logit reweighting:

- **Topic control**: boost logit scores for tokens that continue the current topic using a separately trained topic classifier.
- **Toxicity avoidance**: reduce logit scores for tokens that increase toxicity probability, as estimated by a safety model (used in PPLM, FUDGE).
- **Factual grounding**: reweight tokens using retrieval scores from a knowledge base, encouraging the model to generate text supported by retrieved documents (used in RAG-constrained generation).

These soft constraints don't guarantee validity but shift the output distribution toward desired properties.

## Performance Considerations

Constrained decoding adds overhead to token generation:

| Method | Compilation | Per-token cost | Grammar support |
| --- | --- | --- | --- |
| FSM (Outlines) | Moderate (cached) | $O(\|\text{vocab}\|)$ | Regular |
| CFG (XGrammar) | High (precomputed) | $O(1)$ (lookup) | Context-free |
| LMQL | Moderate | Variable | Custom predicates |
| Guidance | Low | Low (incremental) | Regular + CFG |

XGrammar's precomputed token mask tables reduce per-token constraint overhead to near-zero, making CFG-constrained decoding viable even for high-throughput serving systems.

## Summary

Constrained decoding ensures language model outputs conform to structural constraints by masking invalid tokens before sampling at each generation step. FSM-based decoding handles regular languages (JSON schemas, regexes) via precompiled state machines; CFG-based decoding handles context-free languages (programming grammars, recursive data structures) via incremental parsers. Frameworks including Outlines, XGrammar, Guidance, and LMQL make constrained decoding practical for production systems. Applications span structured data extraction, tool calling, SQL generation, and code synthesis — eliminating format errors that plague unconstrained generation in structured output settings.
