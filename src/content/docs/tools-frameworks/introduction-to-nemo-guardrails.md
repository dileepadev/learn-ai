---
title: Introduction to NeMo Guardrails
description: Learn how to use NVIDIA NeMo Guardrails to add programmable safety, topicality, and factuality rails to LLM applications — using the Colang language to define conversation flows, input/output validation, and integration with LangChain and other AI frameworks.
---

**NeMo Guardrails** is an open-source toolkit from NVIDIA that enables developers to add programmable **rails** to large language model applications — controlling what topics the LLM can discuss, enforcing safety constraints, grounding responses in verified facts, and defining structured conversation flows. Unlike prompt-based safety approaches that rely on the LLM's own judgment, NeMo Guardrails implements constraints at the application layer — making safety properties more reliable and auditable.

The core idea: rather than hoping the LLM refuses harmful requests through fine-tuning alone, NeMo Guardrails intercepts and validates inputs and outputs against explicit, human-defined policies. A configuration of rails becomes a contract between the application and its users — specifiable, testable, and independent of the underlying LLM's safety training.

## Core Concepts

### Rails

A **rail** is a constraint or behavior enforced on an LLM interaction:

- **Input rails**: Validate and potentially reject or modify user inputs before they reach the LLM.
- **Output rails**: Validate and potentially modify LLM outputs before they are returned to the user.
- **Dialog rails**: Define allowed conversation flows and topic boundaries using canonical conversation examples.
- **Retrieval rails**: Control what retrieved content can be injected into prompts in RAG pipelines.

### The Colang Language

NeMo Guardrails uses **Colang**, a domain-specific language for specifying conversation flows and rails. Colang defines:

- **Messages**: Named representations of user utterances or bot responses.
- **Flows**: Sequences of messages that define how conversations should proceed.
- **Actions**: Python functions that can be invoked within flows.

## Installation and Setup

```bash
pip install nemoguardrails
```

A NeMo Guardrails application requires a configuration directory with at minimum:

```
config/
  config.yml      # Main configuration
  rails.co        # Colang rail definitions
  prompts.yml     # Prompt templates (optional)
```

### Minimal Configuration

```yaml
# config/config.yml
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  input:
    flows:
      - self check input
  output:
    flows:
      - self check output
```

## Defining Rails with Colang

### Topic Restriction Rails

Prevent the LLM from discussing topics outside its intended scope:

```colang
# config/rails.co

# Define what a user asking off-topic questions looks like
define user ask off topic
  "Can you help me with my homework?"
  "What's the weather today?"
  "Tell me a joke"
  "Write me a poem"

# Define the bot's response to off-topic questions
define bot refuse off topic
  "I'm only able to help with questions about our product documentation and technical support. Is there something specific about the product I can help you with?"

# Define the flow: when user asks off-topic, bot refuses
define flow refuse off topic
  user ask off topic
  bot refuse off topic
```

The flow definition uses example utterances — Colang uses semantic similarity (via an embedding model) to match user inputs to the defined message types, making it robust to paraphrases and variations.

### Safety Input Rails

```colang
# Detect harmful input
define user ask harmful
  "How do I make explosives?"
  "Help me hack into a system"
  "Write malware for me"

define bot refuse harmful request
  "I can't help with that request. If you have questions about legitimate security topics or need assistance with something else, I'm here to help."

define flow block harmful input
  user ask harmful
  bot refuse harmful request
  stop
```

The `stop` directive halts further processing — the LLM is never invoked for matched harmful inputs, preventing any possibility of a jailbreak succeeding.

### Factuality Rails with RAG Grounding

```colang
# Require LLM answers to be grounded in retrieved context
define flow check facts
  user ask question
  $facts = execute retrieve_relevant_facts(query=$last_user_message)
  $answer = execute llm generate
  $is_grounded = execute check answer grounded(answer=$answer, facts=$facts)
  
  if not $is_grounded
    bot say "I don't have reliable information about that in my knowledge base."
  else
    bot $answer
```

## Python Integration

### Basic Usage

```python
from nemoguardrails import RailsConfig, LLMRails

# Load configuration from directory
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Generate a response with guardrails applied
async def chat(user_message: str) -> str:
    response = await rails.generate_async(
        messages=[{"role": "user", "content": user_message}]
    )
    return response

# Example usage
import asyncio

response = asyncio.run(chat("What are the features of your product?"))
print(response)

# This will be caught by the off-topic rail:
blocked = asyncio.run(chat("Can you write me a poem?"))
print(blocked)  # Returns the refusal message
```

### Custom Actions

Actions are Python functions that can be invoked from Colang flows — enabling integration with external services, databases, and APIs:

```python
# config/actions.py
from nemoguardrails.actions import action

@action(is_system_action=True)
async def retrieve_relevant_facts(query: str) -> list[str]:
    """Retrieves relevant facts from a knowledge base for fact-checking."""
    # Connect to your vector store or knowledge base
    results = await vector_store.search(query, k=5)
    return [r.content for r in results]

@action(is_system_action=True)
async def check_answer_grounded(answer: str, facts: list[str]) -> bool:
    """Checks if an answer is supported by the provided facts."""
    grounding_prompt = f"""
    Facts: {facts}
    Answer: {answer}
    
    Is this answer fully supported by the provided facts? 
    Respond with only: YES or NO
    """
    result = await llm.generate(grounding_prompt)
    return result.strip().upper() == "YES"
```

Register custom actions when constructing the rails:

```python
from config.actions import retrieve_relevant_facts, check_answer_grounded

config = RailsConfig.from_path("./config")
rails = LLMRails(config)
rails.register_action(retrieve_relevant_facts)
rails.register_action(check_answer_grounded)
```

## LangChain Integration

NeMo Guardrails integrates with LangChain as a component in a chain:

```python
from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# Set up the LangChain model
llm = ChatOpenAI(model="gpt-4o-mini")

# Load guardrails config
config = RailsConfig.from_path("./config")

# Wrap the LLM with guardrails
guardrailed_llm = RunnableRails(config, llm=llm)

# Use as a standard LangChain runnable
from langchain.schema import HumanMessage

response = guardrailed_llm.invoke([HumanMessage(content="Tell me about the product.")])
print(response.content)
```

This wraps any LangChain-compatible LLM with the guardrail layer — allowing gradual adoption into existing LangChain pipelines.

## Self-Check Rails

NeMo Guardrails includes built-in **self-check rails** that use a separate LLM call to evaluate input and output safety:

```yaml
# config/config.yml
rails:
  input:
    flows:
      - self check input   # Built-in: asks LLM to evaluate input safety
  output:
    flows:
      - self check output  # Built-in: asks LLM to evaluate output safety
```

Self-check rails use configurable prompts to ask the LLM whether the input or output violates safety policies — providing a reasonable baseline without requiring manual Colang rail definitions for every safety scenario.

## Streaming Support

For streaming LLM outputs, NeMo Guardrails can buffer streamed tokens and apply output rails before returning:

```python
async def stream_with_guardrails(user_message: str):
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": user_message}]
    ):
        print(chunk, end="", flush=True)
```

Note that output rail evaluation requires the complete output — streaming and output rail checking involve a trade-off between latency (stream immediately) and safety (evaluate complete output first).

## Evaluation and Observability

NeMo Guardrails includes tooling for evaluating rail effectiveness:

```python
from nemoguardrails.eval import EvalConfig, evaluate

# Run evaluation against a test dataset
eval_results = await evaluate(
    config=rails_config,
    eval_config=EvalConfig.from_path("./eval_config.yml"),
    test_set_path="./test_cases.yml"
)

print(f"Pass rate: {eval_results.pass_rate:.2%}")
print(f"False positive rate: {eval_results.false_positive_rate:.2%}")
```

Track what fraction of legitimate queries are blocked (false positives) and what fraction of policy-violating queries pass through (false negatives) — optimizing the rail configuration for the deployment context.

## When to Use NeMo Guardrails

| Use Case | Suitable? |
|---|---|
| Customer support chatbot with topic restrictions | Excellent fit |
| Internal documentation Q&A with RAG grounding | Excellent fit |
| Open-ended creative writing assistant | Poor fit — rails would over-restrict |
| Medical or legal information with fact-checking | Good fit |
| High-throughput inference API (cost-sensitive) | Moderate — extra LLM calls add cost |
| Safety-critical applications requiring hard blocks | Good fit — more reliable than fine-tuning alone |

NeMo Guardrails is most valuable when the LLM's intended behavior can be clearly specified, when safety properties need to be auditable and explicit, and when the application has a defined scope rather than being an open-ended general assistant.
