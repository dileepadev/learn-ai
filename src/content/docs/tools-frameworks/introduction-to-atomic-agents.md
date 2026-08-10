---
title: "Introduction to Atomic Agents"
description: Learn about Atomic Agents — a lightweight, composable framework for building predictable and testable AI agent systems using atomic, schema-validated components with full type safety.
---

Most AI agent frameworks optimize for rapid prototyping — they make it easy to connect tools, chain prompts, and get something working in minutes. But they often sacrifice predictability, testability, and maintainability. As agent systems move into production, these tradeoffs become painful: unpredictable behavior, untestable components, and brittle prompt chains that break when models are updated.

**Atomic Agents** is a Python framework built around the opposite philosophy: minimize magic, maximize explicitness. Every component is a small, independently testable unit with strict input/output schemas. Agent pipelines are explicit graphs, not hidden chains. And because every interaction with the LLM is schema-validated, you get structural guarantees about outputs that traditional prompt engineering cannot provide.

## Core Design Principles

Atomic Agents is built on four principles:

1. **Atomicity:** Each agent does one thing. Complex behaviors emerge from composing simple agents, not from building monolithic agents with many responsibilities.

2. **Schema-first:** All agent inputs and outputs are defined as Pydantic models. The LLM is always instructed to produce structured output conforming to these schemas.

3. **No implicit state:** Agents do not maintain hidden state between calls. All context is explicit and inspectable.

4. **Tool composability:** Tools are plain Python functions decorated with type hints. No special framework magic is needed to make a function into a tool.

## Installation and Basic Setup

```bash
pip install atomic-agents
```

Atomic Agents depends on `instructor` for structured output extraction and `pydantic` for schema validation:

```python
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
import instructor
import openai

# Initialize the instructor-patched client
client = instructor.from_openai(openai.OpenAI())
```

## Defining Agent Schemas

Every agent has an explicit `InputSchema` and `OutputSchema`:

```python
from pydantic import BaseModel, Field
from atomic_agents import BaseIOSchema

class SentimentAnalysisInput(BaseIOSchema):
    """Input to the sentiment analysis agent."""
    text: str = Field(..., description="The text to analyze for sentiment")
    language: str = Field(default="en", description="Language of the text (ISO 639-1)")

class SentimentAnalysisOutput(BaseIOSchema):
    """Output of the sentiment analysis agent."""
    sentiment: str = Field(
        ...,
        description="Overall sentiment: positive, negative, or neutral"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    key_phrases: list[str] = Field(
        default_factory=list,
        description="Key phrases contributing to the sentiment"
    )
    explanation: str = Field(..., description="Brief explanation of the classification")
```

These schemas serve multiple purposes: they guide the LLM via the field descriptions, they validate the output, and they form the contract between agents in a pipeline.

## Creating an Atomic Agent

```python
from atomic_agents import AtomicAgent, AgentConfig

# Build the agent
sentiment_agent = AtomicAgent(
    config=AgentConfig(
        client=client,
        model="gpt-4o-mini",
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are an expert sentiment analysis system.",
                "Analyze the sentiment of text accurately and provide structured output."
            ],
            steps=[
                "Read the provided text carefully.",
                "Identify the overall sentiment and key emotional phrases.",
                "Assign a confidence score based on clarity of sentiment signals."
            ],
            output_instructions=[
                "Output must be a single sentiment classification.",
                "Provide 2-5 key phrases that influenced the classification.",
                "Confidence should reflect how clear-cut the sentiment is."
            ]
        ),
        input_schema=SentimentAnalysisInput,
        output_schema=SentimentAnalysisOutput,
    )
)

# Use the agent
result = sentiment_agent.run(
    SentimentAnalysisInput(
        text="The product exceeded all my expectations! Absolutely fantastic quality.",
        language="en"
    )
)

print(result.sentiment)      # "positive"
print(result.confidence)     # 0.97
print(result.key_phrases)    # ["exceeded all my expectations", "Absolutely fantastic quality"]
```

The output is a fully validated Pydantic object — not a raw string that needs parsing.

## Composing Agents into Pipelines

Atomic Agents shine when composed into multi-step pipelines:

```python
class TextClassificationInput(BaseIOSchema):
    text: str
    categories: list[str]

class TextClassificationOutput(BaseIOSchema):
    category: str
    confidence: float
    reasoning: str

class TranslationInput(BaseIOSchema):
    text: str
    target_language: str

class TranslationOutput(BaseIOSchema):
    translated_text: str
    source_language_detected: str

# Create specialized agents
classifier = AtomicAgent(config=AgentConfig(
    client=client, model="gpt-4o-mini",
    input_schema=TextClassificationInput,
    output_schema=TextClassificationOutput,
    system_prompt_generator=SystemPromptGenerator(
        background=["You are an expert text classifier."],
        steps=["Read the text", "Classify into one of the provided categories"],
        output_instructions=["Output the most likely category with confidence"]
    )
))

translator = AtomicAgent(config=AgentConfig(
    client=client, model="gpt-4o-mini",
    input_schema=TranslationInput,
    output_schema=TranslationOutput,
    system_prompt_generator=SystemPromptGenerator(
        background=["You are a professional translator."],
        steps=["Translate the text accurately", "Detect the source language"],
        output_instructions=["Output the complete translated text"]
    )
))

# Pipeline: classify → translate if needed
def classify_and_translate(text: str, target_language: str = "en"):
    classification = classifier.run(TextClassificationInput(
        text=text,
        categories=["support_request", "complaint", "inquiry", "feedback"]
    ))

    if classification.category in ["support_request", "complaint"]:
        translation = translator.run(TranslationInput(
            text=text,
            target_language=target_language
        ))
        return classification, translation

    return classification, None
```

Each agent is independently testable and replaceable.

## Memory and Context Management

Atomic Agents provides explicit memory components rather than hidden conversation history:

```python
from atomic_agents.lib.components.agent_memory import AgentMemory

# Create memory that persists across agent calls
memory = AgentMemory(max_messages=20)

# Attach memory to an agent
agent_with_memory = AtomicAgent(config=AgentConfig(
    client=client,
    model="gpt-4o",
    memory=memory,
    input_schema=ChatInput,
    output_schema=ChatOutput,
    system_prompt_generator=system_prompt_gen
))

# Memory is inspectable at any point
print(memory.get_history())
memory.reset()  # Explicit reset — no hidden state
```

## Tool Use in Atomic Agents

Tools are plain Python functions annotated with Pydantic models:

```python
from atomic_agents import tool
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str
    country_code: str = "US"

class WeatherOutput(BaseModel):
    temperature_celsius: float
    condition: str
    humidity_percent: int

@tool
def get_weather(params: WeatherInput) -> WeatherOutput:
    """Fetch current weather for a city."""
    # Real implementation would call a weather API
    return WeatherOutput(
        temperature_celsius=22.5,
        condition="Partly cloudy",
        humidity_percent=65
    )

# Register tools with an agent
weather_agent = AtomicAgent(config=AgentConfig(
    client=client,
    model="gpt-4o",
    tools=[get_weather],
    input_schema=WeatherQueryInput,
    output_schema=WeatherQueryOutput,
    system_prompt_generator=...
))
```

Tools are type-safe: the agent receives strongly-typed inputs and returns strongly-typed outputs. Mismatches raise validation errors immediately.

## Testing Atomic Agents

The schema-first design makes testing straightforward:

```python
import pytest
from unittest.mock import MagicMock, patch

def test_sentiment_agent_output_schema():
    """Verify output schema is correctly enforced."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = SentimentAnalysisOutput(
        sentiment="positive",
        confidence=0.95,
        key_phrases=["great product"],
        explanation="Clear positive language"
    )

    agent = AtomicAgent(config=AgentConfig(
        client=mock_client,
        model="gpt-4o-mini",
        input_schema=SentimentAnalysisInput,
        output_schema=SentimentAnalysisOutput,
        system_prompt_generator=...
    ))

    result = agent.run(SentimentAnalysisInput(text="Great product!"))

    assert isinstance(result, SentimentAnalysisOutput)
    assert 0.0 <= result.confidence <= 1.0
    assert result.sentiment in ["positive", "negative", "neutral"]

def test_pipeline_data_flow():
    """Test that pipeline correctly passes data between agents."""
    classification = classifier.run(TextClassificationInput(
        text="My order is delayed", categories=["complaint", "inquiry"]
    ))
    assert classification.category in ["complaint", "inquiry"]
    assert isinstance(classification.confidence, float)
```

Because all inputs and outputs are Pydantic models, you can mock at the schema level without mocking LLM API calls in most cases.

## Atomic Agents vs. Other Frameworks

| Feature | Atomic Agents | LangChain | LlamaIndex | CrewAI |
|---|---|---|---|---|
| Schema validation | Pydantic always | Optional | Optional | Limited |
| Testability | High (explicit contracts) | Medium | Medium | Low |
| Hidden state | None | Present | Present | Present |
| Learning curve | Low | High | Medium | Medium |
| Streaming support | Via instructor | Native | Native | Limited |
| Multi-agent | Explicit pipelines | Chains/Agents | Workflows | Role-based |

Atomic Agents sacrifices some convenience for predictability and testability — an explicit trade-off favoring production readiness over rapid prototyping speed.

## When to Choose Atomic Agents

Atomic Agents is well-suited for:

- **Production systems** where reliability and testability are non-negotiable
- **Teams with strong Python engineering culture** who value type safety and explicit contracts
- **Data pipelines** where structured output is essential (extraction, classification, enrichment)
- **Multi-agent systems** with complex data flows that need to be debugged and monitored
- **Applications requiring LLM output validation** before downstream processing

It is less ideal for:

- Quick prototypes where development speed matters more than reliability
- Conversational applications with highly variable unstructured output requirements
- Teams not already using Pydantic or instructor

## Getting Started

The framework is actively maintained and well-documented:

```bash
pip install atomic-agents
```

- GitHub: [BrainBlend-AI/atomic-agents](https://github.com/BrainBlend-AI/atomic-agents)
- Documentation: [atomic-agents.ai](https://atomic-agents.ai)

Atomic Agents represents the "boring engineering" school of AI agent development: prioritize explicitness, type safety, and testability over magic and convenience. For teams building AI systems that need to survive contact with production, this philosophy pays significant dividends.
