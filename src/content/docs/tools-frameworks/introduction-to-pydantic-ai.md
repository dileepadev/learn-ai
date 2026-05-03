---
title: Introduction to PydanticAI
description: Build type-safe AI agents with PydanticAI — the agent framework from the Pydantic team. Learn the Agent class with typed generics, RunContext dependency injection, tool registration with @agent.tool, structured result types, ModelRetry for self-correction, multi-agent composition, streaming, TestModel for unit testing, and multi-turn conversation with message history.
---

**PydanticAI** is a Python agent framework built by the Pydantic team — the creators of the most widely used data validation library in the Python ecosystem. It brings the same philosophy of type safety, validation, and developer ergonomics to AI agent development. Unlike frameworks that expose agents as JSON configuration or dynamic dictionaries, PydanticAI agents are strongly typed Python objects: inputs, outputs, tool parameters, and dependencies are all validated by Pydantic at runtime.

## Installation

```bash
pip install pydantic-ai
# Model-specific: install provider packages
pip install openai anthropic google-generativeai
```

## Core Concepts: The `Agent` Class

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Minimal agent: just a model and system prompt
simple_agent = Agent(
    model=OpenAIModel("gpt-4o-mini"),
    system_prompt="You are a helpful assistant. Be concise."
)

result = simple_agent.run_sync("What is the capital of Japan?")
print(result.output)  # "Tokyo"
print(result.usage)   # Usage(requests=1, request_tokens=26, response_tokens=2, ...)
```

## Typed Dependencies with `RunContext`

PydanticAI's dependency injection system allows tools and system prompts to access application state without global variables:

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

@dataclass
class DatabaseDeps:
    """Dependencies injected into every agent run."""
    db_connection: object          # database connection
    user_id: str                   # current user context
    user_tier: str                 # "free" | "pro" | "enterprise"
    max_results: int = 10

# Agent typed with its dependencies: Agent[DatabaseDeps, OutputType]
db_agent = Agent(
    model=OpenAIModel("gpt-4o"),
    deps_type=DatabaseDeps,
    result_type=str,
    system_prompt="You are a database query assistant."
)

@db_agent.system_prompt
def dynamic_system_prompt(ctx: RunContext[DatabaseDeps]) -> str:
    """System prompt can access dependencies to customize behavior per user."""
    return (
        f"You are assisting user {ctx.deps.user_id} with {ctx.deps.user_tier} tier access. "
        f"Always limit results to {ctx.deps.max_results} items unless asked for more."
    )


@db_agent.tool
async def search_records(ctx: RunContext[DatabaseDeps], query: str) -> list[dict]:
    """
    Search the database for records matching the query.
    
    Tool docstrings become the tool description sent to the model.
    Parameter types and names are inferred from the function signature.
    """
    # Access injected dependencies inside the tool
    db = ctx.deps.db_connection
    user = ctx.deps.user_id
    
    # In practice: actual database query
    records = await db.search(
        query=query,
        user_id=user,
        limit=ctx.deps.max_results
    )
    return [{"id": r.id, "title": r.title, "score": r.score} for r in records]


@db_agent.tool
async def get_record_detail(ctx: RunContext[DatabaseDeps], record_id: str) -> dict:
    """Retrieve detailed information for a specific record by ID."""
    record = await ctx.deps.db_connection.get(record_id, user_id=ctx.deps.user_id)
    if record is None:
        return {"error": f"Record {record_id} not found or access denied"}
    return record.to_dict()


# Running the agent with injected dependencies
async def main():
    deps = DatabaseDeps(
        db_connection=get_db_connection(),
        user_id="user_42",
        user_tier="pro",
        max_results=20
    )
    
    result = await db_agent.run(
        "Find all records about transformer architectures",
        deps=deps
    )
    print(result.output)
```

## Structured Result Types

The most powerful PydanticAI feature is returning structured Pydantic models as agent output — validated automatically:

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskExtraction(BaseModel):
    """Structured output for task extraction from natural language."""
    title: str = Field(description="Short title for the task (max 10 words)")
    description: str = Field(description="Full task description")
    priority: Priority
    due_date: Optional[str] = Field(
        None,
        description="Due date in ISO format (YYYY-MM-DD) if mentioned, else null"
    )
    assignee: Optional[str] = Field(None, description="Person assigned to the task")
    tags: list[str] = Field(default_factory=list, description="Relevant topic tags")
    estimated_hours: Optional[float] = Field(None, ge=0.1, le=1000)

task_agent = Agent(
    model=OpenAIModel("gpt-4o-mini"),
    result_type=TaskExtraction,   # model must return valid TaskExtraction
    system_prompt=(
        "Extract task information from user messages. "
        "Be precise about dates and priorities."
    )
)

result = task_agent.run_sync(
    "Can you remind the team to fix the login page bug before Friday? "
    "It's blocking all new signups — assign it to Sarah, should take about 3 hours."
)

task = result.output   # guaranteed to be TaskExtraction instance
print(task.title)             # "Fix login page bug"
print(task.priority)          # Priority.CRITICAL
print(task.assignee)          # "Sarah"
print(task.estimated_hours)   # 3.0
print(task.tags)              # ["bug", "authentication", "frontend"]
```

## Self-Correction with `ModelRetry`

When a tool encounters invalid input or business logic failures, raising `ModelRetry` causes the agent to try again with the error message as context:

```python
from pydantic_ai import ModelRetry

weather_agent = Agent(
    model=OpenAIModel("gpt-4o-mini"),
    result_type=str,
    system_prompt="Answer weather questions using the get_weather tool."
)

@weather_agent.tool
async def get_weather(ctx: RunContext[None], city: str, country_code: str) -> dict:
    """
    Get current weather for a city.
    
    Args:
        city: City name in English
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB', 'JP')
    """
    if len(country_code) != 2 or not country_code.isalpha():
        raise ModelRetry(
            f"Invalid country_code '{country_code}'. "
            f"Must be 2-letter ISO code like 'US', 'GB', 'JP'. Please retry with correct code."
        )
    
    response = await weather_api.get(city=city, country=country_code.upper())
    if response.status_code == 404:
        raise ModelRetry(
            f"City '{city}' not found in {country_code}. "
            f"Try a nearby larger city or check the spelling."
        )
    
    return response.json()
```

## Multi-Agent Composition

PydanticAI enables one agent to call another as a tool — building hierarchical multi-agent systems:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

# Specialized sub-agents
code_reviewer = Agent(
    model=AnthropicModel("claude-3-5-sonnet-20241022"),
    result_type=str,
    system_prompt=(
        "You are an expert code reviewer. Analyze code for bugs, security issues, "
        "and style violations. Return structured feedback."
    )
)

test_generator = Agent(
    model=OpenAIModel("gpt-4o"),
    result_type=str,
    system_prompt="Generate comprehensive pytest test cases for the provided code."
)

# Orchestrator agent that coordinates the sub-agents
orchestrator = Agent(
    model=OpenAIModel("gpt-4o"),
    result_type=str,
    system_prompt=(
        "You coordinate code review workflows. Use available tools to review code "
        "and generate tests, then synthesize the results."
    )
)

@orchestrator.tool
async def review_code(ctx: RunContext[None], code: str) -> str:
    """Review code quality and identify issues."""
    result = await code_reviewer.run(f"Review this code:\n\n```python\n{code}\n```")
    return result.output

@orchestrator.tool
async def generate_tests(ctx: RunContext[None], code: str) -> str:
    """Generate test cases for the provided code."""
    result = await test_generator.run(
        f"Generate pytest tests for:\n\n```python\n{code}\n```"
    )
    return result.output


async def full_review_workflow(code: str) -> str:
    result = await orchestrator.run(
        f"Please review this code and generate tests for it:\n\n```python\n{code}\n```"
    )
    return result.output
```

## Multi-Turn Conversations with Message History

```python
from pydantic_ai.messages import ModelMessagesTypeAdapter
import json

chat_agent = Agent(
    model=OpenAIModel("gpt-4o-mini"),
    system_prompt="You are a helpful programming tutor."
)

# First turn
result1 = chat_agent.run_sync("What is a Python decorator?")
print(result1.output)

# Second turn: pass previous messages to maintain context
result2 = chat_agent.run_sync(
    "Can you show me a practical example?",
    message_history=result1.new_messages()   # only add new messages each turn
)

# Serialize conversation for persistence
messages_json = ModelMessagesTypeAdapter.dump_json(result2.all_messages())
# Store in database, Redis, etc.

# Restore conversation from stored state
restored_messages = ModelMessagesTypeAdapter.validate_json(messages_json)
result3 = chat_agent.run_sync(
    "How are decorators different from class mixins?",
    message_history=restored_messages
)
```

## Streaming Responses

```python
async def stream_demo():
    async with chat_agent.run_stream("Explain backpropagation step by step.") as response:
        # Stream text tokens as they arrive
        async for chunk in response.stream_text(delta=True):
            print(chunk, end="", flush=True)
        
        print()
        print(f"\n\nTotal tokens: {response.usage().total_tokens}")
```

## Unit Testing with `TestModel`

```python
from pydantic_ai.models.test import TestModel
from pydantic_ai import capture_run_messages

def test_task_agent_extracts_priority():
    """Unit test without making real API calls."""
    
    # TestModel returns predictable, configurable responses
    with task_agent.override(model=TestModel()):
        result = task_agent.run_sync(
            "Fix the critical production outage immediately"
        )
    
    # TestModel returns a valid TaskExtraction with default values
    assert isinstance(result.output, TaskExtraction)

def test_tool_is_called():
    """Verify that the correct tool is invoked for a given input."""
    with capture_run_messages() as messages:
        with db_agent.override(model=TestModel()):
            result = db_agent.run_sync("Find transformer papers", deps=mock_deps)
    
    # Inspect the message sequence to verify tool call behavior
    tool_calls = [m for m in messages if hasattr(m, "tool_name")]
    assert any(tc.tool_name == "search_records" for tc in tool_calls)
```

## Supported Models

| Provider | Model class | Example models |
| --- | --- | --- |
| OpenAI | `OpenAIModel` | gpt-4o, gpt-4o-mini, o1, o3-mini |
| Anthropic | `AnthropicModel` | claude-3-5-sonnet, claude-3-5-haiku |
| Google | `GoogleModel` | gemini-2.0-flash, gemini-1.5-pro |
| Groq | `GroqModel` | llama-3.3-70b, mixtral-8x7b |
| Mistral | `MistralModel` | mistral-large, mistral-small |
| Ollama | `OpenAIModel` (local) | llama3.2, qwen2.5-coder |
| Bedrock | `BedrockModel` | Claude via AWS Bedrock |

PydanticAI's core advantage over alternatives like CrewAI and AutoGen is its emphasis on testability and type safety. The `TestModel` enables deterministic unit tests without API costs, `RunContext` dependency injection makes agents portable across different environments, and the Pydantic result validation gives confidence that agents produce correctly-typed outputs — critical for integrating agent outputs into downstream application logic.
