---
title: Introduction to smolagents
description: Get started with HuggingFace's smolagents — the minimal, code-first agent framework where agents write and execute Python code to call tools rather than structured JSON payloads. Learn CodeAgent vs ToolCallingAgent, custom tool definition with @tool decorator, multi-agent orchestration with ManagedAgent, and how smolagents' design philosophy differs from LangChain and AutoGen.
---

**smolagents** is HuggingFace's minimal agent framework built on a single core insight: **the best way for a language model to use tools is to write Python code**. Rather than generating structured JSON tool calls (the approach of LangChain and OpenAI's function calling), a `CodeAgent` writes actual Python code that calls your functions, runs loops, handles conditionals, and assembles results — then executes that code in a sandboxed environment.

The framework is intentionally small: the core is under 1,000 lines of Python. This makes it easy to understand, debug, and extend — a deliberate contrast to frameworks that have grown complex enough to require their own documentation site to explain their own internals.

## Installation and Quick Start

```bash
pip install smolagents
# For web search and Gradio UI support:
pip install smolagents[all]
# Optional: E2B sandbox for secure code execution
pip install e2b
```

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# Create a CodeAgent with web search capability
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
)

# Run the agent on a task
result = agent.run(
    "What is the current world record for the 100m sprint, "
    "and how does it compare to the average human running speed?"
)

print(result)
```

The agent will write Python code like:

```python
result = web_search("current world record 100m sprint 2024")
record = result[0]["content"]
avg_speed = web_search("average human running speed km/h")
print(f"Record: {record}\nAverage: {avg_speed}")
```

…and execute it, using the output to form a final answer.

## Two Agent Types

### CodeAgent

The flagship agent. Uses the model to write Python code at each step, which is executed in a Python interpreter. Code can call tools, manipulate data, run computations, and build on results from previous steps.

```python
from smolagents import CodeAgent, tool, HfApiModel
import torch
import numpy as np

@tool
def compute_statistics(numbers: list[float]) -> dict:
    """
    Compute basic statistics for a list of numbers.
    
    Args:
        numbers: A list of numerical values to analyze.
    
    Returns:
        Dictionary containing mean, median, std, min, and max.
    """
    arr = np.array(numbers)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": len(arr)
    }


@tool
def query_database(sql: str, database_name: str = "sales") -> list[dict]:
    """
    Execute a read-only SQL query against the specified database.
    
    Args:
        sql: Valid SQL SELECT statement to execute.
        database_name: Name of the database to query ("sales", "inventory", "users").
    
    Returns:
        List of row dictionaries with column names as keys.
    """
    # In production: connect to actual database with read-only credentials
    # Here: mock data for demonstration
    mock_data = {
        "sales": [
            {"date": "2024-01", "revenue": 125000, "units": 450},
            {"date": "2024-02", "revenue": 118000, "units": 420},
            {"date": "2024-03", "revenue": 142000, "units": 510},
        ]
    }
    return mock_data.get(database_name, [])


# Code agents can combine tools in complex ways —
# the model writes code that calls these as regular Python functions
agent = CodeAgent(
    tools=[compute_statistics, query_database],
    model=HfApiModel("meta-llama/Llama-3.3-70B-Instruct"),
    max_steps=10
)

result = agent.run(
    "Retrieve the sales data and analyze the revenue trend. "
    "Compute statistics and determine if revenue is growing."
)
```

### ToolCallingAgent

Uses structured JSON-like tool calls (similar to OpenAI function calling). More constrained than CodeAgent — each step selects one tool and provides arguments as structured data. Better for tasks where code execution is undesirable or where the environment doesn't allow running arbitrary Python:

```python
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, LiteLLMModel

agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=LiteLLMModel("anthropic/claude-3-5-sonnet-20241022"),
    max_steps=5
)

result = agent.run("Find the latest research on quantum computing error correction")
```

## Custom Tool Definition

Tools are defined with the `@tool` decorator or by subclassing the `Tool` class. The docstring is **critical**: smolagents uses it as the tool description passed to the model, so it must clearly specify what the tool does, its argument types, and return format.

```python
from smolagents import Tool
import requests

class WeatherTool(Tool):
    """
    Subclassing Tool gives more control than @tool for complex cases:
    - Custom __init__ for connection setup (API clients, database connections)
    - State management between calls (caching, rate limiting)
    - More complex argument schemas
    """
    
    name = "get_weather"
    description = """
    Retrieve current weather conditions for a specified city.
    Returns temperature in Celsius, humidity percentage, and a description.
    Use this when the user asks about current weather or forecasts.
    """
    inputs = {
        "city": {
            "type": "string",
            "description": "The city name to get weather for (e.g., 'London', 'Tokyo')"
        },
        "country_code": {
            "type": "string",
            "description": "Optional ISO 3166-1 alpha-2 country code (e.g., 'GB', 'JP')",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or "demo"   # demo key for wttr.in (no auth needed)

    def forward(self, city: str, country_code: str = None) -> str:
        """Execute the tool. smolagents calls this method when the agent invokes the tool."""
        location = f"{city},{country_code}" if country_code else city
        
        try:
            response = requests.get(
                f"https://wttr.in/{location}?format=j1",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            current = data["current_condition"][0]
            temp_c = current["temp_C"]
            humidity = current["humidity"]
            description = current["weatherDesc"][0]["value"]
            
            return (
                f"Weather in {city}: {description}, "
                f"Temperature: {temp_c}°C, Humidity: {humidity}%"
            )
        except Exception as e:
            return f"Could not retrieve weather for {city}: {str(e)}"
```

## Multi-Agent Orchestration with ManagedAgent

smolagents supports hierarchical multi-agent systems where agents manage other agents as tools. A **manager agent** delegates subtasks to specialized **managed agents**, each with their own tool set:

```python
from smolagents import CodeAgent, ToolCallingAgent, ManagedAgent, HfApiModel
from smolagents import DuckDuckGoSearchTool, VisitWebpageTool

model = HfApiModel("Qwen/Qwen2.5-72B-Instruct")

# Specialist agent 1: Web research
research_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="research_agent",
    description="Searches the web and retrieves information from web pages. "
                "Use this for factual research tasks."
)

# Specialist agent 2: Data analysis
analysis_agent = CodeAgent(
    tools=[compute_statistics],   # from earlier example
    model=model,
    name="analysis_agent",
    description="Performs data analysis, statistical computations, and "
                "synthesizes quantitative information."
)

# Wrap specialists as managed agents (agents-as-tools)
managed_researcher = ManagedAgent(
    agent=research_agent,
    name="researcher",
    description="A web research agent. Provide it with research questions "
                "and it will search the web and return synthesized findings."
)

managed_analyst = ManagedAgent(
    agent=analysis_agent,
    name="analyst",
    description="A data analysis agent. Provide it with data and questions "
                "and it will compute statistics and provide insights."
)

# Orchestrator: manages both specialists
orchestrator = CodeAgent(
    tools=[managed_researcher, managed_analyst],
    model=model,
    max_steps=15
)

result = orchestrator.run(
    "Research the market share of the top 5 cloud providers in 2024, "
    "then analyze the data to identify which provider is growing fastest "
    "and compute the concentration ratio (HHI) of the market."
)
```

## Model Backends

smolagents supports multiple LLM backends with a consistent interface:

```python
from smolagents import HfApiModel, LiteLLMModel, TransformersModel

# HuggingFace Inference API (cloud)
model = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token="hf_..."
)

# Any provider via LiteLLM (OpenAI, Anthropic, Mistral, Groq, etc.)
model = LiteLLMModel(
    model_id="openai/gpt-4o",
    api_key="sk-..."
)

# Local model via Hugging Face Transformers
model = TransformersModel(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
```

## GradioUI for Instant Chat Interfaces

Any smolagents agent can be exposed as an interactive chat UI with two lines:

```python
from smolagents import GradioUI

GradioUI(agent).launch()
```

This creates a browser-accessible chat interface with tool call visualization — useful for demos, stakeholder access to agent workflows, and human-in-the-loop supervision.

## Security: E2B Sandbox

Code execution in untrusted environments requires sandboxing. smolagents integrates with **E2B** (e2b.dev) cloud sandboxes for executing generated code in isolated containers:

```python
from smolagents import CodeAgent, HfApiModel
from smolagents.sandbox import E2BSandbox

# Code runs in an isolated cloud container, not the local Python process
agent = CodeAgent(
    tools=[...],
    model=HfApiModel("..."),
    executor=E2BSandbox(api_key="e2b_...")
)
```

## Comparison with Other Agent Frameworks

| Feature | smolagents | LangChain | AutoGen | crewAI |
| --- | --- | --- | --- | --- |
| Core philosophy | Code execution | Chains & LCELs | Multi-agent chat | Role-based crews |
| Tool calling style | Python code | JSON/function call | JSON/function call | JSON/function call |
| Codebase size | ~1,000 lines | 100,000+ lines | 30,000+ lines | 20,000+ lines |
| Multi-agent | ManagedAgent | LangGraph | GroupChat | Crew |
| Local models | Yes | Yes | Yes | Yes |
| Code sandbox | E2B | No built-in | No built-in | No built-in |
| Best for | Code-heavy tasks | Complex pipelines | Conversational | Role-based workflows |

smolagents occupies a specific niche: problems where the solution naturally involves writing code (data analysis, file manipulation, web scraping, mathematical computation). For these tasks, code-first agents are demonstrably more capable than JSON-calling agents — the model can use Python's full expressiveness rather than being limited to a predefined set of tool signatures.
