---
title: Introduction to Semantic Kernel
description: An overview of Microsoft's open-source SDK for integrating large language models into applications.
---

Semantic Kernel is an open-source SDK developed by Microsoft that makes it straightforward to build AI-powered applications by connecting large language models (LLMs) with existing code, plugins, and data. It is designed for developers who want to embed AI capabilities into enterprise software without rebuilding their entire stack.

## What Semantic Kernel Does

Semantic Kernel acts as an orchestration layer between your application logic and AI models. It handles the complexity of prompt management, memory, tool invocation, and multi-step reasoning, allowing developers to focus on building features rather than managing raw model API calls.

Core capabilities:

- **Plugins:** Wrapping native functions and semantic (prompt-based) functions as reusable tools the model can invoke.
- **Planners:** AI-driven orchestrators that decompose a high-level goal into a sequence of plugin calls to achieve it.
- **Memory:** Persistent and contextual storage that allows the AI to recall prior interactions, user preferences, or retrieved documents.
- **Connectors:** Built-in integrations with OpenAI, Azure OpenAI, Hugging Face, and vector databases such as Chroma, Pinecone, and Azure AI Search.

## Supported Languages

Semantic Kernel has first-class support for:

- **C#** (most mature, used in production by Microsoft internal teams)
- **Python**
- **Java**

## Key Concepts

### Kernel

The `Kernel` is the central object in Semantic Kernel. It holds model configuration, registered plugins, and services. Everything flows through the kernel.

```python
from semantic_kernel import Kernel

kernel = Kernel()
kernel.add_service(AzureChatCompletion(
    deployment_name="gpt-4o",
    endpoint="https://your-resource.openai.azure.com/",
    api_key="your-api-key"
))
```

### Plugins

A plugin is a collection of functions the AI can call. Functions can be:

- **Semantic functions:** Defined by a prompt template. The model fills in the template at runtime.
- **Native functions:** Standard code functions (Python, C#, Java) annotated so the kernel can expose them to the model.

```python
from semantic_kernel.functions import kernel_function

class WeatherPlugin:
    @kernel_function(description="Get the current weather for a city")
    def get_weather(self, city: str) -> str:
        return f"The weather in {city} is sunny and 22°C."
```

### Planners

Planners allow the kernel to automatically decide which sequence of plugin functions to call in order to satisfy a user request. Available planners include:

- **FunctionChoiceBehavior.Auto:** The model selects and calls functions as needed during conversation.
- **Stepwise planner:** The model reasons step-by-step and emits a structured plan before executing.

### Memory and Vector Search

Semantic Kernel integrates with vector databases to support retrieval-augmented generation (RAG). You can store documents, retrieve relevant chunks, and inject them into prompts automatically.

## How Semantic Kernel Relates to Other Frameworks

| Feature | Semantic Kernel | LangChain | LlamaIndex |
|---|---|---|---|
| Primary language | C#, Python, Java | Python, JS | Python |
| Enterprise focus | Strong | Moderate | Moderate |
| Planner / agent | Built-in | Agents module | Built-in query engines |
| Memory / RAG | Built-in | Built-in | Core feature |
| Microsoft integration | Native | Third-party | Third-party |

Semantic Kernel is a strong choice when your team is already working in C# or deploying on Azure, and when enterprise governance, security, and integration with Microsoft services are priorities.

## When to Use Semantic Kernel

- Building AI features into existing .NET enterprise applications
- Using Azure OpenAI with Azure Active Directory authentication
- Needing a well-supported, production-ready SDK with Microsoft backing
- Orchestrating multiple AI functions in a controlled, auditable workflow

## Getting Started

Install the Python package:

```bash
pip install semantic-kernel
```

Or add the NuGet package for .NET:

```bash
dotnet add package Microsoft.SemanticKernel
```

The [official documentation](https://learn.microsoft.com/semantic-kernel/overview/) and [GitHub repository](https://github.com/microsoft/semantic-kernel) provide quickstarts, samples, and API references.

## Summary

Semantic Kernel bridges the gap between raw LLM APIs and production-ready software. By providing plugins, planners, memory, and connectors as first-class primitives, it allows teams to build reliable, maintainable AI features that integrate naturally with existing enterprise codebases.
