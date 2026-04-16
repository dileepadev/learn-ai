---
title: Introduction to DSPy
description: Learn how DSPy replaces brittle prompt strings with composable, self-optimizing modules — enabling systematic programmatic optimization of LLM pipelines.
---

**DSPy** (Declarative Self-improving Language Programs, Stanford NLP) is a framework that radically rethinks how LLM applications are built. Instead of writing fixed prompt strings, DSPy lets you define the *structure* of what you want — signatures and modules — and then **automatically optimizes** the prompts, few-shot examples, and even model weights to achieve a target metric.

## The Problem with Prompt Engineering

Traditional LLM application development involves:

1. Manually writing prompts for each step.
2. Testing them, finding they break on edge cases.
3. Hand-crafting few-shot examples.
4. Repeating this for every change in model, data, or task.

This process is **fragile, manual, and task-specific**. When you change the model (GPT-4 → Claude → Llama) or the data distribution, you must re-engineer prompts from scratch.

DSPy treats prompts as **implementation details** to be discovered by optimization — not hand-authored artifacts.

## Core Concepts

### Signatures

A **Signature** declares the inputs and outputs of an LLM module using a type-annotated, docstring-style interface:

```python
import dspy

class Emotion(dspy.Signature):
    """Classify the emotion of a sentence."""
    sentence: str = dspy.InputField()
    emotion: Literal["joy", "sadness", "anger", "fear", "surprise"] = dspy.OutputField()
```

The signature says *what* the module should do, not *how* to prompt the model to do it. DSPy generates and refines the prompt from the signature.

### Modules

DSPy modules are analogous to neural network layers — composable building blocks that encapsulate LLM calls:

| Module | Behavior |
|---|---|
| `dspy.Predict` | Single LLM call with the signature |
| `dspy.ChainOfThought` | Adds a scratchpad for reasoning before the answer |
| `dspy.ReAct` | Implements reason-act loops with tool calls |
| `dspy.Retrieve` | Retrieves documents from a retriever |
| `dspy.MultiChainComparison` | Generates multiple chains and selects the best |

```python
class QA(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate_answer(question=question)
```

### Programs

DSPy programs are **Python classes** that compose modules the same way PyTorch composes `nn.Module` layers. The forward pass defines the data flow through LLM calls, retrievals, and logic.

## The Optimization Loop

This is DSPy's killer feature. A **Teleprompter** (optimizer) takes:

- A **program** (composed modules).
- A **training set** (input-output pairs or just inputs with a metric).
- A **metric** (a function that scores program output quality).

And produces a **compiled** version of the program with optimized prompts and few-shot demonstrations:

```python
from dspy.teleprompt import BootstrapFewShot

def metric(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()

optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
compiled_qa = optimizer.compile(QA(), trainset=train_data)
```

The optimizer runs the program on training examples, identifies successful traces, and uses them as few-shot demonstrations — automatically.

## Available Optimizers (Teleprompters)

| Optimizer | Description |
|---|---|
| `BootstrapFewShot` | Uses successful traces as few-shot demos |
| `BootstrapFewShotWithRandomSearch` | Searches over combinations of demos |
| `MIPRO` | Bayesian optimization over instruction + demo combinations |
| `COPRO` | Cross-validates and optimizes instruction candidates |
| `BootstrapFinetune` | Uses traces to fine-tune model weights (for open models) |

`MIPRO` is currently the recommended optimizer for most production use cases, offering the best balance of quality and optimization cost.

## RAG Pipeline Example

```python
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

lm = dspy.LM("openai/gpt-4o-mini")
rm = ChromadbRM(collection_name="docs", persist_directory="./chroma_db")
dspy.configure(lm=lm, rm=rm)

class RAGSignature(dspy.Signature):
    """Answer questions using retrieved context."""
    question: str = dspy.InputField()
    context: list[str] = dspy.InputField(desc="Retrieved passages")
    answer: str = dspy.OutputField()

class RAG(dspy.Module):
    def __init__(self, n_passages=3):
        self.retrieve = dspy.Retrieve(k=n_passages)
        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(question=question, context=context)

# Optimize
optimizer = BootstrapFewShot(metric=my_metric)
compiled_rag = optimizer.compile(RAG(), trainset=trainset)
```

After compilation, `compiled_rag` contains optimized prompts tailored to your retriever and task — without a single manually written prompt.

## DSPy vs. LangChain / LlamaIndex

| | LangChain / LlamaIndex | DSPy |
|---|---|---|
| Prompts | Manually authored | Auto-optimized |
| Composition | Chain/pipeline abstraction | Module + forward pass |
| Optimization | None | First-class (teleprompters) |
| Portability | Some | High — signatures are model-agnostic |
| Learning curve | Moderate | Higher (new paradigm) |
| Best for | Fast prototyping | Production, multi-step pipelines |

DSPy is not a replacement for every LLM framework — but for multi-step pipelines where output quality matters, its optimization-first approach consistently outperforms hand-crafted prompt engineering.

## When to Use DSPy

- **Multi-step pipelines** with interdependent LLM calls (classification → retrieval → generation → verification).
- **Quality-sensitive applications** where manually tuning prompts is too slow or unreliable.
- **Model portability** — same program, different models. DSPy re-optimizes automatically.
- **Research pipelines** — DSPy's programmatic interface is ideal for systematic experiments.

## Getting Started

```bash
pip install dspy-ai
```

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini", api_key="...")
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")
response = qa(question="What is the capital of France?")
print(response.answer)  # Paris
```

DSPy represents a paradigm shift: from *prompt engineering* to *program optimization* — treating LLM applications with the same rigor we apply to machine learning models. As models multiply and pipelines grow more complex, this systematic approach becomes increasingly essential.
