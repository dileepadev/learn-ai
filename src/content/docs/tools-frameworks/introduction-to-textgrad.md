---
title: Introduction to TextGrad
description: A comprehensive introduction to TextGrad, the automatic differentiation framework for AI systems that treats LLM feedback as gradients, enabling end-to-end optimization of multi-component AI pipelines through natural language.
---

# Introduction to TextGrad

**TextGrad** (Yuksekgonul et al., 2024, Stanford) is an automatic differentiation framework for AI systems — but instead of computing numerical gradients, it propagates **natural language feedback** as gradients through compound AI systems. Just as PyTorch's autograd enables gradient-based optimization of neural networks, TextGrad enables **gradient-based optimization of any system whose components can be represented as LLM-callable functions**, including prompts, code, reasoning chains, molecular structures, and agent policies.

## Core Abstraction

TextGrad mirrors PyTorch's API design:

| PyTorch | TextGrad | Semantics |
|---|---|---|
| `torch.Tensor` | `textgrad.Variable` | A value with optional gradient |
| `loss.backward()` | `loss.backward()` | Propagate gradients backwards |
| `optimizer.step()` | `optimizer.step()` | Update variable using gradient |
| Numerical gradient | LLM critique text | The "direction of improvement" |
| Chain rule | LLM-based chain rule | Compose feedback across layers |

## Installation and Basic Setup

```bash
pip install textgrad
export OPENAI_API_KEY="your-key"
```

```python
import textgrad as tg

# Set the backward engine (LLM used to compute gradients)
tg.set_backward_engine("gpt-4o", override=True)
```

## Variables and Gradients

In TextGrad, a `Variable` wraps any string value. Setting `requires_grad=True` enables gradient tracking:

```python
import textgrad as tg

# A prompt we want to optimize
system_prompt = tg.Variable(
    "You are a helpful assistant. Answer questions concisely.",
    requires_grad=True,
    role_description="system prompt for a QA assistant",
)

# A fixed input (no gradient needed)
user_query = tg.Variable(
    "Explain the difference between Type I and Type II errors in statistics.",
    requires_grad=False,
    role_description="user question",
)
```

## Forward Pass: Calling LLMs as Functions

```python
# The LLM call is the "layer" — it maps Variable → Variable
llm = tg.BlackboxLLM("gpt-4o-mini")
response = llm(system_prompt, user_query)

print(response.value)
# → "Type I error is a false positive (rejecting a true null hypothesis)..."
```

## Loss Functions in Natural Language

Instead of cross-entropy, TextGrad uses an **evaluation loss** expressed in natural language:

```python
# Built-in evaluation loss for factual accuracy
eval_system_prompt = tg.Variable(
    (
        "You are a critical evaluator. Assess the response for accuracy, "
        "completeness, and clarity. Be strict. Provide detailed critique."
    ),
    requires_grad=False,
    role_description="evaluation instructions",
)

evaluation_fn = tg.TextLoss(eval_system_prompt)
loss = evaluation_fn(response)

print(loss.value)
# → "The response is mostly accurate but lacks mention of Type II error's
#    relationship to statistical power and sample size..."
```

## Backward Pass and Optimization

```python
# Compute textual gradients
loss.backward()

# The gradient is natural language feedback propagated back to system_prompt
print(system_prompt.gradients)
# → "The system prompt should instruct the assistant to include examples,
#    mention practical implications, and define technical terms clearly..."

# Optimize: TextGrad uses an LLM to apply the gradient as an edit
optimizer = tg.TGD(parameters=[system_prompt])
optimizer.step()

print(system_prompt.value)
# → "You are a helpful assistant specializing in statistics. Answer questions
#    with clear definitions, concrete examples, and practical implications..."
```

## Multi-Step Optimization Loop

```python
import textgrad as tg


def optimize_prompt(initial_prompt: str, eval_questions: list[str], n_steps: int = 5) -> str:
    """Optimize a system prompt using TextGrad."""
    tg.set_backward_engine("gpt-4o")

    prompt = tg.Variable(
        initial_prompt,
        requires_grad=True,
        role_description="system prompt",
    )
    optimizer = tg.TGD(parameters=[prompt])
    llm = tg.BlackboxLLM("gpt-4o-mini")
    evaluator = tg.TextLoss(tg.Variable(
        "Evaluate the response quality: accuracy, depth, clarity. Be critical.",
        requires_grad=False,
        role_description="evaluator",
    ))

    for step in range(n_steps):
        step_loss = None
        for question in eval_questions:
            q_var = tg.Variable(question, requires_grad=False, role_description="question")
            response = llm(prompt, q_var)
            loss = evaluator(response)
            if step_loss is None:
                step_loss = loss
            # Accumulate gradients across multiple questions
            loss.backward()

        print(f"Step {step+1} loss: {step_loss.value[:80]}...")
        optimizer.step()
        optimizer.zero_grad()

    return prompt.value
```

## Code Optimization

TextGrad can optimize code quality, correctness, and efficiency:

```python
import textgrad as tg

tg.set_backward_engine("gpt-4o")

# Code to optimize
code = tg.Variable(
    """
def find_duplicates(lst):
    seen = []
    dups = []
    for item in lst:
        if item in seen:
            dups.append(item)
        else:
            seen.append(item)
    return dups
""",
    requires_grad=True,
    role_description="Python function to find duplicates in a list",
)

# Test cases as loss signal
test_fn = tg.TextLoss(tg.Variable(
    (
        "Evaluate the Python code for correctness, time complexity, "
        "and Pythonic style. Focus on algorithmic efficiency."
    ),
    requires_grad=False,
    role_description="code evaluator",
))

optimizer = tg.TGD(parameters=[code])

for _ in range(3):
    loss = test_fn(code)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(code.value)
# → def find_duplicates(lst):
#       seen = set()
#       return list({x for x in lst if x in seen or seen.add(x)})
#   (or similar O(n) solution)
```

## Multi-Component Pipelines

TextGrad's power emerges in **compound AI systems** where gradients propagate through chains of LLM calls:

```python
import textgrad as tg

tg.set_backward_engine("gpt-4o")

# Two-stage pipeline: retriever prompt + answer prompt
retrieval_prompt = tg.Variable(
    "Extract the key entities and concepts from the question.",
    requires_grad=True,
    role_description="retrieval query formulation prompt",
)
answer_prompt = tg.Variable(
    "Answer the question based on the retrieved context.",
    requires_grad=True,
    role_description="answer generation prompt",
)

llm = tg.BlackboxLLM("gpt-4o-mini")
evaluator = tg.TextLoss(tg.Variable(
    "Evaluate answer correctness against the ground truth.",
    requires_grad=False,
    role_description="QA evaluator",
))

def rag_pipeline(question_var, context_var):
    # Stage 1: formulate retrieval query
    query = llm(retrieval_prompt, question_var)
    # Stage 2: generate answer using context
    combined = tg.Variable(
        f"Context: {context_var.value}\n\nQuery: {query.value}",
        requires_grad=False,
        role_description="context + query",
    )
    return llm(answer_prompt, combined)

# TextGrad propagates gradients through both stages
for question, context, _ in training_data:
    q = tg.Variable(question, requires_grad=False, role_description="question")
    c = tg.Variable(context, requires_grad=False, role_description="context")
    answer = rag_pipeline(q, c)
    loss = evaluator(answer)
    loss.backward()

optimizer = tg.TGD(parameters=[retrieval_prompt, answer_prompt])
optimizer.step()
```

## Molecule Optimization

TextGrad was demonstrated for drug discovery — optimizing molecular structures for binding affinity and drug-likeness:

```python
import textgrad as tg
from rdkit import Chem
from rdkit.Chem import Descriptors

tg.set_backward_engine("gpt-4o")

molecule = tg.Variable(
    "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin SMILES
    requires_grad=True,
    role_description="drug candidate SMILES string",
)

drug_evaluator = tg.TextLoss(tg.Variable(
    (
        "Evaluate this drug molecule (SMILES) for: "
        "1) Lipinski's Rule of Five compliance, "
        "2) predicted blood-brain barrier penetration, "
        "3) synthetic accessibility. Suggest specific modifications."
    ),
    requires_grad=False,
    role_description="drug discovery evaluator",
))

optimizer = tg.TGD(parameters=[molecule])

for step in range(5):
    loss = drug_evaluator(molecule)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step}: {molecule.value}")
```

## TextGrad vs. Alternative Optimization Approaches

| Approach | Gradient Type | Requires Differentiability | Scalability | Examples |
|---|---|---|---|---|
| TextGrad | Natural language | No — black-box | Any LLM-callable component | Prompt, code, agent |
| DSPy | Discrete optimization | No | Module-level | Prompt, few-shot examples |
| Automatic Prompt Engineer | Search-based | No | Prompt level only | Prompt optimization |
| RLHF / PPO | Numerical reward | Needs reward model | Policy level | LLM fine-tuning |
| Direct backprop | Numerical gradient | Yes — white-box | Embedding level | LoRA, fine-tuning |

## Limitations and Considerations

- **Cost**: each backward pass requires LLM API calls proportional to the number of gradient-requiring variables
- **Stochasticity**: LLM-based gradients are stochastic — results vary between runs
- **Convergence**: no formal convergence guarantees; performance depends on the backward engine quality
- **Scope**: optimizes discrete, symbolic outputs — not suitable for learning new world knowledge or fine-tuning weights

## Summary

TextGrad provides a compelling abstraction for optimizing compound AI systems where end-to-end numerical backpropagation is unavailable. By treating LLM-generated critiques as gradients and propagating them backwards through chains of LLM calls, TextGrad enables automated prompt optimization, code improvement, molecule design, and multi-agent workflow tuning — all without requiring access to model weights or differentiable operations. Its PyTorch-inspired API makes it accessible to ML practitioners, and its modular design supports complex multi-stage pipelines that would otherwise require expensive manual iteration.
