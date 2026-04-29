---
title: Prompt Chaining in LLM Applications
description: Learn how to compose sequences of LLM calls — where the output of one prompt feeds the input of the next — covering sequential, conditional, and parallel chain patterns, error handling, when to chain versus use a single prompt, and practical implementation strategies.
---

**Prompt chaining** is the practice of decomposing a complex task into a sequence of simpler LLM calls, where the output of each step becomes part of the input to the next. Rather than asking a single, complex prompt to do everything at once, chaining breaks the work into focused sub-tasks — each of which the model can perform more reliably than the combined task.

The fundamental insight: **LLMs perform better on clear, narrow tasks than on complex, multi-requirement tasks**. A prompt asking a model to "read this legal document, identify all obligations, check them for inconsistencies, summarize the risks, and output a structured JSON report" will produce worse results than four separate, focused prompts accomplishing each step in sequence.

Prompt chaining is distinct from:

- **Chain-of-thought prompting**: A reasoning technique within a single prompt (the model reasons step-by-step before answering).
- **Agentic systems**: Multi-step systems where the model decides which tools to call next (the model controls flow).
- **Multi-agent orchestration**: Multiple specialized agents collaborating with message passing.

In prompt chaining, the **developer controls the flow** — the sequence of calls is hard-coded or deterministic, not decided dynamically by the LLM.

## Why Chain Prompts?

**Context management**: Each LLM call works within a focused context window — only the relevant information for that step is included, reducing distraction.

**Error isolation**: Errors in step N can be caught and handled before they propagate to step N+1 — unlike a single monolithic prompt where a reasoning error early on corrupts the entire output.

**Modular testing**: Each step in a chain can be evaluated and improved independently.

**Specialization**: Different steps can use different models — an expensive frontier model for reasoning steps, a cheaper model for formatting or extraction.

**Parallel execution**: Independent steps can run concurrently, reducing total latency.

## Chain Patterns

### Sequential Chain

The most basic pattern — outputs flow linearly from one step to the next:

```python
import openai

client = openai.OpenAI()

def llm(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def summarize_then_answer(document: str, question: str) -> str:
    """
    Sequential chain: summarize a long document, then answer a question
    about the summary. Avoids losing information in a very long context.
    """
    # Step 1: Summarize the document
    summary = llm(
        system="You are a precise document summarizer. Capture all key facts and figures.",
        user=f"Summarize this document, preserving all quantitative data and key claims:\n\n{document}"
    )
    
    # Step 2: Answer the question using the summary
    answer = llm(
        system="Answer questions based strictly on the provided context.",
        user=f"Context:\n{summary}\n\nQuestion: {question}"
    )
    
    return answer
```

### Extract-Transform-Format Chain

A common pattern for structured data extraction:

```python
import json

def extract_transform_format(raw_text: str) -> dict:
    """
    3-step chain:
    1. Extract raw entities and relationships
    2. Validate and normalize the extracted data
    3. Format as structured JSON
    """
    
    # Step 1: Extract — focus purely on finding relevant information
    extracted = llm(
        system="Extract all mentioned companies, dates, financial figures, and events. "
               "Be exhaustive. Use bullet points.",
        user=f"Extract from:\n\n{raw_text}"
    )
    
    # Step 2: Validate — catch and fix extraction errors
    validated = llm(
        system="Review extracted data for accuracy and consistency. "
               "Remove duplicates. Flag any uncertain extractions with (?).",
        user=f"Original text:\n{raw_text}\n\nExtracted data:\n{extracted}\n\n"
             f"Clean and validate the extracted data."
    )
    
    # Step 3: Format — convert to structured JSON
    json_output = llm(
        system="Convert the provided data to valid JSON. "
               "Output ONLY the JSON object, no explanation.",
        user=f"Convert to JSON with keys: companies, dates, financials, events:\n\n{validated}"
    )
    
    return json.loads(json_output)
```

### Conditional Chain

Routing to different sub-chains based on the content of previous outputs:

```python
def classify_then_handle(user_message: str) -> str:
    """
    Classify the intent of a message, then route to a specialized handler.
    The routing logic is code — not LLM-controlled.
    """
    
    # Step 1: Classify intent
    intent = llm(
        system="Classify the user message into exactly one category: "
               "COMPLAINT, QUESTION, REFUND_REQUEST, COMPLIMENT, OTHER. "
               "Output only the category name.",
        user=user_message
    ).strip().upper()
    
    # Step 2: Route to specialized handler based on classification
    if intent == "COMPLAINT":
        return llm(
            system="You are an empathetic customer service agent handling complaints. "
                   "Acknowledge the issue, apologize, and offer concrete next steps.",
            user=user_message
        )
    elif intent == "REFUND_REQUEST":
        return llm(
            system="You are a customer service agent handling refund requests. "
                   "Ask for order number and reason, explain the 5-7 business day processing time.",
            user=user_message
        )
    elif intent == "QUESTION":
        return llm(
            system="You are a helpful support agent. Answer questions clearly and concisely.",
            user=user_message
        )
    else:
        return llm(
            system="You are a friendly customer service agent.",
            user=user_message
        )
```

### Parallel Chain

Independent sub-tasks run concurrently for reduced total latency:

```python
import asyncio
import openai

async_client = openai.AsyncOpenAI()

async def async_llm(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0
    )
    return response.choices[0].message.content

async def parallel_analysis(document: str) -> dict:
    """
    Analyze a document from multiple independent angles simultaneously.
    All three analyses run in parallel, then merge results.
    """
    
    # All three run concurrently — total time = max(t1, t2, t3), not t1+t2+t3
    sentiment_task = async_llm(
        "Rate the sentiment as POSITIVE, NEUTRAL, or NEGATIVE with a 1-sentence reason.",
        f"Document:\n{document}"
    )
    topics_task = async_llm(
        "List the 3 main topics covered as a comma-separated list.",
        f"Document:\n{document}"
    )
    action_items_task = async_llm(
        "List all action items or commitments mentioned. Use bullet points.",
        f"Document:\n{document}"
    )
    
    sentiment, topics, action_items = await asyncio.gather(
        sentiment_task, topics_task, action_items_task
    )
    
    # Merge step: synthesize parallel results
    synthesis = await async_llm(
        "Write a 2-sentence executive summary given the analysis below.",
        f"Sentiment: {sentiment}\nTopics: {topics}\nAction Items:\n{action_items}"
    )
    
    return {
        "sentiment": sentiment,
        "topics": topics,
        "action_items": action_items,
        "summary": synthesis
    }
```

### Iterative Refinement Chain

Progressively improve an output through multiple review-and-revise cycles:

```python
def iterative_refinement(task: str, content: str, iterations: int = 2) -> str:
    """
    Generate content, then refine it through multiple critique-and-revise cycles.
    """
    # Initial generation
    draft = llm(
        system=f"Complete the following task with high quality.",
        user=f"Task: {task}\n\nContent to work with:\n{content}"
    )
    
    for i in range(iterations):
        # Critique pass — find specific weaknesses
        critique = llm(
            system="You are a rigorous editor. Identify specific, concrete weaknesses. "
                   "Do not give general praise. List 3-5 specific improvements.",
            user=f"Task: {task}\n\nCurrent draft:\n{draft}\n\n"
                 f"What specific improvements would most improve this?"
        )
        
        # Revision pass — apply the critique
        draft = llm(
            system="Apply the suggested improvements precisely. Maintain strengths.",
            user=f"Current draft:\n{draft}\n\nCritique:\n{critique}\n\n"
                 f"Produce an improved version addressing all critiques."
        )
    
    return draft
```

## Output Parsing and Validation

Between chain steps, parse and validate LLM outputs before passing them downstream:

```python
import json
import re
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

def parse_llm_json(raw_output: str, schema: Type[T]) -> T:
    """
    Safely parse LLM JSON output with validation.
    Handles common LLM output issues: markdown code fences, trailing commas.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_output.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    try:
        data = json.loads(cleaned)
        return schema(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"LLM output failed validation: {e}\n\nRaw output:\n{raw_output}")

# Example usage with Pydantic schema
class ExtractedEntities(BaseModel):
    companies: list[str]
    dates: list[str]
    amounts: list[str]

def safe_extract_chain(document: str) -> ExtractedEntities:
    raw = llm(
        system="Output valid JSON matching: {companies: [], dates: [], amounts: []}",
        user=f"Extract from:\n{document}"
    )
    return parse_llm_json(raw, ExtractedEntities)
```

## When to Chain vs. Single Prompt

**Use chaining when**:

- The task naturally decomposes into distinct phases (extract → validate → format).
- Output from step N must be significantly transformed before step N+1.
- Different steps require different expertise or system prompts.
- You need intermediate outputs for debugging or auditing.
- Steps can run in parallel for latency savings.
- The combined prompt would exceed the context window.

**Avoid chaining when**:

- The task is truly atomic (summarize a paragraph, translate a sentence).
- The overhead of multiple API calls costs more than the quality benefit.
- Step dependencies are so tight that separation provides no logical clarity.
- You're building a prototype where simplicity matters more than optimization.

A useful rule of thumb: if you find yourself writing a single prompt with more than three distinct instructions ("first do X, then Y, then Z"), it's a candidate for a chain.

## Error Handling in Chains

Errors in early chain steps corrupt all subsequent steps. Strategies:

```python
def resilient_chain(document: str) -> dict:
    """Chain with per-step error handling and fallbacks."""
    
    # Step 1 with retry
    for attempt in range(3):
        try:
            extracted = llm("Extract key facts as bullet points.", document)
            if len(extracted) < 50:  # Sanity check: output too short
                raise ValueError("Extraction too short — likely failure")
            break
        except Exception as e:
            if attempt == 2:
                extracted = "Extraction failed. Use original document."
    
    # Step 2: proceed even with degraded step 1
    summary = llm(
        "Summarize the key facts.",
        f"Extracted facts:\n{extracted}\n\nOriginal document:\n{document[:2000]}"
    )
    
    return {"extraction": extracted, "summary": summary}
```

Prompt chaining is one of the most practical and underrated techniques in production LLM engineering — transforming unreliable single-step outputs into robust, debuggable, modular pipelines that consistently deliver high-quality results.
