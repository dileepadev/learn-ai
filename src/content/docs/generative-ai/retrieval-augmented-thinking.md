---
title: "Retrieval-Augmented Thinking (RAT): Grounding Reasoning with Retrieval"
description: A deep dive into Retrieval-Augmented Thinking (RAT), a technique that interleaves chain-of-thought reasoning with dynamic retrieval to improve accuracy and reduce hallucination in LLM reasoning tasks.
---

Chain-of-Thought (CoT) prompting dramatically improves LLM reasoning by decomposing complex problems into intermediate steps. But CoT has a fundamental limitation: the model reasons entirely from its parametric memory. When the correct answer requires up-to-date information, domain-specific facts, or precise numbers that the model was not trained on, CoT reasoning can confidently produce plausible but incorrect intermediate steps — and compound those errors through the rest of the reasoning chain.

**Retrieval-Augmented Thinking (RAT)**, proposed by Wang et al. (2024), addresses this by interleaving retrieval with the reasoning process itself: at each reasoning step, the model decides whether it needs external information, retrieves relevant documents, and incorporates the retrieved content before continuing.

## The Core Problem: Parametric Reasoning is Brittle

Consider this problem: *"What is the capital gain tax rate for a single filer in California earning $\$180,000$ in 2024, and how does it compare to the federal rate?"*

A standard CoT approach would have the model reason step-by-step from its training data. But:
- Tax rates change annually
- State and federal rates interact in complex ways
- The model may have been trained before the 2024 rates were published

Even with CoT, the model may produce confident but stale or incorrect figures. A standard RAG approach would retrieve documents upfront, but the model may not know exactly what to retrieve until it has reasoned partway through the problem.

RAT solves this by making retrieval a **dynamic, reasoning-time decision**.

## How RAT Works

The RAT framework operates as a loop:

```
1. Generate the first reasoning step from the question
2. Decide: does this step need external verification or facts?
3. If yes: formulate a retrieval query based on the current reasoning context
4. Retrieve and re-read: fetch top-k documents and integrate them
5. Revise the current reasoning step if the retrieved evidence contradicts it
6. Continue to the next reasoning step
7. Repeat until the final answer is reached
```

This is distinct from standard RAG, which retrieves once before reasoning, and from ReAct, which interleaves actions (including retrieval) with thought steps but does not specifically revise prior reasoning in light of new evidence.

### Formal Description

Let $Q$ be the question and $T = [t_1, t_2, \dots, t_n]$ be the chain of thought. At each step $i$:

$$t_i' = \text{LLM}(Q, t_1, \dots, t_{i-1}, \text{Retrieve}(q_i))$$

Where:
- $q_i$ is a retrieval query derived from the current reasoning context
- $\text{Retrieve}(q_i)$ returns top-$k$ documents from a knowledge base
- $t_i'$ is the revised reasoning step after reading the retrieved documents

The final answer is derived from the completed revised chain $T' = [t_1', t_2', \dots, t_n']$.

## Implementation

A basic RAT implementation using a tool-calling capable LLM:

```python
from dataclasses import dataclass
from typing import Optional
import openai

@dataclass
class RATStep:
    thought: str
    query: Optional[str]
    retrieved_docs: list[str]
    revised_thought: str

SYSTEM_PROMPT = """You are a careful reasoning assistant. When reasoning through a problem:
1. Write your initial thought for the current step.
2. If the thought requires factual verification, output a retrieval query in the format: RETRIEVE: <query>
3. After seeing retrieved documents, output a revised thought that incorporates the evidence.
4. Continue to the next step.
Always cite which retrieved document supports each claim."""

def rat_reasoning(question: str, retriever, max_steps: int = 8) -> list[RATStep]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nBegin step-by-step reasoning:"}
    ]
    steps = []

    for step_num in range(max_steps):
        # Generate initial thought
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stop=["RETRIEVE:", "\n\nStep"]
        )
        thought = response.choices[0].message.content

        # Check if retrieval is needed
        if "RETRIEVE:" in thought:
            # Parse query
            query_start = thought.index("RETRIEVE:") + len("RETRIEVE:")
            query = thought[query_start:].strip().split("\n")[0]

            # Retrieve documents
            docs = retriever.search(query, top_k=3)
            doc_context = "\n\n".join([f"[Doc {i+1}]: {d}" for i, d in enumerate(docs)])

            # Add retrieved docs and ask for revised thought
            messages.append({"role": "assistant", "content": thought})
            messages.append({
                "role": "user",
                "content": f"Retrieved documents:\n{doc_context}\n\nRevise your reasoning step:"
            })

            revised_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            revised_thought = revised_response.choices[0].message.content

            steps.append(RATStep(
                thought=thought, query=query,
                retrieved_docs=docs, revised_thought=revised_thought
            ))
            messages.append({"role": "assistant", "content": revised_thought})
        else:
            steps.append(RATStep(
                thought=thought, query=None,
                retrieved_docs=[], revised_thought=thought
            ))
            messages.append({"role": "assistant", "content": thought})

        # Check if we've reached the final answer
        if "final answer" in thought.lower() or "therefore" in thought.lower():
            break

    return steps
```

## RAT vs. Related Approaches

| Approach | When Retrieval Happens | Reasoning Revision | Best For |
|---|---|---|---|
| Standard RAG | Before reasoning (once) | No | Factual Q&A, document QA |
| ReAct | Interleaved (multiple) | No | Multi-hop reasoning, tool use |
| FLARE | During generation, when uncertain | Partial | Long-form generation |
| RAT | Interleaved + revision | Yes | Complex multi-step reasoning |
| IRCoT (Interleaved Retrieval CoT) | Per CoT step | Limited | Multi-hop QA |
| Self-RAG | On-demand with reflection tokens | Yes | General RAG with quality control |

RAT's distinguishing feature is **revision**: when retrieved evidence contradicts a prior reasoning step, RAT explicitly updates that step. Most other methods do not revise prior reasoning; they only condition future steps on new information.

## Retrieval Query Formulation

The quality of RAT's retrieval depends heavily on how well the LLM formulates queries from its reasoning context. Several strategies improve this:

**Step-back prompting:** Before formulating the retrieval query, the model first asks itself what general principle or concept the current step is invoking, then queries for that principle rather than the specific surface-level question.

**Hypothetical Document Embeddings (HyDE):** Instead of querying with a question, the model generates a hypothetical answer, then retrieves documents similar to that answer. This often returns more relevant results for factual questions.

**Multi-query retrieval:** Generate 3–5 different phrasings of the same retrieval need and take the union of results, reducing the impact of lexical mismatch.

```python
def formulate_rat_query(reasoning_context: str, current_step: str, llm) -> list[str]:
    """Generate multiple retrieval queries for the current reasoning step."""
    prompt = f"""Given this reasoning context:
{reasoning_context}

And this current reasoning step where I need factual verification:
{current_step}

Generate 3 distinct search queries that would retrieve the most relevant information.
Format: one query per line."""

    response = llm.complete(prompt)
    queries = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
    return queries[:3]
```

## When RAT Outperforms Standard CoT and RAG

RAT shows the most benefit on tasks that combine:

1. **Multi-hop reasoning**: Problems requiring connecting information from multiple sources
2. **Temporal sensitivity**: Questions about recent events, current statistics, or changing facts
3. **Domain-specific precision**: Medical dosing, legal statutes, financial regulations — where parametric memory may be outdated or imprecise
4. **Error propagation risk**: Long reasoning chains where early errors cascade

In the original RAT paper, the technique improves performance on:
- Medical question answering (MedQA): +6.7% over CoT
- Complex QA (StrategyQA): +5.2% over CoT
- Temporal reasoning: significant improvements on questions about post-training events

## Limitations and Practical Considerations

**Latency:** Each retrieval step adds network latency. A 6-step reasoning chain with retrieval at each step may add 2–4 seconds of latency compared to pure CoT.

**Retrieval quality bottleneck:** RAT's correctness is bounded by the retrieval system. Poor retrieval results in confident but wrong reasoning revisions.

**Verbosity:** Revised reasoning chains are significantly longer than standard CoT, increasing token costs.

**Circular reasoning risk:** If the retrieval corpus contains the same misinformation the model is trying to correct, RAT can reinforce rather than fix errors.

**Hallucinated retrieval needs:** Models sometimes generate unnecessary retrieval queries for facts they know correctly, wasting latency.

For production deployments, a hybrid approach often works best: use a fast router to classify questions as "retrieval needed" vs. "parametric reasoning sufficient," applying RAT only to the former class.

## Relation to Agentic Systems

RAT is conceptually a simplified agentic loop where the only tool is retrieval. More general frameworks like ReAct, Toolformer, and modern tool-use LLMs extend this to multiple tools (calculators, code interpreters, databases). RAT's specific contribution is the **revision** mechanism — the idea that not just future steps but prior reasoning can be updated in light of new evidence.

This revision dynamic is increasingly recognized as important for building reliable reasoning systems: a model that cannot update prior conclusions when confronted with contradictory evidence will compound errors rather than correct them.
