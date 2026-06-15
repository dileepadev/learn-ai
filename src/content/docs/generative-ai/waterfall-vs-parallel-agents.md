---
title: Waterfall vs. Parallel Agent Architectures
description: Comparing sequential (waterfall) and parallel agent execution patterns — when to use each, trade-offs in latency, cost, and correctness, and how to combine them.
---

When building multi-agent AI systems, one of the first architectural decisions is how agents relate to each other: do they run one after another, or simultaneously? These two fundamental patterns — waterfall (sequential) and parallel — have very different trade-offs.

## Waterfall (Sequential) Agents

In a waterfall architecture, agents form a pipeline. Each agent waits for the previous one to complete and uses its output as input.

```
User Query → Agent A → Agent B → Agent C → Final Answer
```

### When Waterfall Makes Sense

- **Strict data dependencies:** Agent B genuinely needs Agent A's output to proceed. A researcher agent must finish before a writer agent can draft from its findings.
- **Quality gates:** Each stage validates or enriches the previous one. A planning agent, then an execution agent, then a review agent.
- **Simpler orchestration:** Sequential logic is easier to implement, debug, and reason about.
- **State must be consistent:** When agents share state and concurrent writes would create conflicts.

### Waterfall Example

```
1. Retrieval Agent  — fetches relevant documents
2. Extraction Agent — pulls structured data from documents
3. Analysis Agent   — performs calculations on extracted data
4. Report Agent     — writes a narrative from the analysis
```

Each step is blocked until the prior completes.

### Latency Profile
Total latency = sum of all agent execution times. If each takes 5 seconds, a 4-step pipeline takes ~20 seconds minimum.

---

## Parallel Agents

In a parallel architecture, multiple agents run concurrently and their results are merged by an aggregator.

```
User Query → Agent A ─┐
           → Agent B ─┼→ Aggregator → Final Answer
           → Agent C ─┘
```

### When Parallel Makes Sense

- **Independent sub-tasks:** Each agent can work without the others' output. Searching three different data sources simultaneously is the canonical example.
- **Redundancy / consensus:** Run the same task across multiple agents and vote or select the best answer. Reduces variance and catches hallucinations.
- **Time-sensitive applications:** Latency equals the slowest single agent, not the sum — a 4× speedup if four equal-duration tasks run in parallel.
- **Ensemble approaches:** Different model types or prompts in parallel, then merge outputs.

### Parallel Example

```
User: "What are the latest developments in quantum computing?"

→ Web Search Agent  (searches Arxiv, news)
→ Database Agent    (queries internal knowledge base)
→ Expert Agent      (generates from model's training knowledge)
   ↓ all complete
→ Synthesis Agent   (merges and deduplicates results)
```

### Latency Profile
Total latency ≈ max(all agent execution times) + aggregation time.

---

## Hybrid Architectures

Most real systems combine both patterns: parallel within a stage, sequential across stages.

```
Stage 1 (parallel): Research Agent A + Research Agent B + Research Agent C
         ↓
Stage 2 (sequential): Synthesis Agent
         ↓
Stage 3 (parallel): Draft Writer + Citation Formatter
         ↓
Stage 4 (sequential): Final Review Agent
```

This is sometimes called a **DAG (Directed Acyclic Graph)** architecture — the natural generalization of both patterns.

## Key Trade-offs

| Dimension | Waterfall | Parallel |
|-----------|-----------|----------|
| Latency | High (sum of stages) | Low (max of agents) |
| Cost | Lower (agents run once) | Higher (multiple agents run) |
| Correctness | Easier (clear data flow) | Harder (aggregation logic) |
| Debugging | Simpler (linear trace) | Harder (concurrent traces) |
| Scalability | Limited | High |

## Aggregation Strategies for Parallel Agents

When parallel agents return different outputs, merging them requires care:

- **First-wins:** Use the first successful response, discard others. Fast but loses information.
- **Voting / majority:** Run identical agents multiple times, return the most common answer. Good for factual classification.
- **Merge and deduplicate:** Combine all results, removing duplicates. Good for search and retrieval.
- **LLM synthesis:** Pass all parallel outputs to a final LLM that synthesizes a coherent answer. High quality, adds latency and cost.
- **Confidence-weighted:** Use model-reported confidence scores to weight or select outputs.

## Practical Recommendations

Start with waterfall for correctness and simplicity. Move to parallel when you identify independent sub-tasks or latency is a bottleneck. Use frameworks like **LangGraph**, **CrewAI**, or **AutoGen** which natively support both patterns and provide tooling for managing concurrent agent execution, shared state, and error handling.

Always measure: parallel agents cost more per request, and the latency savings only materialize if agents truly run independently on separate compute.
