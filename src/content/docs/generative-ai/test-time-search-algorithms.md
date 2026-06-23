---
title: "Test-Time Search: Best-of-N & MCTS in Reasoning Models"
description: Explore test-time search algorithms like Best-of-N sampling and Monte Carlo Tree Search (MCTS) that scale model reasoning during inference.
---

Traditionally, scaling Large Language Models meant scaling parameter size during training. However, reasoning models (such as OpenAI's o1 or DeepSeek-R1) shift this paradigm by scaling compute during **inference (test-time)**. 

By allowing models to search, evaluate, and refine their reasoning steps before returning a final answer, test-time search algorithms scale accuracy curves. Two foundational search algorithms used in this paradigm are **Best-of-N Sampling** and **Monte Carlo Tree Search (MCTS)**.

---

## Scaling Inference Compute

When a standard LLM answers a query, it executes a single, greedy forward pass, spending the same amount of compute on "2 + 2" as it does on a complex math olympiad problem.

Test-time search algorithms allocate compute dynamically:
- **Generation:** Generate multiple diverse candidate reasoning chains.
- **Evaluation:** Grade intermediate steps using Process Reward Models (PRMs) or code executions.
- **Selection:** Search the tree of generated paths to select the most logically sound answer.

---

## Best-of-N Sampling (Rejection Sampling)

Best-of-N sampling is the simplest form of test-time compute scaling:
1. **Parallel Sampling:** The system samples $N$ independent completions (reasoning paths) from the model at a higher temperature (e.g., $T=0.7$ to $T=1.0$ to encourage diversity).
2. **Scoring:** A verifier—either an Outcome Reward Model (ORM) grading the final answer or a Process Reward Model (PRM) grading each step—assigns a score to each of the $N$ completions.
3. **Selection:** The completion with the highest cumulative score is returned.

$$\text{Best-of-N}(x) = \arg\max_{y_i \in \{y_1, \dots, y_N\}} \text{Score}(x, y_i)$$

### The Trade-off
Best-of-N is easy to parallelize, but it is highly inefficient for complex, multi-step problems. If a reasoning path contains 10 steps and the model makes a mistake at step 2, Best-of-N wastes compute generating the remaining 8 steps of that broken path.

---

## Monte Carlo Tree Search (MCTS)

MCTS addresses Best-of-N's inefficiencies by evaluating and branching at the **step level** instead of the full sequence level.

MCTS structures reasoning as a search tree where each node is an intermediate reasoning step. The algorithm executes four repeating phases to build the search tree:

```
  1. Selection             2. Expansion           3. Simulation           4. Backpropagation
    [Node A]                  [Node A]               [Node A]                 [Node A (New Score)]
       |                         |                      |                          ^
    [Node B]                  [Node B]               [Node B]                 [Node B (Updated)]
                                 |                      |                          ^
                            [Node C (New)]  - - - -> [Node D (Rollout)] - - - - - - +
```

### 1. Selection
Starting at the root (the prompt), the algorithm selects the most promising child node. Promising paths are balanced between **exploitation** (nodes with high reward scores) and **exploration** (nodes that have not been visited often), typically using the Upper Confidence bound applied to Trees (UCT) formula:

$$\text{UCT}(v) = \frac{Q(v)}{N(v)} + c \sqrt{\frac{\ln N(v_p)}{N(v)}}$$

Where $Q(v)$ is the accumulated reward, $N(v)$ is the visit count of node $v$, $v_p$ is the parent node, and $c$ is an exploration constant.

### 2. Expansion
Once a leaf node is selected, the LLM is queried to generate multiple candidate next steps, expanding the search tree with new child nodes.

### 3. Simulation (Rollout)
From the newly expanded node, the system performs a quick rollout—sampling tokens until it reaches a terminal state (an answer)—to estimate the node's long-term viability.

### 4. Backpropagation
The reward of the simulation (often computed via a Process Reward Model) is propagated back up the tree, updating the $Q(v)$ and $N(v)$ values of all ancestor nodes.

---

## Comparison: Best-of-N vs. MCTS

| Feature | Best-of-N Sampling | Monte Carlo Tree Search (MCTS) |
|---|---|---|
| **Granularity** | Sequence-level evaluation | Step-level evaluation |
| **Search Tree** | Flat (one level of depth) | Deep, branched hierarchy |
| **Pruning** | None (samples run to completion) | Early pruning of low-scoring branches |
| **Compute Cost** | High (wasted on dead ends) | Moderate (optimizes allocation) |
| **Implementation** | Simple (parallel API requests) | Complex (requires state tracking & tree structure) |

---

## Python Concept: Best-of-N Selector

Below is a Python demonstration of how Best-of-N sampling leverages a Process Reward Model to rank multiple candidate reasoning chains.

```python
import numpy as np

def run_best_of_n(prompt, generator_llm, prm_model, n=10):
    candidates = []
    scores = []
    
    # 1. Sample N completions in parallel
    for _ in range(n):
        # Sample completions at high temperature for diversity
        completion = generator_llm.generate(prompt, temperature=0.8)
        candidates.append(completion)
        
    # 2. Score intermediate steps in each completion
    for candidate in candidates:
        # Split candidate into logical steps (typically divided by double newlines)
        steps = [step for step in candidate.split("\n\n") if step.strip()]
        
        # PRM scores each step (returns values in range [0.0, 1.0])
        step_scores = prm_model.score_steps(prompt, steps)
        
        # Aggregate score (e.g., minimum score or product of step probabilities)
        # Using minimum score is a common heuristic to penalize chains with a single logical flaw
        composite_score = np.min(step_scores)
        scores.append(composite_score)
        
    # 3. Select the candidate with the highest reward
    best_idx = np.argmax(scores)
    return candidates[best_idx], scores[best_idx]
```
