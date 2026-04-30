---
title: Tree of Thoughts Prompting
description: Learn Tree of Thoughts (ToT) — a prompting framework that enables LLMs to explore multiple reasoning paths in parallel, evaluate intermediate steps, and backtrack when stuck, enabling systematic problem-solving on tasks that linear chain-of-thought prompting cannot reliably solve.
---

**Tree of Thoughts** (Yao et al., Princeton & Google DeepMind, 2023) is a prompting framework that reframes LLM inference as a deliberate search over a tree of reasoning steps. Rather than generating a single linear chain of thought from start to finish, ToT generates multiple candidate "thoughts" at each step, evaluates their promise, and uses search algorithms (breadth-first, depth-first, or beam search) to navigate toward a solution.

The core insight is that many problems humans find difficult require **exploration, planning, and backtracking** — capabilities that linear autoregressive generation lacks by design. Chain-of-thought prompting produces one path through the reasoning space; Tree of Thoughts explores many.

## Motivation: When Chain-of-Thought Fails

Chain-of-thought (CoT) prompting substantially improves LLM performance on multi-step reasoning tasks. But it has a fundamental limitation: **if the model commits to an incorrect reasoning step early, it typically persists with that error**. The model cannot backtrack, evaluate whether its current path is viable, or consider alternatives.

Consider the **Game of 24** (use four numbers and arithmetic operations to equal 24). Standard CoT with GPT-4 solves only ~4% of problems. Tree of Thoughts achieves ~74% — because it can try multiple candidate decompositions, evaluate which are feasible, and abandon dead ends.

Similar patterns appear in:

- **Crossword puzzle solving**: Partial fills can conflict; backtracking is essential.
- **Creative writing with constraints**: Multiple structural approaches need exploration.
- **Multi-step planning**: Early decisions constrain later options in non-obvious ways.

## Framework Components

ToT decomposes the problem into four interacting components:

### 1. Thought Decomposition

Define the unit of "thought" appropriate for the task:

- A single reasoning step (e.g., one arithmetic operation).
- A paragraph of planning.
- A candidate answer to a sub-question.
- A word (for crosswords) or sentence (for writing).

The granularity determines the tree's branching factor and depth.

### 2. Thought Generation

At each node, generate $k$ candidate next thoughts using the LLM:

```python
import openai
from typing import Any

client = openai.OpenAI()

def generate_thoughts(problem: str, current_state: str, k: int = 5,
                       task_description: str = "") -> list[str]:
    """
    Generate k candidate next thoughts from the current state.
    Uses a single prompt with temperature > 0 for diversity,
    or multiple independent samples.
    """
    prompt = f"""{task_description}

Problem: {problem}

Current progress:
{current_state}

Generate {k} different possible next steps or approaches. 
Each should be meaningfully different from the others.
List them numbered 1 through {k}."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        n=1  # Generate all k in one call
    )
    
    raw = response.choices[0].message.content
    # Parse numbered list — robust splitting
    thoughts = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:
            thoughts.append(line.split(".", 1)[1].strip())
    return thoughts[:k]
```

### 3. State Evaluation

A critical ToT component absent from CoT: a **value function** that scores intermediate states without reaching the final answer:

```python
def evaluate_state(problem: str, state: str, 
                   evaluation_criteria: str = "") -> float:
    """
    Ask the LLM to evaluate the promise of the current reasoning state.
    Returns a score in [0, 1].
    
    Three evaluation strategies:
    - "value": single scalar score
    - "vote": majority vote across multiple LLM samples  
    - "binary": sure/likely/impossible classification
    """
    prompt = f"""{evaluation_criteria}

Problem: {problem}

Current state:
{state}

Evaluate whether this reasoning path is likely to lead to a correct solution.
Rate on a scale:
- 1 (sure): This is definitely on the right track
- 0.5 (likely): This might work but has some issues
- 0 (impossible): This path cannot lead to a correct solution

Output only the number (0, 0.5, or 1) followed by a brief reason."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    ).choices[0].message.content.strip()
    
    for val in ["1", "0.5", "0"]:
        if response.startswith(val):
            return float(val)
    return 0.5  # default if unparseable


def vote_evaluate_state(problem: str, state: str, 
                        n_votes: int = 5) -> float:
    """
    Vote-based evaluation: ask the LLM n times whether
    this state is the best among alternatives. Returns fraction of votes.
    """
    votes = []
    for _ in range(n_votes):
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # use cheaper model for voting
            messages=[{
                "role": "user",
                "content": f"Problem: {problem}\nState: {state}\n"
                           f"Is this reasoning likely to succeed? Answer YES or NO."
            }],
            temperature=0.7
        ).choices[0].message.content.strip().upper()
        votes.append(1.0 if "YES" in response else 0.0)
    return sum(votes) / len(votes)
```

### 4. Search Algorithm

With thought generation and evaluation in place, apply a search strategy:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ThoughtNode:
    state: str
    parent: Optional["ThoughtNode"] = None
    value: float = 0.0
    depth: int = 0
    children: list["ThoughtNode"] = field(default_factory=list)
    
    def path(self) -> list[str]:
        """Return full reasoning path from root to this node."""
        nodes = []
        node = self
        while node is not None:
            nodes.append(node.state)
            node = node.parent
        return list(reversed(nodes))


def breadth_first_search_tot(
    problem: str,
    task_description: str,
    evaluation_criteria: str,
    b: int = 5,       # beam width (number of states to keep per level)
    k: int = 3,       # thoughts to generate per state
    max_depth: int = 4,
    is_terminal: callable = None
) -> Optional[ThoughtNode]:
    """
    BFS/Beam search variant of Tree of Thoughts.
    
    At each level, keep the b highest-value states
    and expand each with k new thoughts.
    """
    # Initialize with empty state
    frontier = [ThoughtNode(state="(start)", depth=0, value=1.0)]
    
    for depth in range(1, max_depth + 1):
        print(f"\nDepth {depth}: expanding {len(frontier)} states...")
        candidates = []
        
        for node in frontier:
            # Generate k candidate next thoughts
            thoughts = generate_thoughts(
                problem, node.state, k=k,
                task_description=task_description
            )
            
            for thought in thoughts:
                new_state = node.state + f"\nStep {depth}: {thought}"
                child = ThoughtNode(
                    state=new_state,
                    parent=node,
                    depth=depth
                )
                child.value = evaluate_state(
                    problem, new_state, evaluation_criteria
                )
                node.children.append(child)
                candidates.append(child)
                
                # Check if this is a terminal solution
                if is_terminal and is_terminal(problem, new_state):
                    print(f"Solution found at depth {depth}!")
                    return child
        
        # Prune to beam width b
        candidates.sort(key=lambda n: n.value, reverse=True)
        frontier = candidates[:b]
        
        # If all states are impossible, stop early
        if all(n.value == 0 for n in frontier):
            print("All paths exhausted — no solution found.")
            return None
    
    # Return best final state
    return max(frontier, key=lambda n: n.value) if frontier else None
```

## Complete Example: Game of 24

```python
def solve_game_of_24(numbers: list[int]) -> Optional[str]:
    """
    Use ToT to solve the Game of 24:
    Use all four numbers with +, -, *, / to reach 24.
    """
    problem = f"Use the numbers {numbers} with +, -, *, / (each exactly once) to equal 24."
    
    task_description = """You are solving the Game of 24.
At each step, pick two numbers from the remaining list, apply an operation,
and replace them with the result. Show the operation and new remaining numbers."""
    
    evaluation_criteria = """Evaluate whether the remaining numbers and operations
can still reach 24. If the current total is 24 and no numbers remain, it's solved.
If no valid combination remains, mark impossible."""
    
    def is_terminal(problem: str, state: str) -> bool:
        # Simple check: does the state claim to reach 24?
        return "= 24" in state and "solution" in state.lower()
    
    result = breadth_first_search_tot(
        problem=problem,
        task_description=task_description,
        evaluation_criteria=evaluation_criteria,
        b=5, k=3, max_depth=6,
        is_terminal=is_terminal
    )
    
    if result:
        return "\n".join(result.path())
    return "No solution found"
```

## ToT vs. Related Approaches

| Method | Path structure | Evaluation | Backtracking | Best for |
|---|---|---|---|---|
| **Chain-of-Thought** | Single linear | None | No | Standard reasoning |
| **Self-Consistency** | Multiple (parallel) | Majority vote (final) | No | Arithmetic, factual |
| **Tree of Thoughts** | Tree (sequential + branching) | Per-step value | Yes | Planning, search |
| **Reasoning as Planning (RAP)** | Tree + MCTS | LLM world model | Yes | Long-horizon planning |
| **Graph of Thoughts** | DAG (thoughts can merge) | Per-node | Yes | Aggregation tasks |

## Practical Considerations

**Cost**: ToT requires many more LLM calls than CoT — $O(b \times k \times d)$ calls for beam search with beam width $b$, branching factor $k$, and depth $d$. For $b=5, k=3, d=4$, this is 60+ calls versus 1 for CoT.

**Task fit**: ToT provides the most value on tasks where:

- The search space is large (many possible approaches).
- Intermediate states are evaluable (the LLM can judge partial progress).
- Backtracking is necessary (early mistakes derail linear reasoning).

For tasks where chain-of-thought already achieves high accuracy, or where evaluation of intermediate states is itself unreliable, the cost overhead is rarely justified.

**Cheaper evaluation**: Using a small model (GPT-4o-mini) for evaluation while reserving a larger model (GPT-4o) for thought generation significantly reduces cost without major quality loss.

Tree of Thoughts represents a foundational shift in how we think about LLM inference — from single-pass generation to deliberate, evaluative search — and serves as the conceptual foundation for more sophisticated reasoning-time compute approaches like MCTS-guided inference and process reward models.
