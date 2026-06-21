---
title: Graph-of-Thoughts (GoT) Prompting
description: Explore Graph-of-Thoughts (GoT), a prompting framework that structures LLM reasoning as a directed acyclic graph, allowing merging, splitting, and feedback loops.
---

Reasoning frameworks for Large Language Models have progressed from linear progression models to hierarchical search architectures. 
- **Chain-of-Thought (CoT)** structures reasoning as a linear sequence of thoughts.
- **Tree-of-Thoughts (ToT)** allows the model to explore multiple reasoning branches and backtrack when a branch fails.

**Graph-of-Thoughts (GoT)** is a generalization that models LLM reasoning as a **Directed Acyclic Graph (DAG)**. Under GoT, individual thoughts are nodes, and dependencies are directed edges. This allows thoughts to be merged (e.g., combining insights from three different reasoning paths), split into parallel sub-tasks, or cycled through feedback loops, matching human brainstorming processes.

---

## The Limitations of Chain and Tree Architectures

1. **Linear Constraints (CoT):** CoT cannot backtrack or recover if it makes a logical mistake early in the chain.
2. **Strict Hierarchy (ToT):** ToT can branch out and search alternative paths, but it cannot combine findings from different branches. If path A finds fact $X$ and path B finds fact $Y$, a ToT structure cannot easily merge them without creating a new parent node and re-evaluating.

GoT solves this by allowing arbitrary connections between thought steps.

---

## The Components of Graph-of-Thoughts

A GoT framework consists of four primary modules:

```
                  +--------------------------------+
                  |    Controller (Graph State)    |
                  +---------------+----------------+
                                  |
            +---------------------+---------------------+
            |                     |                     |
            v                     v                     v
    [Thought Generator]   [Thought Evaluator]   [Graph Parser/Mutator]
            |                     |                     |
            +---------------------+---------------------+
                                  |
                                  v
                       Output (Optimal Path)
```

1. **Thought State (Graph):** A DAG $G = (V, E)$ where each node $v \in V$ is an LLM-generated thought (e.g., a candidate proof step, a draft paragraph, or an intermediate calculation), and each edge $(u, v) \in E$ indicates that thought $v$ was derived from thought $u$.
2. **Thought Generator:** Instructs the LLM to generate new thought nodes based on current graph states. Operations include:
   - **Branching (Splitting):** Generating multiple diverse thoughts from a single node.
   - **Merging (Aggregation):** Synthesizing information from multiple nodes into a single consolidated thought.
3. **Thought Evaluator:** An LLM or code validator that assigns a score to nodes, identifying which paths are promising and which should be abandoned.
4. **Graph Parser/Mutator:** Decides which operations to run next (e.g., pruning low-scoring nodes or triggering a merge operation) based on evaluation scores.

---

## Unique Graph Operations in GoT

- **Aggregation (Merging):** Useful for writing tasks or multi-document retrieval. The model drafts sections in parallel (nodes $A, B, C$) and then runs a merge prompt to compile them into a unified summary (node $D$).
- **Refinement (Feedback Loops):** A node is passed back through an evaluation/critique loop. The model modifies the thought step iteratively until it passes verification criteria.
- **Backpropagation of Value:** If a downstream node is evaluated as highly successful, its value cascades back to increase the scores of the ancestor nodes that generated it, directing the search algorithm.

---

## GoT vs. ToT vs. CoT

| Feature | Chain-of-Thought (CoT) | Tree-of-Thoughts (ToT) | Graph-of-Thoughts (GoT) |
|---|---|---|---|
| **Structure** | Linear Path | Tree Hierarchy | Directed Acyclic Graph (DAG) |
| **Backtracking** | No | Yes (DFS / BFS) | Yes (Any-path traversal) |
| **Merging Paths** | No | No | Yes (Multi-parent nodes) |
| **Parallel Tasks** | No | Yes (Independent branches) | Yes (Cooperating nodes) |
| **Token Efficiency**| High | Low | Medium-Low (highly targeted) |

---

## Implementing Graph-of-Thoughts: Python Concept

Below is a conceptual system implementation showing how to orchestrate a merge operation on intermediate thought steps.

```python
class ThoughtNode:
    def __init__(self, thought_id, content, parent_ids=None):
        self.id = thought_id
        self.content = content
        self.parents = parent_ids or []
        self.score = 0.0

class GoTRunner:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.nodes = {}

    def generate_thoughts_parallel(self, base_node_id, num_branches=3):
        # Generate diverse ideas from a base thought
        base_content = self.nodes[base_node_id].content
        prompt = f"Given this thought: '{base_content}', generate {num_branches} distinct next steps."
        responses = self.llm.call(prompt) # Returns a list of strings
        
        new_ids = []
        for resp in responses:
            node_id = f"node_{len(self.nodes)}"
            self.nodes[node_id] = ThoughtNode(node_id, resp, parent_ids=[base_node_id])
            new_ids.append(node_id)
        return new_ids

    def evaluate_node(self, node_id):
        # Grade the thought quality
        node = self.nodes[node_id]
        prompt = f"Evaluate the validity of this thought: '{node.content}'. Rate 0.0 to 1.0."
        score = float(self.llm.call(prompt))
        node.score = score
        return score

    def merge_thoughts(self, node_ids):
        # Consolidate thoughts from multiple branches
        contents = [self.nodes[nid].content for nid in node_ids]
        prompt = f"Combine the following ideas into a single, cohesive next step:\n" + "\n".join(contents)
        merged_content = self.llm.call(prompt)
        
        merged_id = f"node_{len(self.nodes)}"
        self.nodes[merged_id] = ThoughtNode(merged_id, merged_content, parent_ids=node_ids)
        return merged_id
```
