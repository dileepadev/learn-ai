---
title: "Chain-of-Thought Prompting: Techniques and Best Practices"
description: "Master chain-of-thought prompting to improve LLM reasoning — from basic CoT to self-consistency, tree of thoughts, and advanced prompting strategies for complex tasks."
---

Chain-of-thought (CoT) prompting dramatically improves LLM performance on reasoning tasks by encouraging explicit step-by-step thinking. This guide covers the techniques, variations, and best practices for getting the most out of CoT prompting.

## What Is Chain-of-Thought Prompting?

Chain-of-thought prompting asks the model to generate a sequence of reasoning steps before producing the final answer:

```python
# Without CoT
prompt = "What is 15% of 80?"
# → "12"

# With CoT
prompt = """What is 15% of 80? Let's think step by step.
First, 10% of 80 is 8.
Then, 5% of 80 is half of that, which is 4.
So 15% is 8 + 4 = 12."""
# → "12"
```

The difference is not just verbosity — CoT changes *how* the model reasons, leading to more accurate results on complex problems.

## Why CoT Works

CoT improves performance for several reasons:

1. **Attention allocation**: Step-by-step reasoning forces the model to process intermediate states.
2. **Error detection**: Each step can be checked, reducing propagation of early mistakes.
3. **Intermediate states**: The model can use partial results in subsequent steps.
4. **Working memory**: Each step serves as a reminder of what has been done.

## Basic CoT Prompting

### Zero-Shot CoT

Add a simple instruction to trigger reasoning:

```python
zero_shot_cot_prompt = """Solve the following problem. Think step by step.

Question: If a train travels 60 miles per hour for 2 hours, how far does it go?
Step by step:
"""
```

### Few-Shot CoT

Show examples of reasoning:

```python
few_shot_prompt = """Here are some examples of solving math problems step by step.

Question: John has 3 apples. He buys 5 more. How many does he have?
Answer: John starts with 3 apples. He buys 5 more. 3 + 5 = 8 apples.

Question: A shirt costs $20. Sales tax is 8%. How much is the tax?
Answer: The tax rate is 8%, which is 0.08. 20 * 0.08 = $1.60 in tax.

Question: {question}
Step by step:
"""
```

## Self-Consistency

Self-consistency improves CoT by generating multiple reasoning paths and taking the majority answer:

```python
def self_consistency(question: str, model, num_samples: int = 5) -> int:
    answers = []
    
    for _ in range(num_samples):
        # Generate reasoning + answer (with temperature for diversity)
        response = model.generate(
            question,
            temperature=0.8,
            stop_at_answer=False
        )
        answer = extract_answer(response)
        answers.append(answer)
    
    # Majority vote
    from collections import Counter
    return Counter(answers).most_common(1)[0][0]
```

Self-consistency typically improves accuracy 5–15% on reasoning benchmarks with 5–10 samples.

## Tree of Thoughts (ToT)

Tree of Thoughts extends CoT by exploring multiple reasoning branches and using a deliberate search process:

```python
class TreeOfThoughts:
    def __init__(self, model, beam_width=3):
        self.model = model
        self.beam_width = beam_width
    
    def solve(self, problem: str, max_depth: int = 5) -> str:
        # Root: initial problem
        tree = TreeNode(problem)
        
        for depth in range(max_depth):
            # Generate next steps for each node
            for node in tree.bfs_traversal():
                if not node.is_terminal():
                    children = self.model.generate_next_steps(
                        node.state,
                        num=self.beam_width
                    )
                    node.add_children(children)
            
            # Prune to top beams
            tree.prune(self.beam_width)
        
        return tree.get_best_solution()
```

### ToT Search Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Breadth-first** | Explore all nodes at each level | Tasks with many valid paths |
| **Depth-first** | Follow promising paths deep | Tasks where solutions require depth |
| **Beam search** | Keep top-k partial solutions | Balance between exploration/exploitation |
| **Monte Carlo** | Random exploration with evaluation | Complex search spaces |

## Graph of Thoughts (GoT)

Graph of Thoughts allows reasoning steps to merge and branch:

```python
# A reasoning graph where steps can converge
thoughts = {
    "thought_1": {"content": "Consider X", "depends_on": []},
    "thought_2": {"content": "Consider Y", "depends_on": []},
    "thought_3": {"content": "Combine X and Y", "depends_on": ["thought_1", "thought_2"]},
    "thought_4": {"content": "Final synthesis", "depends_on": ["thought_3"]},
}
```

This is useful when multiple independent analyses need to be combined.

## Advanced CoT Techniques

### Decomposition

Break complex problems into subproblems:

```python
decomposition_prompt = """First, break down this problem into subproblems.
Then solve each subproblem step by step.

Problem: {question}

Subproblems:
1.
2.
3.

Now solve each subproblem:
"""
```

### Self-Ask

Have the model ask itself clarifying questions:

```python
self_ask_prompt = """Question: {question}

Are there any clarifications needed?
Q:
A:

Now answer:
"""
```

### Verification

Add explicit verification steps:

```python
verification_prompt = """Solve this problem step by step, then verify your answer.

Problem: {question}

Solution:
[reasoning steps]

Verification:
- Check each step
- Is the final answer consistent with all steps?
- Does the answer make sense?

Final answer:
"""
```

## Handling CoT Failures

### Reasoning Truncation
The model may stop reasoning prematurely:

```python
# Prompt for complete reasoning
completeness_prompt = """Answer thoroughly. Include ALL steps, intermediate calculations, and a final verification.
"""
```

### Irrelevant Reasoning
The model may go off-track:

```python
# Keep reasoning focused
focused_prompt = """Focus only on relevant information.
Ignore distractors and stay on topic.

Question: {question}
Relevant information: [list]
Irrelevant information: [ignore]

Step by step solution:
"""
```

### Calculation Errors
For math problems, the model may make arithmetic mistakes:

```python
# Use external calculation for verification
def solve_with_calc(question: str, model, calculator):
    reasoning = model.generate(f"{question}\nLet's think step by step.")
    # Extract numbers from reasoning
    numbers = extract_numbers(reasoning)
    # Verify calculations
    calculated = calculator.verify(numbers)
    # Return with verification
    return f"{reasoning}\n\nVerification: {calculated}"
```

## CoT for Different Task Types

| Task Type | CoT Strategy | Example |
|-----------|--------------|---------|
| **Math word problems** | Show examples with worked solutions | 15% improvement on GSM8K |
| **Logical reasoning** | Explicit premises and conclusions | Syllogisms, deductive logic |
| **Code generation** | Document algorithm before coding | LeetCode problems |
| **Multi-hop QA** | Break into individual hops | "Who is X's mother?" |
| **Planning** | List steps explicitly | Task decomposition |

## Measuring CoT Effectiveness

Track these metrics:

```python
def evaluate_cot(question: str, expected_answer: str, model):
    results = {
        "direct": model.generate(question),  # No CoT
        "cot": model.generate(f"{question}\nThink step by step."),
        "few_shot_cot": model.generate(
            format_few_shot(question, examples) + "\nAnswer:"
        ),
    }
    
    metrics = {}
    for method, answer in results.items():
        metrics[method] = {
            "correct": answer == expected_answer,
            "steps": count_reasoning_steps(answer),
            "length": len(answer),
        }
    
    return metrics
```

## Common CoT Mistakes to Avoid

1. **Over-explaining**: Too many trivial steps can confuse the model.
2. **Wrong examples**: Poor few-shot examples hurt more than no examples.
3. **Inconsistent format**: Varying the reasoning format breaks the pattern.
4. **Temperature too low**: For self-consistency, use higher temperature.
5. **Ignoring failures**: Track which question types fail with CoT.

Chain-of-thought prompting is one of the most effective ways to improve LLM reasoning. The key is matching the CoT strategy to the task — simple math problems benefit from basic CoT, while complex planning tasks need trees or graphs of thoughts.