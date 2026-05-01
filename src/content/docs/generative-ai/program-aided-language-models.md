---
title: Program-Aided Language Models (PAL)
description: Explore Program-Aided Language Models (PAL) — a reasoning technique where LLMs generate executable Python code instead of natural language reasoning chains, offloading computation to a Python interpreter to eliminate arithmetic errors, track state precisely, and solve complex multi-step problems reliably.
---

**Program-Aided Language Models** (Gao et al., Carnegie Mellon University, 2022) address a fundamental limitation of chain-of-thought prompting: language models are unreliable calculators. An LLM reasoning through arithmetic in natural language frequently makes computation errors — carrying the wrong digit, misremembering intermediate values, or applying operations out of order.

PAL's insight is that LLMs should do what they are good at — **understanding problems and writing code** — and delegate what they are bad at — **precise arithmetic and state tracking** — to a Python interpreter. The LLM generates a Python program that, when executed, produces the correct answer.

## Motivation: Why Natural Language Reasoning Fails

Consider a multi-step word problem:

> "There are 15 trees in a grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did the workers plant?"

Chain-of-thought can handle this easily. But consider:

> "Roger has 5 tennis balls. He buys 2 cans of tennis balls. Each can has 3 balls. He gives away a third of his total balls. How many does he have left?"

A CoT solution must track the running count correctly across each step. LLMs frequently make off-by-one errors or mis-apply the division. PAL instead generates:

```python
# Roger's tennis balls
initial_balls = 5
cans_bought = 2
balls_per_can = 3
balls_bought = cans_bought * balls_per_can
total = initial_balls + balls_bought    # 11
given_away = total / 3                  # 3.67 → but we should floor this
remaining = total - int(given_away)     # 8
answer = remaining
```

The interpreter handles the arithmetic exactly; the LLM only needs to translate the problem into correct code structure.

## The PAL Framework

PAL decomposes reasoning into two stages:

1. **Program generation**: The LLM receives the problem and few-shot examples of (problem, program) pairs. It generates a Python program that computes the answer.
2. **Execution**: The generated program is run in a Python interpreter. The output is the final answer.

```python
import openai
import ast
import sys
from io import StringIO
from typing import Optional

client = openai.OpenAI()

# Few-shot examples for arithmetic word problems
ARITHMETIC_EXAMPLES = '''
Q: There are 23 students in a class. The teacher adds 5 new students. 
Then 3 students leave. How many students are in the class?

# Solution
initial = 23
added = 5
left = 3
total = initial + added - left
answer = total

Q: A factory produces 120 widgets per hour. It operates for 8 hours a day,
5 days a week. How many widgets does it produce in a week?

# Solution
per_hour = 120
hours_per_day = 8
days_per_week = 5
answer = per_hour * hours_per_day * days_per_week

Q: {question}

# Solution
'''

def generate_program(question: str, examples: str = ARITHMETIC_EXAMPLES) -> str:
    """
    Use the LLM to generate a Python program that solves the given question.
    """
    prompt = examples.format(question=question)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stop=["Q:"]   # Stop before generating the next question
    )
    return response.choices[0].message.content.strip()


def execute_program(program: str, timeout: int = 10) -> Optional[str]:
    """
    Execute the generated Python program in a sandboxed environment
    and return the value of the `answer` variable.

    Security note: In production, use a proper sandbox (e.g., subprocess with
    resource limits, RestrictedPython, or a containerized execution environment).
    """
    # Capture stdout and get the `answer` variable
    local_vars = {}
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Validate it's syntactically valid Python before executing
        ast.parse(program)
        exec(program, {"__builtins__": {}}, local_vars)   # restrict builtins
        return str(local_vars.get("answer", "No answer variable found"))
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    except Exception as e:
        return f"RuntimeError: {e}"
    finally:
        sys.stdout = old_stdout


def pal_solve(question: str) -> dict:
    """
    Full PAL pipeline: generate program → execute → return answer.
    """
    program = generate_program(question)
    answer = execute_program(program)
    return {"question": question, "program": program, "answer": answer}
```

## Multi-Step Reasoning with Symbolic State

PAL's power scales with problem complexity. For problems that require tracking multiple entities over time — a key failure mode of chain-of-thought — the generated code acts as explicit state:

```python
# Few-shot example for symbolic/multi-entity reasoning
SYMBOLIC_EXAMPLE = '''
Q: Alice has twice as many apples as Bob. Bob has 5 more apples than Carol.
Carol has 10 apples. How many apples does Alice have?

# Solution
carol = 10
bob = carol + 5           # Bob has 5 more than Carol
alice = 2 * bob           # Alice has twice as many as Bob
answer = alice

Q: A train leaves City A at 9am traveling at 60 mph toward City B.
Another train leaves City B at 10am traveling at 80 mph toward City A.
The cities are 300 miles apart. At what hour do the trains meet?

# Solution
distance = 300            # miles
speed_a = 60              # mph, Train A (leaves at 9am)
speed_b = 80              # mph, Train B (leaves at 10am)

# After Train B departs (t hours after 10am):
# Train A has already traveled 60 miles
# Remaining distance = 300 - 60 = 240 miles
# Closing speed = 60 + 80 = 140 mph
distance_covered_by_a_before_b_leaves = speed_a * 1  # 1 hour head start
remaining = distance - distance_covered_by_a_before_b_leaves
closing_speed = speed_a + speed_b
hours_after_10am = remaining / closing_speed
meeting_hour = 10 + hours_after_10am
answer = round(meeting_hour, 2)
'''
```

## PAL for Date and Calendar Reasoning

Calendar arithmetic is notoriously error-prone for LLMs. PAL leverages Python's `datetime` module:

```python
DATETIME_EXAMPLE = '''
import datetime

Q: Jane's birthday is March 15. She wants to throw a party exactly
45 days before her birthday in 2025. What date should the party be?

# Solution
import datetime
birthday = datetime.date(2025, 3, 15)
days_before = 45
party_date = birthday - datetime.timedelta(days=days_before)
answer = party_date.strftime("%B %d, %Y")

Q: {question}

# Solution
import datetime
'''

def pal_solve_datetime(question: str) -> dict:
    """PAL with datetime imports pre-injected."""
    program = generate_program(question, examples=DATETIME_EXAMPLE)
    # Allow datetime import in sandboxed execution
    local_vars = {}
    try:
        import datetime as dt
        exec(program, {"datetime": dt, "__builtins__": {}}, local_vars)
        return {"answer": str(local_vars.get("answer", ""))}
    except Exception as e:
        return {"answer": f"Error: {e}"}
```

## PAL vs. Chain-of-Thought: When to Use Each

| Characteristic | Chain-of-Thought | PAL |
|---|---|---|
| Arithmetic accuracy | Prone to errors | Exact (interpreter) |
| State tracking | Error-prone at depth | Explicit in code |
| Abstract reasoning | Strong | Strong |
| Common-sense Q&A | Strong | Neutral |
| Setup complexity | None | Requires execution env |
| Latency | Lower | Slightly higher |
| Explainability | Natural language | Code + result |

PAL outperforms CoT on **GSM8K** (grade school math: +15–20%), **MathQA** (algebra/geometry: +10%), and **SVAMP** (math word problems), while remaining comparable on tasks requiring abstract or commonsense reasoning where code generation provides no advantage.

## Self-Debugging Generated Programs

A natural extension is to feed execution errors back to the LLM for self-correction:

```python
def pal_with_self_debug(question: str, max_retries: int = 3) -> dict:
    """
    Generate program → execute → if error, show error to LLM → retry.
    """
    program = generate_program(question)
    
    for attempt in range(max_retries):
        result = execute_program(program)
        
        if not result.startswith(("SyntaxError", "RuntimeError")):
            return {"answer": result, "attempts": attempt + 1, "program": program}
        
        # Self-debug: show error and ask for a fix
        fix_prompt = f"""The following Python program has an error:

```python
{program}
```

Error: {result}

Fix the program to correctly solve: {question}
Output only the corrected Python code."""

        program = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0
        ).choices[0].message.content.strip()
        # Strip markdown fences if present
        if program.startswith("```python"):
            program = program[9:]
        if program.endswith("```"):
            program = program[:-3]
        program = program.strip()
    
    return {"answer": "Failed after retries", "attempts": max_retries, "program": program}

```

## Security Considerations

Executing LLM-generated code requires a careful security posture. In production:

- **Sandbox execution**: Use subprocess with strict resource limits (CPU time, memory, no network, no filesystem access).
- **Allow-list builtins**: Restrict `exec()` builtins to mathematical operations and approved modules.
- **Timeout enforcement**: Kill runaway programs (infinite loops, heavy computation).
- **Output validation**: Verify the result is the expected type before returning to the user.

PAL represents a broader paradigm — **tool-augmented reasoning** — where LLMs act as orchestrators that delegate specialized subtasks to reliable external systems. This same principle underlies function calling, code interpreters in ChatGPT, and more general agentic tool-use architectures.
