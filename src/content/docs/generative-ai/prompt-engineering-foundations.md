---
title: "Prompt Engineering Foundations"
description: "Master the fundamentals of prompt engineering — from basic techniques like zero-shot and few-shot prompting to advanced strategies for eliciting the best LLM responses."
---

Prompt engineering is the craft of crafting inputs that guide LLMs to produce desired outputs. It's become one of the most valuable skills for working with language models.

## The Anatomy of a Prompt

A well-structured prompt typically contains:

```python
prompt = """
[System Prompt - Defines behavior]
You are an expert Python developer who writes clean, well-documented code.

[Context - Background information]
The codebase uses FastAPI for the web framework and SQLAlchemy for database operations.

[Task - What to do]
Write a function to create a new user in the database.

[Constraints - Rules to follow]
- Use async/await syntax
- Include proper error handling
- Validate email format
- Return the created user object

[Format - Output structure]
Return the function as a standalone code block.
"""

def construct_prompt(system, context, task, constraints, format_spec):
    return f"{system}\n\n{context}\n\n{task}\n\n{constraints}\n\n{format_spec}"
```

## Zero-Shot Prompting

The simplest approach: ask directly without examples.

```python
# Zero-shot
prompt = "Translate 'Hello, how are you?' to Spanish."
```

When zero-shot works well:
- The task is well-defined and common.
- The model has seen similar tasks during training.
- The instruction is clear and specific.

### Improving Zero-Shot Performance

```python
# Add specificity
weak = "Explain quantum computing."
strong = """Explain quantum computing to a computer science undergraduate.
Cover: qubits, superposition, entanglement, and practical applications.
Use analogies to complex numbers where helpful. Limit to 300 words."""

# Add step-by-step instruction
better = """Explain quantum computing step by step:
1. Start with classical computing limitations
2. Introduce the qubit
3. Explain superposition
4. Explain entanglement
5. Describe one practical application

Use simple language."""
```

## Few-Shot Prompting

Show examples of the desired behavior:

```python
few_shot_prompt = """
Convert these sentences to passive voice:

Active: The cat chased the mouse.
Passive: The mouse was chased by the cat.

Active: Scientists discovered a new species.
Passive: A new species was discovered by scientists.

Active: The chef cooked a delicious meal.
Passive:
"""
```

### Designing Effective Examples

```python
def create_few_shot_examples(task_examples):
    """
    Create a few-shot prompt with diverse, accurate examples.
    
    Best practices:
    1. Use 3-5 examples (not too few, not too many)
    2. Cover diverse cases
    3. Ensure examples are correct
    4. Maintain consistent format
    """
    prompt = "Examples for the task:\n\n"
    
    for i, (input_val, output_val) in enumerate(task_examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Input: {input_val}\n"
        prompt += f"Output: {output_val}\n\n"
    
    return prompt
```

## Instruction Engineering

### Clear and Specific Instructions

```python
# Vague
bad = "Write something about climate change."

# Clear and specific
good = """Write a 3-paragraph summary of climate change causes and effects.
Use a formal, academic tone suitable for a general science audience.
Focus on: greenhouse gases, temperature rise, and ocean acidification.
Cite specific statistics where relevant."""
```

### Formatting Instructions

```python
# Output format specification
format_prompt = """
Analyze the following product review and extract information in JSON format:

{
    "sentiment": "positive|negative|neutral",
    "aspects": ["list", "of", "aspects", "mentioned"],
    "key_points": ["main", "points", "from", "review"],
    "recommendation": "would_recommend boolean",
    "confidence": 0.0_to_1.0
}

Review: {review_text}
"""
```

## Prompt Templates and Variables

```python
from string import Template

class PromptTemplate:
    def __init__(self, template):
        self.template = template
    
    def substitute(self, **kwargs):
        return self.template.substitute(kwargs)
    
    def safe_substitute(self, **kwargs):
        return self.template.safe_substitute(kwargs)

# Example usage
template = PromptTemplate(
    """Analyze the following $domain document:
    
Document: $text
    
Extract:
- Main topic
- Key entities
- Summary (2-3 sentences)
    
Output format: JSON"""
)

prompt = template.substitute(
    domain="legal",
    text=contract_text
)
```

## Chain-of-Thought Prompting

Encourage explicit reasoning:

```python
cot_prompt = """Solve this math problem step by step.
Show your work for each step.

Problem: If a train travels at 60 mph for 3 hours, how far does it go?

Step 1: Identify the formula. Distance = Speed × Time.
Step 2: Plug in the values. Distance = 60 × 3.
Step 3: Calculate. 60 × 3 = 180.
Answer: 180 miles.

---

Problem: A bookstore has 500 books. They sell 35 books each day.
How many days until they have 100 books left?

Step 1: Calculate books to sell. 500 - 100 = 400 books.
Step 2: Use the formula. Days = books_to_sell / books_per_day.
Step 3: Calculate. 400 / 35.
Step 4: Round up (can't sell partial days). 12 days (with 5 books remaining).
Answer: 12 days.

---

Problem: $problem
Step 1:
"""
```

## Self-Consistency

Generate multiple responses and take the best:

```python
def self_consistent_prompt(question, model, num_samples=5):
    """Generate multiple answers and return most common."""
    answers = []
    
    for _ in range(num_samples):
        response = model.generate(
            f"Think step by step and provide the final answer.\n\n{question}"
        )
        answer = extract_final_answer(response)
        answers.append(answer)
    
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    
    return most_common, answers
```

## Prompt Optimization Strategies

### Iterative Refinement

```python
def refine_prompt(original_prompt, feedback):
    """Iteratively improve a prompt based on output issues."""
    current = original_prompt
    
    for issue in feedback:
        if "unclear" in issue.lower():
            current = add_clarification(current, issue)
        if "too verbose" in issue.lower():
            current = simplify_language(current)
        if "incorrect format" in issue.lower():
            current = tighten_format(current)
        if "missed details" in issue.lower():
            current = add_constraints(current, issue)
    
    return current

# Example refinement
def add_clarification(prompt, issue):
    """Add specific clarifications based on feedback."""
    clarification = f"\n\nClarification based on feedback: {issue}\n"
    return prompt + clarification
```

### Testing Prompts

```python
def evaluate_prompt(prompt, test_cases, model):
    """Evaluate a prompt against a set of test cases."""
    results = []
    
    for test in test_cases:
        response = model.generate(prompt.replace("{input}", test["input"]))
        
        result = {
            "input": test["input"],
            "expected": test["expected"],
            "actual": response,
            "correct": evaluate_response(response, test["expected"]),
            "issues": identify_issues(response, test["expected"])
        }
        results.append(result)
    
    return aggregate_results(results)
```

## Common Prompting Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| Ambiguous instructions | Unpredictable output | Be specific and explicit |
| Too many tasks | Confused output | One task per prompt |
| Missing context | Irrelevant responses | Provide relevant background |
| Inconsistent format | Parse errors | Define format explicitly |
| Over-constraining | Missing valid answers | Allow alternatives |

## Advanced Techniques

### Persona Prompting

```python
persona = """You are Dr. Sarah Chen, a renowned astrophysicist at MIT.
You have a talent for making complex topics accessible.

Your speaking style:
- Uses concrete analogies
- Asks rhetorical questions
- Shows genuine enthusiasm for the subject

Now, explain dark matter to a curious 10-year-old."""
```

### Constitutional AI Prompting

```python
constitutional_prompt = """You are a helpful AI assistant.

Before responding, check that your answer:
1. Is factually accurate
2. Does not contain harmful content
3. Acknowledges uncertainty when appropriate
4. Is concise while being complete

User query: {query}

Your response:
"""
```

### Negative Prompting

```python
negative_prompt = """Generate a summary of the article.

Do NOT:
- Include personal opinions
- Mention specific statistics unless directly asked
- Use filler phrases like "In conclusion"
- Go over 200 words

Article: {article}

Summary:
"""
```

Effective prompt engineering is part science, part art. The key is to be clear about what you want, provide appropriate context, and iterate based on results. As models improve, prompt engineering skills remain essential for getting the best performance.