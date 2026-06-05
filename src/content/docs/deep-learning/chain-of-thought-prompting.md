---
title: "Chain-of-Thought Prompting"
description: "Understand chain-of-thought prompting — a technique that elicits reasoning in large language models by generating intermediate steps."
date: "2026-03-20"
tags: ["deep-learning", "llms", "prompting", "reasoning"]
---

Chain-of-thought (CoT) prompting is a prompting technique that improves reasoning in large language models by encouraging the model to generate intermediate reasoning steps rather than answering directly. This simple change dramatically improves performance on complex reasoning tasks.

## How Chain-of-Thought Works

Instead of asking a question and expecting a direct answer:

```
Q: What is 15% of 50?
A: 7.5
```

Chain-of-thought prompting asks for intermediate steps:

```
Q: What is 15% of 50?
A: First, 10% of 50 is 5. Then 5% is half of that, which is 2.5. So 15% is 5 + 2.5 = 7.5.
```

The key insight is that explicitly generating reasoning steps helps the model:
- Break down complex problems into manageable subproblems
- Allocate computation to the parts that matter most
- Enable self-correction when intermediate steps go wrong

## Basic Chain-of-Thought Prompt

```python
cot_prompt = """Solve the following problem step by step.

Q: If there are 5 boxes and each box contains 3 red balls and 4 blue balls, how many total balls are there?

A: First, calculate balls per box: 3 red + 4 blue = 7 balls per box.
   Then multiply by number of boxes: 5 boxes × 7 balls = 35 balls.
   Answer: 35

Q: A train travels 60 miles in 1.5 hours. What is its average speed in mph?

A: Speed = Distance / Time
   60 miles / 1.5 hours = 40 mph
   Answer: 40 mph

Q: {question}
A: """
```

## Zero-Shot Chain-of-Thought

A simpler variant that just asks the model to "think step by step":

```python
zero_shot_cot = """Question: {question}

Let's think step by step.
"""
```

This works surprisingly well without requiring few-shot examples.

## Self-Consistency with Chain-of-Thought

Rather than taking a single CoT generation, sample multiple reasoning paths and take the majority answer:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def self_consistency(model, tokenizer, question, num_samples=5):
    """Sample multiple CoT paths and return majority answer."""
    answers = []
    
    for _ in range(num_samples):
        # Generate with temperature > 0 for diversity
        inputs = tokenizer.encode(question, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers.append(extract_answer(response))
    
    # Return majority vote
    return Counter(answers).most_common(1)[0][0]
```

## Benefits and Limitations

**Benefits:**
- Dramatically improves reasoning on arithmetic, commonsense, and symbolic tasks
- Makes model behavior more interpretable
- Enables detection of reasoning errors

**Limitations:**
- Works best with larger models (typically 50B+ parameters)
- Can encourageverbose or circular reasoning on simple tasks
- Sensitive to prompt wording and few-shot examples

Chain-of-thought prompting represents a fundamental shift in how we interact with LLMs — treating them not just as answer generators but as systems that can reason when given the right scaffolding.