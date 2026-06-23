---
title: Self-Consistency in Chain-of-Thought (CoT-SC)
description: Explore Self-Consistency in Chain-of-Thought (CoT-SC), a decoding strategy that samples multiple reasoning paths and selects the most common final answer.
---

Chain-of-Thought (CoT) prompting enables Large Language Models to solve complex multi-step reasoning problems by generating intermediate reasoning steps before arriving at a final answer. However, greedy decoding (always picking the token with the highest probability) is prone to errors: if the model makes a single mistake in the reasoning chain, it will arrive at an incorrect final answer.

**Self-Consistency (CoT-SC)** is a decoding strategy that replaces greedy decoding. Instead of generating a single reasoning path, it samples multiple diverse reasoning paths from the model and aggregates the results, electing the **most common final answer** as the correct output.

---

## The Intuition: Multiple Paths to the Truth

Complex reasoning problems typically have a single correct final answer (e.g., in math or logic), but there are many different ways to write out the steps to solve them. 

Greedy decoding restricts the model to a single path. If that path contains a calculation error, the model fails. Self-Consistency leverages the fact that:
1. Correct reasoning paths tend to converge on the same correct final answer.
2. Incorrect reasoning paths diverge and arrive at many different incorrect answers.

By sampling multiple reasoning paths and finding the consensus final answer, the system filters out random calculation or logical slips.

---

## How Self-Consistency Works

The CoT-SC pipeline consists of three steps:

```
Prompt ---> Sample N Paths (Temperature > 0) 
            |---> Path 1 ---> Answer: 42
            |---> Path 2 ---> Answer: 42
            |---> Path 3 ---> Answer: 15
            |---> Path 4 ---> Answer: 42
            +---> Majority Vote Selector ---> Final Answer: 42
```

1. **Prompting:** The model is prompted using standard Chain-of-Thought prompts (e.g., containing step-by-step examples).
2. **Sampling:** Instead of using greedy decoding (temperature = 0), the system samples a set of $N$ completions (typically $N \in [10, 40]$) using temperature sampling (e.g., $T=0.7$) or nucleus sampling ($p=0.95$). This generates diverse reasoning paths.
3. **Majority Voting:** The system parses the final answer from each generated path and counts the frequencies of each final answer. The answer with the highest frequency is selected as the output:

$$\text{Final Answer} = \arg\max_{a} \sum_{i=1}^{N} \mathbb{I}(\text{Answer}(y_i) == a)$$

Where $y_i$ is the $i$-th generated completion and $\mathbb{I}$ is the indicator function.

---

## Key Benefits of CoT-SC

- **Performance Gain:** Self-Consistency significantly boosts performance on math (GSM8K, MATH) and symbolic reasoning tasks, often outperforming standard CoT by 5% to 15%.
- **Robustness to Prompts:** It is less sensitive to the specific phrasing or formatting of the few-shot exemplars in the prompt compared to greedy CoT.
- **Easy Implementation:** It requires no additional model training, parameter updates, or complex search structures—it operates entirely as a post-generation voting wrapper.

---

## Python Concept: Self-Consistency Runner

Below is a Python demonstration of how to implement the majority voting selector for Self-Consistency.

```python
from collections import Counter
import re

def extract_final_answer(text):
    """
    Helper to extract the final numerical answer from a reasoning chain.
    Typically targets formatting like "The answer is \boxed{X}" or "#### X".
    """
    # Simple regex looking for a number at the end of the text
    matches = re.findall(r"The answer is (\d+)", text)
    if matches:
        return matches[-1]
    
    # Fallback to last number in text
    numbers = re.findall(r"\d+", text)
    if numbers:
        return numbers[-1]
    return None

def run_self_consistency(prompt, generator_llm, num_samples=10):
    completions = []
    
    # 1. Sample N completions at a higher temperature
    for _ in range(num_samples):
        completion = generator_llm.generate(prompt, temperature=0.7)
        completions.append(completion)
        
    # 2. Extract final answers
    answers = []
    for comp in completions:
        ans = extract_final_answer(comp)
        if ans is not None:
            answers.append(ans)
            
    if not answers:
        return "No answer parsed", completions
        
    # 3. Perform majority vote
    vote_counts = Counter(answers)
    most_common_answer, count = vote_counts.most_common(1)[0]
    
    print(f"Votes: {dict(vote_counts)}")
    print(f"Selected Answer: {most_common_answer} (with {count}/{len(answers)} votes)")
    
    return most_common_answer, completions
```
