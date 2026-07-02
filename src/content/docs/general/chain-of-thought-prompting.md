---
title: "Chain of Thought Prompting: Making AI Reasoning Visible"
description: "How encouraging step-by-step reasoning improves AI accuracy and how to structure prompts for better thinking."
---

Ask an AI to solve "If a store has 15 apples and sells 6, how many are left?" it might answer instantly. Ask it to show its work, and it becomes more accurate, especially on complex problems. This is Chain of Thought (CoT) prompting.

## The CoT Difference

**Without CoT:**
```
Question: A store has 15 apples. They sell 6. How many are left?
Model: 9
```

**With CoT:**
```
Question: A store has 15 apples. They sell 6. How many are left?

Let me think step by step:
1. Initially: 15 apples
2. Sold: 6 apples
3. Remaining: 15 - 6 = 9 apples

Answer: 9
```

Both reach the correct answer, but the reasoning process makes the model more accurate on harder problems.

## Why CoT Works

1. **Intermediate Verification:** Errors in early steps become obvious
2. **Decomposition:** Complex problems break into manageable pieces
3. **Token Economics:** Generating reasoning tokens prevents hallucinations (the model "thinks" instead of guessing)
4. **Self-Correction:** The model can catch its own mistakes while reasoning

## Prompt Structures for CoT

### 1. Few-Shot CoT
Provide examples with reasoning:

```
Example 1:
Question: Maria has 3 cats. For each cat, she buys 2 toys. How many toys total?
Reasoning: She has 3 cats × 2 toys per cat = 6 toys
Answer: 6

Example 2:
Question: A car travels 60 mph for 2 hours, then 45 mph for 1 hour. Total distance?
Reasoning: 
- First part: 60 mph × 2 hours = 120 miles
- Second part: 45 mph × 1 hour = 45 miles
- Total: 120 + 45 = 165 miles
Answer: 165

Now solve:
Question: ...
```

### 2. Zero-Shot CoT
Just ask for reasoning directly:

```
Question: [complex problem]

Let me think step by step:
```

Surprisingly effective even without examples.

### 3. Structured Reasoning

```
Question: [problem]

Step 1: Identify what we know
Step 2: Identify what we need to find
Step 3: Choose a strategy
Step 4: Execute the strategy
Step 5: Verify the answer
```

## When CoT Helps Most

| Task | Benefit |
|------|---------|
| **Math/Logic** | High - step-by-step computation |
| **Code Generation** | High - algorithmic thinking |
| **Multi-step Planning** | High - breaks into substeps |
| **Reasoning Puzzles** | Very High - naturally suited to CoT |
| **Creative Writing** | Low - doesn't help much |
| **Summarization** | Low - doesn't help much |
| **Classification** | Medium - sometimes helpful |

## Advanced Techniques

### Tree of Thought
Instead of one chain, explore multiple reasoning paths:

```
Initial problem
├─ Strategy A
│  ├─ Sub-path A1 ✗ (dead end)
│  └─ Sub-path A2 ✓ (promising)
├─ Strategy B
│  └─ Sub-path B1 ✓ (solution found)
└─ Strategy C
   └─ Sub-path C1 ✗ (dead end)
```

### Self-Consistency
Generate multiple reasoning chains and vote on the answer:

```
Chain 1: [reasoning] → Answer: 42
Chain 2: [reasoning] → Answer: 42
Chain 3: [reasoning] → Answer: 41

Majority vote: 42 (2/3 agreement)
```

### Instruction Refinement
```
"Solve this problem by breaking it into small steps.
For each step, explain your reasoning clearly.
If you're uncertain, explain why and consider alternatives."
```

## Cost-Performance Tradeoff

**Without CoT:**
- Input: 100 tokens
- Output: 50 tokens
- Total: 150 tokens per request

**With CoT:**
- Input: 100 tokens  
- Output: 200 tokens (includes reasoning)
- Total: 300 tokens per request (2x cost)

**But:** CoT typically improves accuracy 5-15%, making it worth the extra tokens for accuracy-sensitive tasks.

## Limitations

- Doesn't guarantee correct answers (reasoning can be wrong)
- Adds latency (more tokens to generate)
- Doesn't work well for creative or subjective tasks
- Can confabulate reasoning to support incorrect answers