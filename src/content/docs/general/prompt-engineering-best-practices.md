---
title: "Prompt Engineering: Best Practices and Anti-Patterns"
description: "Proven techniques to write better prompts and common mistakes that degrade AI performance."
---

The difference between a mediocre AI response and an excellent one is often just the prompt. Good prompting is a skill that directly impacts the quality of your AI system.

## Core Principles

### 1. Clarity Over Cuteness
**Bad:**
```
"yo, can u help me with some code? 🤔"
```

**Good:**
```
"Write a Python function that validates email addresses using regex"
```

### 2. Specificity Beats Vagueness
**Bad:**
```
"Explain AI"
```

**Good:**
```
"Explain how transformer models use attention mechanisms to process text sequences"
```

### 3. Context Matters
**Bad:**
```
"Is this good?"
```

**Good:**
```
"Review this customer service response for tone, accuracy, and helpfulness:
[response text]"
```

## Structural Best Practices

### System vs. User Messages
Keep instructions and data separate:

```
System Message (your rules):
"You are a Python expert. Answer only Python questions."

User Message (the actual request):
"Write a function to reverse a string"
```

### Format Your Input
```
Use clear sections:

CONTEXT:
[background information]

TASK:
[what you want done]

CONSTRAINTS:
[limitations or requirements]

FORMAT:
[how you want the output]
```

### Role-Based Prompting
Assign the AI a role:

```
GOOD: "You are an experienced product manager. Analyze this feature request and identify potential issues"

VS.

BAD: "Analyze this feature request"

Why: The first prompt activates relevant expertise patterns
```

## Output Control Techniques

### 1. Specify Output Format
```
"Respond in JSON format:
{
  'action': 'string',
  'priority': 'high/medium/low',
  'reasoning': 'string'
}"
```

### 2. Length Constraints
```
"In 100 words or less: ..."
"Keep your answer to 2-3 sentences: ..."
"Write an exhaustive 5-paragraph essay: ..."
```

Specific lengths > generic "be concise"

### 3. Example Output
```
"Output should look like:
Status: [good/bad]
Risk Level: [1-10]
Next Steps: [list]"
```

## Prompt Patterns

### The Persona Pattern
```
"You are a [expert role] with [specific expertise].
Your communication style is [style].
You [specific trait or capability].

Now, [task]"
```

Example:
```
"You are a security expert specializing in cloud infrastructure.
Your communication style is direct and technical.
You identify risks clearly and provide actionable mitigations.

Now, review this AWS configuration for security issues."
```

### The Few-Shot Pattern
```
"Here are examples of [task]:
[Example 1]
[Example 2]
[Example 3]

Now, [new task similar to examples]"
```

### The Chain-of-Thought Pattern
```
"Solve this step by step:
1. [first step]
2. [second step]
3. [final step]

Problem: [task]"
```

Or even simpler:
```
"Solve this problem step by step:
[problem]"

(Model will naturally break it down)
```

### The Constraint Pattern
```
"Given these constraints:
- [constraint 1]
- [constraint 2]
- [constraint 3]

Complete this task: [task]"
```

## Temperature and Randomness

For prompts:
- **Deterministic task** (extraction, classification): Temperature 0-0.3
- **Balanced task** (general Q&A): Temperature 0.7-0.9
- **Creative task** (brainstorming): Temperature 1.0-1.2

Don't use high temperature for fact-based tasks.

## Anti-Patterns (Things to Avoid)

### ❌ Being Too Polite
```
BAD: "If you don't mind, could you possibly help with...?"
GOOD: "Summarize this document"

Why: AI isn't offended; extra politeness wastes tokens
```

### ❌ Apologizing Unnecessarily
```
BAD: "Sorry to bother you, but I have a complex question..."
GOOD: "[Ask the complex question directly]"

Why: Again, wastes tokens and adds nothing
```

### ❌ Weak Constraints
```
BAD: "Try to be concise"
GOOD: "In 150 words or less"

Why: Specific constraints work; vague suggestions don't
```

### ❌ Contradictory Instructions
```
BAD: "Be creative but accurate", "Think outside the box but follow all guidelines"
GOOD: "Prioritize accuracy. You may suggest creative alternatives if they're factually sound."

Why: Resolve conflicts explicitly
```

### ❌ Expecting Common Sense
```
BAD: "Assume you know the context"
GOOD: "[Provide explicit context]"

Why: AI can't read minds; provide what you think is obvious
```

### ❌ Vague Success Criteria
```
BAD: "Write good code"
GOOD: "Write efficient Python code that handles edge cases and follows PEP 8"

Why: Concrete criteria enable better output
```

## Advanced Techniques

### Negative Examples
```
"Here's what NOT to do:
[example of bad output]

Now do this correctly:
[task]"
```

Why: Negative examples help model understand boundaries.

### Decomposition
```
Instead of:
"Build a complete e-commerce system"

Use:
"First, design the database schema for an e-commerce system.
Then, write the API endpoints for product management."

Why: Breaks complex tasks into manageable pieces
```

### Authority Grounding
```
"According to [authoritative source], ..."
"Based on industry best practices, ..."
"Following the official documentation, ..."

Why: Anchors model in credible information
```

### Hypothesis Testing
```
"I believe [hypothesis]. Is this correct?
Here's my reasoning: [reasoning]"

Why: Model corrects incorrect assumptions more effectively
```

## Testing Your Prompts

### 1. A/B Testing
```
Version A: Generic prompt
Version B: Structured prompt
Version C: Few-shot prompt

Test on 10-20 examples, measure quality
Pick the best version
```

### 2. Adversarial Testing
```
Try to break your prompt:
- What if input is ambiguous?
- What if input is very long?
- What if input is very short?
- What if input is in a different language?

Refine prompt based on failures
```

### 3. Consistency Testing
```
Ask the same question 5 times
Do you get similar answers?
If not, your prompt is under-specified
```

## Prompt Optimization Checklist

- [ ] Is my task clearly defined?
- [ ] Did I provide necessary context?
- [ ] Did I specify the output format?
- [ ] Are my constraints specific and clear?
- [ ] Did I remove unnecessary politeness?
- [ ] Did I resolve any conflicting instructions?
- [ ] Is my prompt using the right temperature?
- [ ] Have I tested on representative examples?
- [ ] Does the output meet my success criteria?
- [ ] Are there edge cases I haven't considered?

## Real-World Examples

### Example 1: Code Review
**Before:**
```
"Review this code"
```

**After:**
```
"Review this Python code for:
1. Security vulnerabilities
2. Performance issues
3. Adherence to PEP 8
4. Potential bugs or edge cases

Output format: 
- Issue: [description]
- Severity: [high/medium/low]
- Fix: [suggested fix]

Code:
[code]"
```

### Example 2: Content Classification
**Before:**
```
"Is this positive or negative?"
```

**After:**
```
"Classify this customer review as:
- Positive (customer satisfied)
- Negative (customer unsatisfied)
- Neutral (no clear sentiment)
- Mixed (both positive and negative elements)

Also provide:
- Confidence: [0-100]%
- Key sentiment words: [list]
- Suggested response if needed: [optional]

Review: [review text]"
```

## The Golden Rule

**Treat the AI like a smart but new employee:**
- Give explicit instructions (don't assume they know)
- Provide examples (show what good looks like)
- Define success criteria (how will you judge success?)
- Provide context (what problem are we solving?)