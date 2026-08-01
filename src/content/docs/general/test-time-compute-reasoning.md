---
title: Test-Time Compute for Better Reasoning
description: How allocating more computation during inference improves model reasoning capabilities.
---

Test-time compute refers to techniques that allow models to perform additional computation during inference to improve output quality, particularly for complex reasoning tasks.

## The Core Idea

Traditional models have fixed compute at inference time. Test-time compute methods let models "think longer" on harder problems, allocating more resources to difficult inputs while remaining efficient on easy ones.

## Approaches to Test-Time Compute

**1. Chain-of-Thought Scaling**
Generate multiple reasoning chains and verify or aggregate them. More chains provide better coverage of solution space.

**2. Search-Based Methods**
- **Best-of-N**: Generate N samples, select the best using a verifier
- **Tree-of-Thought**: Explore reasoning paths systematically with backtracking
- **MCTS**: Monte Carlo Tree Search over reasoning steps

**3. Iterative Refinement**
Models critique and improve their own outputs over multiple passes. Each iteration adds computation but improves quality.

**4. Verification**
Use separate verifier models to evaluate candidate solutions, enabling selection among multiple attempts.

## Trade-offs

**Compute vs. Quality**
More test-time compute generally improves quality but increases latency and cost. The relationship often follows diminishing returns.

**Optimal Compute Allocation**
Not all problems need the same compute. Adaptive methods allocate resources based on problem difficulty:
- Easy queries: Single forward pass
- Medium queries: 2-3 iterations
- Hard queries: Extensive search or many samples

## Practical Implementations

**OpenAI o1**
Uses reinforcement learning to learn how to effectively use test-time compute, learning when to think longer and when to answer directly.

**AlphaCode**
Generates many candidate solutions and filters them using test cases, allocating more compute to harder problems.

**Self-Consistency**
Samples multiple reasoning paths and takes majority vote, improving accuracy on reasoning benchmarks.

## When Test-Time Compute Helps

- Mathematical and logical reasoning
- Code generation with test cases
- Problems where verification is easier than generation
- Tasks requiring planning or multi-step reasoning

## Limitations

- Increased latency may be unacceptable for real-time applications
- Higher inference costs
- Requires effective verification or selection mechanisms
- May over-complicate simple problems
