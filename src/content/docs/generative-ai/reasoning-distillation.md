---
title: Reasoning Distillation - Transferring Chain-of-Thought to Smaller Models
description: Learn how reasoning traces from large models are distilled into smaller students through supervised fine-tuning, and where the approach succeeds and falls short.
---

Reasoning distillation trains a smaller model to imitate the step-by-step reasoning of a stronger teacher. Rather than only matching final answers, the student learns from the teacher's intermediate reasoning traces.

## Why Distill Reasoning

Large reasoning models spend many tokens thinking before answering, which is accurate but expensive. A distilled small model can capture much of that capability at a fraction of the serving cost. Notably, open-source models distilled from strong reasoning-model traces reached surprisingly high scores on math and coding benchmarks compared with training small models using reinforcement learning from scratch.

## The Standard Recipe

1. **Collect prompts**: diverse problems in math, code, science, and logic
2. **Sample traces**: have the teacher generate reasoning plus a final answer
3. **Filter**: keep traces whose final answers are verifiably correct (using tests, answer matching, or a verifier)
4. **Fine-tune**: supervised training of the student on prompt, reasoning, and answer
5. **Optional RL**: further improve the student with reinforcement learning against verifiable rewards

## Data Format

```json
{
  "prompt": "If 3x + 5 = 20, what is x?",
  "reasoning": "Subtract 5 from both sides: 3x = 15. Divide by 3: x = 5.",
  "answer": "5"
}
```

## Key Design Choices

- **Quantity versus quality**: a few thousand carefully curated, difficult examples can already shift behavior; more data helps with breadth
- **Trace length**: overly long traces teach the student to ramble; some pipelines select shorter correct traces
- **Diversity**: cover varied problem types to avoid narrow specialization
- **Multiple traces per prompt**: increases coverage of solution paths

## Limitations

- The student imitates the *form* of reasoning and may not reproduce the underlying competence on unfamiliar problems
- It is bounded by the teacher: errors in teacher traces propagate
- Terms of service for hosted teacher models may restrict using their outputs for training
- Faithfulness is uncertain, since written reasoning may not reflect the actual computation

## Evaluation

Test on held-out problems that differ from the training distribution, and compare accuracy against token cost, since a student that reaches the same accuracy with fewer tokens is a real win.
