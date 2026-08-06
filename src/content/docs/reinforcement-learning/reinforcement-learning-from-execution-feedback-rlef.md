---
title: Reinforcement Learning from Execution Feedback (RLEF)
description: Understand RLEF, an alignment and post-training methodology for coding and reasoning agents using deterministic execution environments and compiler pass/fail signals.
---

Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) have proven highly successful for aligning LLMs on subjective tasks like writing, summarization, and helpful chat. However, for objective reasoning tasks — such as software engineering, formal mathematical proving, and tool manipulation — human preference models introduce severe limitations:
- Human evaluators struggle to spot subtle bugs or logic errors in complex code.
- Reward models trained on human preferences can be easily hacked by verbose or plausible-sounding incorrect answers.

**Reinforcement Learning from Execution Feedback (RLEF)** replaces subjective reward models with **deterministic runtime execution signals** obtained directly from code interpreters, compilers, unit test suites, and formal verifiers.

## How RLEF Works

RLEF frames code generation and automated reasoning as a Markov Decision Process (MDP) where the environment provides hard ground-truth feedback:

```
+---------------+     Prompt / Problem     +---------------+
| Code Problem  | -----------------------> | LLM Policy    |
+---------------+                          +---------------+
                                                   |
                                            Generated Code
                                                   v
+---------------+     Pass / Fail / Trace  +---------------+
| Reward Engine | <----------------------- | Code Sandbox  |
+---------------+                          +---------------+
        |
    Reward Signal (R)
        v
+---------------+
| Policy Update | (GRPO / PPO / Rejection Fine-Tuning)
+---------------+
```

### 1. Generation Step
The model (policy $\pi_\theta$) samples multiple candidate code solutions for a programming problem or mathematical proof.

### 2. Execution Sandbox
Each generated program is executed in a secure, isolated sandbox environment (e.g., Docker container or Firecracker microVM) against:
- Unit tests & integration tests
- Static analysis & linter checks
- Runtime memory and execution time limits

### 3. Feedback Extraction
The environment returns explicit signals:
- `EXECUTION_SUCCESS` (Passes 100% of hidden unit tests)
- `ASSERTION_FAILURE` (Fails specific input/output test cases)
- `SYNTAX_ERROR` / `COMPILATION_ERROR`
- `TIMEOUT` / `MEMORY_LIMIT_EXCEEDED`

### 4. Reward Shaping
Rewards are calculated using precise deterministic scoring:
$$R(x, y) = \begin{cases}
+1.0 & \text{if all unit tests pass} \\
-0.5 & \text{if compilation/syntax error} \\
\frac{N_{\text{passed}}}{N_{\text{total}}} - \gamma \cdot \text{TimePenalty} & \text{if partial test pass}
\end{cases}$$

## Key Optimization Algorithms in RLEF

### Group Relative Policy Optimization (GRPO)
Used extensively in models like DeepSeek-R1, GRPO eliminates the need for a separate critic network by sampling a group of $G$ solutions $\{y_1, y_2, \dots, y_G\}$ for prompt $x$ and computing normalized relative rewards:

$$A_i = \frac{R(x, y_i) - \text{mean}(R)}{\text{std}(R)}$$

The policy is updated using clip objective optimization based on these relative execution advantage scores.

### Execution-Guided Iterative Rejection Fine-Tuning (RFT)
Instead of continuous RL gradients, RFT samples thousands of candidate programs per problem, filters out any program that fails execution, and fine-tunes the base LLM on the verified successful trajectories (SFT on verified data).

## RLEF vs. RLHF: Feature Comparison

| Feature | RLHF / DPO | RLEF |
|---|---|---|
| **Reward Source** | Learned Neural Reward Model | Deterministic Compiler / Sandbox |
| **Task Domain** | Open-ended text, style, chat | Code, math, SQL, formal logic |
| **Reward Hacking** | High (Model exploits reward model flaws) | Virtually Zero (Tests either pass or fail) |
| **Scalability** | Expensive (Requires human annotators) | Infinite (Automated code execution) |
| **Feedback Fine-Granularity** | Coarse preference scores | Line-level stack traces & unit test counts |

## Python Conceptual Workflow

```python
import docker

def execute_in_sandbox(code_str: str, unit_tests: str) -> dict:
    """Executes generated code against unit tests in an isolated sandbox."""
    full_script = f"{code_str}\n\n{unit_tests}"
    
    client = docker.from_env()
    try:
        container = client.containers.run(
            image="python:3.11-slim",
            command=["python3", "-c", full_script],
            detach=False,
            mem_limit="256m",
            nano_cpus=1000000000,  # 1 CPU
            timeout=5  # 5 second timeout
        )
        return {"status": "SUCCESS", "reward": 1.0}
    except docker.errors.ContainerError as e:
        return {"status": "TEST_FAILED", "reward": 0.0, "error": str(e)}
    except Exception as e:
        return {"status": "TIMEOUT_OR_CRASH", "reward": -0.5, "error": str(e)}
```

## Challenges in RLEF

1. **Flaky & Non-Deterministic Tests:** Unit tests relying on network calls, timing, or random seeds can introduce noisy reward signals.
2. **Execution Overhead:** Running millions of code sandboxes in parallel during policy optimization requires infrastructure optimization.
3. **Specification Incompleteness:** If unit test suites are incomplete, the LLM may learn code solutions that pass the weak tests but fail edge cases.

## Summary

Reinforcement Learning from Execution Feedback transforms post-training for code and reasoning models. By substituting subjective human preferences with automated compiler and test suite feedback, RLEF enables models to achieve superhuman performance on complex software engineering benchmarks like SWE-bench and HumanEval.

## Further Reading

- DeepSeek AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- Shao et al. (2024), *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*
- Le et al. (2022), *CoderRL: Mastering Code Generation through Pretrained Models and Execution Feedback*
