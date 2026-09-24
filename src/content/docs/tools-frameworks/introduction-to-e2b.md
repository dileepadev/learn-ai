---
title: Introduction to E2B - Secure Code Execution Sandboxes for AI Agents
description: Learn how E2B provides isolated, ephemeral sandboxes so AI agents can safely execute generated code without risking the host system.
---

E2B provides secure, isolated sandbox environments specifically designed for AI agents that need to execute LLM-generated code — running arbitrary generated code directly on a host system is a serious security risk that a code-execution sandbox is built to contain.

## Why Agents Need Sandboxed Execution

An agent that writes and runs its own code (for data analysis, debugging, or building small tools) needs somewhere to actually execute that code, but the code itself is untrusted output from a language model that could contain bugs, unintended side effects, or in an adversarial setting, actively malicious instructions from a prompt injection. Running that code directly on a production server or the host machine risks file system damage, resource exhaustion, or data exfiltration.

## Basic Usage

```python
from e2b_code_interpreter import Sandbox

sandbox = Sandbox()
execution = sandbox.run_code("""
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
""")
print(execution.logs.stdout)
sandbox.kill()
```

Each sandbox runs in an isolated microVM, giving strong isolation from the host and from other sandboxes, while still starting quickly enough (typically under a second) to fit into an interactive agent loop without noticeable added latency.

## Persistent State Within a Session

A sandbox can persist state (installed packages, written files, variables in a running interpreter) across multiple code execution calls within the same session, which matters for agents that iteratively build on previous steps — installing a library once and reusing it across several subsequent executions rather than resetting the environment each call.

## Practical Guidance

Use a dedicated code execution sandbox like E2B any time an agent needs to run generated code as part of its workflow — data analysis agents, coding assistants that verify their own output, or agents that need to process files. Always treat sandbox output (including error messages) as untrusted when feeding it back into the agent's context, since a sufficiently adversarial input could still attempt to manipulate the agent's subsequent behavior even from within an isolated execution environment.
