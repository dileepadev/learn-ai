---
title: Introduction to Temporal - Durable Execution for Long-Running Agents
description: Learn how Temporal's durable execution model keeps long-running, stateful workflows reliable across crashes, retries, and version changes.
---

Temporal is a durable execution platform: it lets you write ordinary code for long-running, stateful workflows while Temporal guarantees that execution state survives process crashes, deployments, and infrastructure failures.

## Durable Execution

A Temporal workflow is written as regular code, but every step's result is recorded in an event history. If the process running the workflow crashes, Temporal replays the event history against the same workflow code to reconstruct exact in-memory state, then resumes exactly where it left off — no lost progress, no manual checkpointing logic required from the developer.

```python
@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, task: str) -> str:
        plan = await workflow.execute_activity(make_plan, task)
        for step in plan.steps:
            await workflow.execute_activity(execute_step, step, retry_policy=...)
        return await workflow.execute_activity(summarize_results, plan)
```

Activities (the actual side-effecting work, like calling an LLM or an external API) run outside the deterministic workflow code and can be retried automatically according to a configurable policy without corrupting workflow state.

## Why This Matters for AI Agents

Long-running AI agents that plan, call tools, wait on human approval, or run multi-hour research tasks need exactly the guarantees Temporal provides: surviving a deploy mid-task, retrying a flaky LLM call without restarting the whole agent, and maintaining consistent state across steps that might span minutes to days. Building this reliability by hand — with manual state persistence, idempotency keys, and retry logic — is a substantial undertaking that Temporal absorbs into the platform.

## Signals and Human-in-the-Loop

Temporal workflows can receive signals — external messages that inject new information into a running workflow, such as a human approving or editing an agent's proposed plan mid-execution. This makes Temporal a natural fit for agent architectures that need to pause for human review without losing accumulated context or restarting from scratch.

## Practical Guidance

Reach for Temporal when agent workflows are long-running, need to survive deploys and crashes without losing progress, or require durable human-in-the-loop checkpoints. For short-lived, single-request agent interactions that complete within seconds, the operational overhead of running a Temporal cluster is usually not worth it — a simpler orchestration approach or direct function calls suffice.
