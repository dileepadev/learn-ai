---
title: "LLM Agent Architecture: Designing Systems That Can Plan and Act"
description: "Explore architectural patterns for building LLM-powered agents — from single-agent systems to multi-agent swarms, and how to design agents that can plan, use tools, and reason about complex tasks."
---

LLM agents go beyond simple chat interfaces. They can plan multi-step actions, use tools, maintain memory across sessions, and collaborate with other agents. This guide covers the architectural patterns that make these capabilities possible.

## What Makes Something an Agent?

An agent is an LLM-powered system that can:

1. **Reason** about goals and plan actions to achieve them.
2. **Act** by calling tools, executing code, or manipulating external systems.
3. **Remember** past interactions and learn from them.
4. **Iterate** based on observations and feedback.

Not every LLM application is an agent. A simple RAG chatbot is not an agent — it retrieves and generates but doesn't take autonomous action.

## Single-Agent Architecture

The simplest agent design: one LLM that reasons, plans, and acts.

### The Agent Loop

```python
class Agent:
    def __init__(self, model, tools, system_prompt):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.memory = []  # Conversation history
    
    def run(self, task: str) -> str:
        # 1. Plan: Decide what to do
        plan = self.plan(task, self.memory)
        
        # 2. Execute: Carry out the plan step by step
        for step in plan:
            if isinstance(step, ToolCall):
                result = self.call_tool(step.name, step.args)
                self.memory.append((step, result))
            else:
                # Final response
                return step
        
        return "Task completed"
```

### Planning Strategies

**ReAct (Reason + Act)** interleaves reasoning and action:

```
Thought: I need to find the current weather in Tokyo.
Action: get_weather(city="Tokyo")
Observation: Tokyo: 22°C, partly cloudy
Thought: The weather is mild. Let me provide the answer.
Response: It's 22°C in Tokyo today.
```

**Plan-and-Execute** generates a full plan before acting:

```
Task: Research quantum computing and write a summary.

Plan:
1. Search for "quantum computing basics"
2. Find recent developments
3. Compare different approaches
4. Write summary

Then execute each step sequentially.
```

**Self-Correcting** agents detect and fix their own errors:

```
Step 1: Call search("quantum computing")
Result: Got unrelated results about physics

Step 2 (Correction): search("quantum computing for beginners")

Step 3: Process results
```

## Tool Design

Tools are the agent's interface to the world. Well-designed tools are:

```python
# Good tool design
@tool
def search_wikipedia(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Search Wikipedia for relevant articles. Use for factual questions,
    historical information, or general knowledge queries.
    
    Args:
        query: Search query, be specific for better results
        max_results: Number of results (1-10)
    
    Returns:
        List of article titles and snippets
    """
    return wikipedia.search(query)[:max_results]

# Avoid
@tool  # Bad: Overly broad, unclear when to use
def search(query: str):
    pass
```

### Tool Categories

1. **Information retrieval**: Search, database queries, API calls.
2. **Computation**: Calculator, code execution, data processing.
3. **Action**: Sending emails, posting to social media, placing orders.
4. **Manipulation**: Reading/writing files, interacting with GUIs.

### Tool Error Handling

```python
def safe_tool_call(tool_name: str, args: dict, max_retries: int = 2):
    for attempt in range(max_retries):
        try:
            return call_tool(tool_name, args)
        except TimeoutError:
            if attempt == max_retries - 1:
                return {"error": "Timeout after retries", "retryable": False}
            wait(2 ** attempt)  # Exponential backoff
        except PermissionError:
            return {"error": "Permission denied", "retryable": False}
        except Exception as e:
            return {"error": str(e), "retryable": True}
```

## Memory Systems

Agents need memory beyond the current context.

### Memory Types

| Type | Duration | Capacity | Use Case |
|------|----------|----------|----------|
| **Working memory** | Current session | Context window | Current task |
| **Episodic memory** | Days to weeks | Vector database | Past conversations |
| **Semantic memory** | Permanent | Knowledge base | Facts and knowledge |
| **Procedural memory** | Permanent | Agent code | Skills and tools |

### Implementing Episodic Memory

```python
class EpisodicMemory:
    def __init__(self, embedding_model, vector_store):
        self.embedder = embedding_model
        self.store = vector_store
    
    def add(self, interaction: dict):
        """Store an interaction with embedding."""
        text = f"{interaction['query']} → {interaction['response']}"
        embedding = self.embedder.encode(text)
        self.store.add(embedding, metadata=interaction)
    
    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """Retrieve past interactions similar to query."""
        query_embedding = self.embedder.encode(query)
        results = self.store.search(query_embedding, k=k)
        return results
```

### Retrieving Relevant Memory

```python
def get_context_for_task(agent, task: str) -> str:
    # Retrieve recent conversations
    recent = agent.memory[-10:]
    
    # Retrieve related past experiences
    related = agent.episodic_memory.retrieve(task, k=3)
    
    # Retrieve relevant knowledge
    facts = agent.semantic_memory.retrieve(task, k=5)
    
    # Combine into context
    context = f"Task: {task}\n\n"
    context += "Related memories:\n" + format_memories(related) + "\n\n"
    context += "Relevant facts:\n" + format_facts(facts) + "\n\n"
    context += "Recent conversation:\n" + format_conversation(recent)
    
    return context
```

## Multi-Agent Architecture

Multiple agents can collaborate on complex tasks.

### Supervisor Pattern

```python
class SupervisorAgent:
    def __init__(self, workers: List[Agent]):
        self.workers = {w.name: w for w in workers}
        self.supervisor = Agent(model, tools=[], prompt=...)
    
    def handle_task(self, task: str) -> str:
        # Analyze task and assign to best worker
        worker_name = self.supervisor.plan(f"Assign this task: {task}")
        worker = self.workers[worker_name]
        
        # Delegate and collect result
        result = worker.run(task)
        
        # Synthesize final response
        return self.supervisor.synthesize(task, result)
```

### Debate Pattern

```python
class DebateAgents:
    def __init__(self, agents: List[Agent], rounds: int = 3):
        self.agents = agents
        self.rounds = rounds
    
    def debate(self, question: str) -> str:
        positions = [None] * len(self.agents)
        
        for round in range(self.rounds):
            for i, agent in enumerate(self.agents):
                # Agent considers other positions
                context = f"Question: {question}\n\n"
                context += "Other perspectives:\n"
                for j, pos in enumerate(positions):
                    if j != i and pos:
                        context += f"Agent {j}: {pos}\n"
                
                positions[i] = agent.respond(context)
        
        # Final synthesis
        return synthesize(positions)
```

### Swarm Pattern

Decentralized agent coordination:

```python
class AgentSwarm:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.message_board = MessageBoard()
    
    def solve(self, problem: str) -> dict:
        # Broadcast problem to all agents
        for agent in self.agents:
            agent.receive_task(problem)
        
        # Agents work in parallel, posting updates
        while not self.has_consensus():
            self.message_board.broadcast_updates()
            await_asyncio(1.0)  # Check periodically
        
        return self.message_board.get_consensus()
```

## Agent Evaluation

Evaluating agents is hard — they can take many paths to a goal.

### Task Completion Metrics

```python
def evaluate_agent(agent, tasks: List[dict]):
    results = []
    
    for task in tasks:
        result = agent.run(task["input"])
        success = check_success(result, task["expected"])
        metrics = {
            "success": success,
            "steps": count_steps(result),
            "time": get_execution_time(result),
            "tool_calls": count_tool_calls(result),
        }
        results.append(metrics)
    
    # Aggregate
    return {
        "success_rate": mean(r.success for r in results),
        "avg_steps": mean(r.steps for r in results),
        "avg_time": mean(r.time for r in results),
    }
```

### Evaluating Reasoning Quality

For open-ended tasks, use LLM-based evaluation:

```python
def evaluate_reasoning(output: str, criteria: List[str]) -> dict:
    evaluation_prompt = f"""Evaluate this agent's output:
    
Output: {output}
    
Criteria:
{chr(10).join(f'- {c}' for c in criteria)}
    
Rate each criterion 1-5 and explain."""
    
    eval_result = llm.generate(evaluation_prompt)
    return parse_evaluation(eval_result)
```

## Common Agent Pitfalls

1. **Infinite loops**: Agent repeatedly calls the same tool. Solution: track recent actions and detect loops.

2. **Hallucinated tools**: Agent invents tools that don't exist. Solution: constrain tool list, validate calls.

3. **Context overflow**: Agent generates very long reasoning traces. Solution: summarize history, use truncated context.

4. **Goal drift**: Agent forgets the original task. Solution: repeat goal periodically.

5. **Error cascading**: One wrong tool call leads to more errors. Solution: validate tool outputs.

Agent architecture is an evolving field. The patterns here are a starting point — each application will require customization based on the specific tasks, tools, and reliability requirements.