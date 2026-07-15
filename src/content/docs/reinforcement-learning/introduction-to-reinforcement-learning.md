---
title: Reinforcement Learning - Training Agents Through Rewards
description: Understanding RL fundamentals, Markov Decision Processes, and key algorithms.
---

Reinforcement Learning (RL) is fundamentally different from supervised learning. Instead of learning from labeled examples, RL agents learn through interaction: taking actions, receiving rewards, and discovering optimal behaviors. This post explores RL foundations.

## The RL Framework

**Core Interaction Loop:**

```
Agent observes state
    ↓
Agent takes action
    ↓
Environment transitions to new state
    ↓
Agent receives reward
    ↓
Repeat
```

**Key Entities:**

- **Agent:** The learner (robot, game AI, etc.)
- **Environment:** The domain the agent interacts with
- **State:** Description of current situation
- **Action:** What agent can do
- **Reward:** Feedback signal (positive/negative)
- **Policy:** Strategy mapping states to actions

## Markov Decision Process (MDP)

Mathematical framework for RL.

### Components

**State Space (S):** All possible states

**Action Space (A):** All possible actions

**Transition Function P(s'|s,a):**
- Probability of moving from state s to state s' taking action a
- Captures environment dynamics

**Reward Function R(s,a):**
- Immediate reward for taking action a in state s

**Discount Factor γ (gamma):**
- How much to value future rewards vs immediate rewards
- Typically 0.9-0.99
- 0: Only immediate reward matters
- 1: All rewards matter equally

### Markov Property

"The future depends only on the present, not the history"

**Implication:** Current state contains all information needed for decision

**Why It Matters:**
- Simplifies state representation
- Makes problems computationally tractable

## Value and Policy

### State Value V(s)

Expected cumulative future reward from state s.

```
V(s) = E[R_t + γ R_{t+1} + γ² R_{t+2} + ...]
       Expected sum of discounted rewards
```

**Intuition:** "How good is this state?"

### Action Value Q(s,a)

Expected cumulative reward from taking action a in state s.

```
Q(s,a) = E[R_t + γ V(s')]
         Immediate reward + future value
```

**Intuition:** "How good is this action in this state?"

### Policy π

Strategy mapping states to actions.

**Deterministic:** π(s) = a (always pick action a in state s)

**Stochastic:** π(a|s) = probability of action a in state s

**Goal:** Find optimal policy π* that maximizes cumulative reward

## Value Iteration Algorithm

Simple algorithm to learn state values.

**Bellman Equation:**
```
V(s) = max_a E[R(s,a) + γ V(s')]
       Take best action, get reward + future value
```

**Algorithm:**
```
Initialize V(s) = 0 for all states
Repeat until convergence:
    For each state s:
        V(s) = max_a [R(s,a) + γ ∑ P(s'|s,a) V(s')]
               Take best action based on current value estimates
```

**Result:** V(s) converges to true state values

**Drawback:** Requires knowing transition probabilities and rewards

## Policy Iteration

Learn policy directly.

**Algorithm:**
```
Initialize policy π
Repeat until convergence:
    Policy Evaluation: Calculate V for current policy
    Policy Improvement: Update policy to greedy w.r.t. V
```

**Convergence:** Guaranteed to find optimal policy

**Comparison:**
- Value Iteration: Single step per iteration
- Policy Iteration: Full evaluation per iteration (more stable)

## Q-Learning: Model-Free Learning

Learn without knowing transition function or rewards (discover through interaction).

**Q-Learning Update:**
```
Q(s,a) ← Q(s,a) + α [R + γ max_{a'} Q(s', a') - Q(s,a)]
         └─ Current estimate
                ↓
         └─ Temporal Difference (TD) error: how wrong we were
```

**Process:**
1. Agent takes action a in state s
2. Observes reward R and next state s'
3. Updates Q-value based on error
4. Repeats

**Key Properties:**
- Off-policy: Can learn from any behavior policy
- Model-free: No need to know environment dynamics
- Converges to optimal Q-values

### Exploration vs Exploitation

**Problem:** Agent must balance:
- **Exploration:** Try new actions to discover better rewards
- **Exploitation:** Use known good actions

**Common Strategies:**

**ε-greedy:**
```
With probability ε: Take random action (explore)
With probability 1-ε: Take best known action (exploit)
```

**Decay ε over time:** Explore early, exploit later

### Q-Learning Example: Grid World

```
Goal: Navigate from start to goal
State: (x, y) position
Actions: Up, Down, Left, Right
Rewards: -1 per step, +10 at goal
```

**Training:**
- Agent explores, taking mostly random actions initially
- Learns Q-values through interactions
- Policy gradually improves
- Eventually finds optimal path

## Deep Q-Learning (DQN)

Q-learning for high-dimensional state spaces (e.g., images).

**Problem:** Q-table for images infeasible (too many possible states)

**Solution:** Use neural network to approximate Q-values

```
Input: State (image)
    ↓
Neural Network
    ↓
Output: Q-values for all actions
```

**Training:**
- Replay buffer: Store past experiences
- Sample mini-batches: Decorrelate training data
- Target network: Stabilize learning

**Innovations:**
- Experience replay: Break correlations
- Target network: Reduce divergence
- Dueling networks: Separate value and advantage

**Breakthrough:** Mastered Atari games at human level

## Policy Gradient Methods

Learn policy directly (instead of learning Q-values).

**Idea:**
```
∇ J(π) = ∇ E[R_t]
         Gradient of expected reward
```

**Update Policy:**
```
π ← π + α ∇ J(π)
    Update in direction that increases reward
```

### REINFORCE Algorithm

Basic policy gradient:

```
Update π based on trajectory return
∇ J(π) ∝ E[∇ log π(a|s) R_t]
         Increase probability of actions that led to high returns
```

**Intuition:**
- If return was high: Increase action probability
- If return was low: Decrease action probability

### Actor-Critic Methods

Combine policy gradient (actor) with value estimation (critic).

```
Actor: Policy π(a|s) - what to do
Critic: Value V(s) - how good is state

Update Actor using Critic's value estimate
Critic learns to evaluate states
```

**Advantage:** Lower variance, faster convergence

## Policy Gradient vs Q-Learning

| Aspect | Policy Gradient | Q-Learning |
|--------|-----------------|-----------|
| **Learns** | Policy directly | Value function |
| **Action Selection** | Stochastic | Deterministic (ε-greedy) |
| **Exploration** | Built-in | Must implement |
| **Continuous Actions** | Natural | Difficult |
| **Sample Efficiency** | Lower | Higher |

## RL Applications

### Game Playing

- **Atari:** DQN surpassed human performance
- **Chess:** AlphaZero learned chess from scratch
- **Go:** AlphaGo defeated world champions
- **Dota 2, StarCraft:** Complex multi-agent games

### Robotics

- **Control:** Robot arm reaching targets
- **Navigation:** Robot path planning
- **Manipulation:** Grasping objects
- **Learning:** Robots learning from interaction

### Autonomous Driving

- **Decision Making:** Navigate traffic
- **Path Planning:** Optimal route
- **Safety:** Learn collision avoidance

### Resource Allocation

- **Data Center Cooling:** Optimize energy usage
- **Traffic Control:** Optimize flow
- **Power Grid:** Manage electricity distribution

## Challenges in RL

### Sample Efficiency

**Problem:** Need many interactions to learn

**Solutions:**
- Imitation learning: Learn from demonstrations first
- Transfer learning: Start from related task
- Model-based RL: Learn environment model

### Sparse Rewards

**Problem:** Agent rarely receives feedback

**Solutions:**
- Reward shaping: Add intermediate rewards
- Hierarchical RL: Break into subproblems
- Curiosity-driven: Explore interesting areas

### Non-Stationary Environment

**Problem:** Environment changes during learning

**Solutions:**
- Continual learning
- Adapt policies online
- Robust policies

### Exploration in Large Spaces

**Problem:** Too many possibilities to explore

**Solutions:**
- Curiosity-driven exploration
- Count-based exploration
- Intrinsic motivation

## Practical Considerations

### Simulation vs Real World

**Simulation:**
- Infinite interactions
- Fast learning
- No real-world constraints

**Real World:**
- Sample efficient
- Costly mistakes
- Reality gap

**Solution:** Learn in simulation, transfer to real world

### Reward Design

"What you measure is what you get"

- Poor reward design → Poor behavior
- Agent optimizes what you specify, not what you intend

**Example:**
```
Goal: Maximize game score
Bad reward: Just final score (sparse)
Better reward: Score increments (shaped)
```

## RL Frameworks

### OpenAI Gym

Standardized RL environments:
- Classic control problems
- Atari games
- Robotics simulation

### PyTorch/TensorFlow

Deep learning frameworks for RL:
- Build custom agents
- Full control

### RLlib

Distributed RL framework:
- Scales to many workers
- Production ready

## Conclusion

Reinforcement learning enables agents to learn optimal behavior through interaction. The Markov Decision Process provides the mathematical framework. Value-based methods (Q-learning) and policy-based methods (policy gradient) offer different approaches, each with tradeoffs. Deep RL brings RL to high-dimensional problems like game playing and robotics. While RL faces challenges like sample efficiency and reward design, it remains powerful for problems where learning from demonstration or explicit specification is infeasible. Understanding RL fundamentals is essential as autonomous systems become more prevalent.
