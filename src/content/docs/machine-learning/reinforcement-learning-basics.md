---
title: "Reinforcement Learning Basics: Foundations of Agent Learning"
description: "Learn the fundamentals of reinforcement learning — from MDPs and value functions to Q-learning and policy gradients, the algorithms behind AI that learns from experience."
---

Reinforcement learning (RL) is a paradigm of machine learning where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, RL doesn't require labeled data — the agent learns from the consequences of its actions.

## The RL Framework

An RL agent operates within an environment, taking actions and receiving rewards:

```python
class RLEnvironment:
    def reset(self):
        """Return initial state."""
        pass
    
    def step(self, action):
        """Apply action, return (next_state, reward, done)."""
        pass
    
    def render(self):
        """Visualize the environment."""
        pass

class RLAgent:
    def select_action(self, state):
        """Choose an action based on current state."""
        pass
    
    def update(self, state, action, reward, next_state):
        """Learn from the transition."""
        pass
```

### The Markov Decision Process

RL problems are formalized as Markov Decision Processes (MDPs):

```python
class MDP:
    def __init__(self, states, actions, transition_probs, rewards, gamma, horizon=None):
        self.states = states              # State space
        self.actions = actions            # Action space
        self.P = transition_probs         # P(s'|s,a): transition probabilities
        self.R = rewards                  # R(s,a): expected reward
        self.gamma = gamma                # Discount factor (0-1)
        self.H = horizon                  # Maximum steps (None = infinite)
```

### The Reward Signal

```python
# Example: CartPole environment reward
def cartpole_reward(state, action, next_state):
    """Reward is 1 for each step the pole stays upright."""
    angle = next_state[2]
    return 1.0 if abs(angle) < 0.15 else 0.0

# Example: Chess reward
def chess_reward(outcome):
    """
    +1 for win, -1 for loss, 0 otherwise
    (intermediate rewards are sparse in chess)
    """
    return outcome
```

## Value Functions

Value functions estimate how good it is to be in a state or take an action:

```python
class ValueFunction:
    def value(self, state):
        """V(s): Expected return from state s following policy π."""
        pass

class ActionValueFunction:
    def q_value(self, state, action):
        """Q(s,a): Expected return from taking action a in state s."""
        pass
```

### The Bellman Equations

The Bellman equation expresses value functions recursively:

```python
def bellman_optimality(state, value_function, mdp):
    """
    V*(s) = max_a [ R(s,a) + γ * Σ P(s'|s,a) * V*(s') ]
    """
    q_values = []
    for action in mdp.actions:
        expected_reward = mdp.R[state][action]
        future_value = sum(
            mdp.P[state][action][next_state] * value_function.value(next_state)
            for next_state in mdp.states
        )
        q_values.append(expected_future_value := expected_reward + mdp.gamma * future_value)
    
    return max(q_values)
```

## Q-Learning: Learning Action Values

Q-learning learns the optimal action-value function directly:

```python
class QLearning:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, gamma=0.99):
        self.Q = initialize_q_table(state_dim, action_dim)
        self.lr = learning_rate
        self.gamma = gamma
    
    def select_action(self, state, epsilon=0.1):
        """ε-greedy action selection."""
        if random.random() < epsilon:
            return random.choice(range(self.action_dim))
        return argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        """
        Q(s,a) ← Q(s,a) + α [ r + γ * max_a' Q(s',a') - Q(s,a) ]
        """
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state])
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.Q[state][action] = current_q + self.lr * td_error
```

### Deep Q-Networks (DQN)

For large state spaces, use neural networks to approximate Q-values:

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DeepQLearning:
    def __init__(self, state_dim, action_dim):
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        self.replay_buffer = ReplayBuffer(capacity=100000)
    
    def update(self, batch_size=32):
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions)
        
        # Compute target Q values (with target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        """Copy weights from Q network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
```

## Policy Gradient Methods

Instead of learning values, directly optimize the policy:

```python
class PolicyGradient:
    def __init__(self, policy_network, lr=0.01):
        self.policy = policy_network
        self.optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)
    
    def select_action(self, state):
        """Sample action from policy distribution."""
        probs = self.policy(state)
        dist = Categorical(probs)
        return dist.sample().item()
    
    def update(self, trajectories):
        """
        Policy Gradient: ∇J(θ) = E[ ∇log π(a|s) * Q(s,a) ]
        
        trajectories: list of (state, action, reward, ...) sequences
        """
        returns = []
        for traj in trajectories:
            # Compute discounted returns
            R = 0
            returns_traj = []
            for r in reversed(traj.rewards):
                R = r + self.gamma * R
                returns_traj.append(R)
            returns.extend(reversed(returns_traj))
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        # Compute policy loss
        loss = 0
        for t, (state, action, R) in enumerate(zip(traj.states, traj.actions, returns)):
            log_prob = torch.log(self.policy(state)[action])
            loss += -log_prob * R  # Negative because we maximize
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## Proximal Policy Optimization (PPO)

PPO is a simple and effective policy gradient method:

```python
class PPO:
    def __init__(self, policy, clip_epsilon=0.2, lr=3e-4):
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    def update(self, states, actions, old_log_probs, advantages, returns, epochs=10):
        """Update policy using PPO clipped objective."""
        for _ in range(epochs):
            # Get new action distribution
            new_probs = self.policy(states)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.policy.value(states)
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

## Exploration vs Exploitation

Balancing exploration and exploitation is key in RL:

```python
class ExplorationStrategies:
    def epsilon_greedy(self, state, Q, epsilon=0.1):
        if random.random() < epsilon:
            return random_action()
        return argmax(Q(state))
    
    def ucb(self, state, Q, counts, c=2):
        """Upper Confidence Bound: prefer less-tried actions."""
        u = c * sqrt(log(sum(counts)) / counts)
        return argmax(Q(state) + u)
    
    def boltzmann(self, state, Q, temperature=1.0):
        """Softmax over Q values."""
        exp_q = exp(Q(state) / temperature)
        probs = exp_q / sum(exp_q)
        return sample(probs)
```

## Training RL Agents

```python
class RLTrainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    
    def train(self, num_episodes=1000, verbose_interval=100):
        returns = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Store and update
                self.agent.replay_buffer.add(state, action, reward, next_state, done)
                if len(self.agent.replay_buffer) > batch_size:
                    self.agent.update()
                
                state = next_state
                total_reward += reward
            
            returns.append(total_reward)
            
            if episode % verbose_interval == 0:
                avg_return = mean(returns[-verbose_interval:])
                print(f"Episode {episode}: Avg Return = {avg_return:.2f}")
        
        return returns
```

Reinforcement learning powers some of the most impressive AI achievements — from game playing to robotics. Understanding these fundamentals opens the door to building agents that learn from experience rather than data.