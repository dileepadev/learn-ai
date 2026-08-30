---
title: Hindsight Experience Replay (HER)
description: Discover how Hindsight Experience Replay accelerates multi-goal sparse-reward reinforcement learning by turning arbitrary failures into successful demonstrations.
---

In robotics and control, most real-world tasks have **sparse binary rewards**: a robotic arm attempting to insert a peg into a hole receives a reward of $r = 0$ only if the peg is successfully inserted, and $r = -1$ (or $0$) for every unsuccessful attempt.

Under standard reinforcement learning, an agent starting with random exploration will virtually never stumble upon the exact millimetric coordinate required to trigger the reward. Because every attempted trajectory ends in total failure with identical negative returns, the gradient vanishes and learning cannot begin.

In 2017, OpenAI researchers (Andrychowicz et al.) introduced **Hindsight Experience Replay (HER)**. HER mimics human psychological learning: **even when we fail to achieve our intended objective, we have still learned how to achieve whatever outcome our actions actually produced**. By retroactively re-labeling failed trajectories with the states actually reached, HER turns 100% of failed exploration attempts into successful demonstrations.

---

## The Core Intuition of Hindsight

```
Original Episode (Intended Goal g = "Top Shelf"):
Robot moves arm -> misses shelf -> drops cup onto the [Floor Table]
Reward: -1 (Total Failure!)
Standard RL: Discards episode or learns nothing from uniform negative signal.

Hindsight Experience Replay (Re-labeled Goal g' = "Floor Table"):
Robot moves arm -> drops cup onto the [Floor Table]
Hindsight Insight: "If my goal all along had been to place the cup on the Floor Table,
                     this trajectory was 100% PERFECT!"
Re-labeled Reward: +0 (Success!)
```

By substituting the achieved state as an alternative goal, the agent immediately receives positive reward signals and learns the underlying physics of object manipulation, which quickly generalizes to arbitrary target goals.

---

## Multi-Goal RL & Universal Value Function Approximators (UVFA)

HER operates in the **Multi-Goal Reinforcement Learning** framework using **Universal Value Function Approximators (UVFA)** (Schaul et al., 2015).

Instead of conditioning policies on state $s$ alone, both the actor and the critic condition explicitly on an additional **goal vector $g \in \mathcal{G}$**:

$$\pi_\theta(a \mid s, g), \quad Q^\phi(s, a, g)$$

The binary sparse reward function is defined relative to the goal:

$$r(s, a, g) = \begin{cases} 0 & \text{if } \|\text{state\_to\_goal}(s') - g\|_2 \le \epsilon \\ -1 & \text{otherwise} \end{cases}$$

---

## The HER Replay Strategy

When an agent executes an episode of length $T$ targeting original goal $g$:

$$\tau = \left( (s_0, a_0, r_0, s_1, g), \; (s_1, a_1, r_1, s_2, g), \; \dots, \; (s_{T-1}, a_{T-1}, r_{T-1}, s_T, g) \right)$$

HER stores this transition sequence twice in the replay buffer:
1. **Original Record:** Stored with the original intended goal $g$ and actual observed rewards $r_t$.
2. **Hindsight Re-labeled Record:** A subset of transitions is copied, and goal $g$ is replaced with an **alternative goal $g'$** selected from states achieved during or after the episode, with rewards recomputed using the new goal:

$$r' = r(s_t, a_t, g')$$

```
Episode Trajectory:   s_0 ──► s_1 ──► s_2 ──► s_3 ──► s_T (Achieved State)
Intended Goal g:      [ × ] (Never reached -> all rewards -1)

Hindsight Goal g' = s_T:
At step s_{T-1} taking a_{T-1} reaching s_T:
Reward r(s_{T-1}, a_{T-1}, g') = 0 (SUCCESS!)
```

---

## Goal Sampling Strategies

How should hindsight goals $g'$ be chosen? Andrychowicz et al. evaluated four strategies:

| Strategy | Selection Method | Performance |
| :--- | :--- | :--- |
| **Final** | Set $g'$ to the final state achieved at the end of the episode ($g' = s_T$). | Good; simple to implement |
| **Future (Best)** | For transition $(s_t, a_t)$, sample $g'$ from states observed **at time $t' > t$** in the same episode. | **State-of-the-Art**; provides dense progressive learning |
| **Episode** | Sample $g'$ uniformly at random from any state visited during the episode. | Moderate |
| **Random** | Sample $g'$ uniformly at random from the global environment goal space. | Poor (fails to leverage achieved trajectory) |

### The `Future` Strategy: Why It Dominates
Under the `Future` strategy with ratio $k = 4$, for every real transition stored, 4 additional hindsight transitions are synthesized using goals achieved later in that same episode. This creates an **automatic curriculum**: the agent first learns to achieve states 1 step into the future, then 5 steps, and eventually reaches arbitrary distant goals across hundreds of steps.

---

## Combining HER with Off-Policy Algorithms

HER is an experience replay modification and **requires an off-policy algorithm** (such as DDPG, TD3, or Soft Actor-Critic), because the re-labeled transitions were generated under goals different from the behavior policy.

```python
# Conceptual HER Re-labeling Buffer
def store_episode_with_her(episode_transitions, replay_buffer, her_ratio=4):
    T = len(episode_transitions)
    
    for t in range(T):
        s_t, a_t, _, s_next, original_goal = episode_transitions[t]
        
        # 1. Store original transition
        r_orig = compute_reward(s_next, original_goal)
        replay_buffer.add(s_t, a_t, r_orig, s_next, original_goal)
        
        # 2. Sample k future hindsight goals
        for _ in range(her_ratio):
            future_idx = np.random.randint(t, T)
            hindsight_goal = state_to_goal(episode_transitions[future_idx][3])
            
            # Recompute reward under new goal
            r_her = compute_reward(s_next, hindsight_goal)
            replay_buffer.add(s_t, a_t, r_her, s_next, hindsight_goal)
```

---

## Benchmark Results on Robotic Manipulation

On OpenAI's Gym robotics benchmarks (FetchReach, FetchPush, FetchSlide, FetchPickAndPlace):
- **DDPG without HER:** 0% success rate on Push, Slide, and PickAndPlace after millions of steps (completely fails due to sparse rewards).
- **DDPG with HER:** Achieves **near 100% success rate** on PickAndPlace within a fraction of the sample budget, directly mastering complex 3D grasping and multi-stage object stacking.

---

## Key Takeaways

- Hindsight Experience Replay solves the sparse reward problem in goal-conditioned RL by re-labeling failed attempts with actually achieved states.
- The `future` sampling strategy creates an automated curriculum that bridges local motor control to distant multi-step goals.
- HER integrates seamlessly with any off-policy continuous actor-critic algorithm (DDPG, TD3, SAC).
