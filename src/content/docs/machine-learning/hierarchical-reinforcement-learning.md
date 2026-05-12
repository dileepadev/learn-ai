---
title: Hierarchical Reinforcement Learning
description: Explore hierarchical reinforcement learning — the options framework, feudal networks, Hierarchical Actor-Critic (HAC), subgoal discovery, skill reuse, and how temporal abstraction enables agents to solve long-horizon tasks that flat RL fails to learn.
---

A fundamental challenge in reinforcement learning is **long-horizon planning**: tasks that require hundreds or thousands of sequential decisions before any reward signal arrives. Flat RL agents that operate at the level of primitive actions struggle because the credit assignment problem becomes intractable — how does an action taken 500 steps ago affect the reward received now? **Hierarchical Reinforcement Learning (HRL)** addresses this by introducing multiple levels of temporal abstraction, where higher-level policies select **goals or skills** that lower-level policies execute over extended time periods.

## Temporal Abstraction and the Options Framework

The **options framework** (Sutton, Precup, and Singh, 1999) provides a formal foundation for temporal abstraction. An **option** is a triple $\omega = (I_\omega, \pi_\omega, \beta_\omega)$:

- $I_\omega \subseteq \mathcal{S}$: the **initiation set** — states where the option can be started.
- $\pi_\omega: \mathcal{S} \times \mathcal{A} \to [0, 1]$: the **intra-option policy** — the behavior while the option is executing.
- $\beta_\omega: \mathcal{S} \to [0, 1]$: the **termination condition** — the probability of ending the option in each state.

Options generalize primitive actions: a primitive action $a$ corresponds to an option that initiates in all states, executes $a$ for one step, and terminates with probability 1. An option such as "navigate to the door" initiates when the agent sees a door in range, executes a walking policy, and terminates when the agent reaches the door.

### Semi-Markov Decision Processes

When using options, the system behaves as a **Semi-Markov Decision Process (SMDP)**: transitions happen at variable time intervals $\tau$ (the duration of the option), not at fixed discrete time steps. Value functions over options are defined using the discounted return where the discount factor accumulates over the option's duration:

$$Q(\mathcal{S}, \omega) = \mathbb{E}\left[\sum_{t=0}^{\tau-1} \gamma^t r_t + \gamma^\tau V(s_\tau)\right]$$

This formulation enables standard value-based algorithms (Q-learning, actor-critic) to be applied at the option level, with each option treated as a temporally extended "action."

## Feudal Networks

**Feudal Networks** (Vezhnevets et al., 2017, FuN) implement a two-level hierarchy:

- **Manager**: operates at a slower timescale, observes a compressed state representation, and outputs a **goal vector** $g_t$ every $c$ time steps.
- **Worker**: operates at every time step, receives the current goal $g_t$ from the manager, and selects primitive actions to move the state embedding toward the goal.

The key insight is **transition policy gradient**: the Manager is trained with a reward that includes whether the Worker successfully moved in the direction of the Manager's goal, even if the extrinsic reward is sparse. This decouples the learning signals and allows each level to be trained independently.

$$r_t^M = r_t^{env} + \frac{1}{c} \sum_{i=1}^c d_{\cos}(s_{t+i} - s_t, g_t)$$

where $d_{\cos}$ is cosine similarity. The Manager's internal reward includes both environment reward and how well the Worker executed the goal.

FuN demonstrated learning multi-step tasks in 3D mazes (VizDoom, DMLab) that flat A3C agents failed to solve, by allowing the Manager to set directional goals in the learned state space.

## Hierarchical Actor-Critic (HAC)

**Hierarchical Actor-Critic** (Levy et al., 2019) extends FuN to arbitrary numbers of levels and addresses a critical problem in HRL: **non-stationarity**. When the lower-level policy changes during training, the higher-level policy's goals become invalid — a goal of "stand up" produced by the Manager during early training may require entirely different Worker actions once the Worker has been retrained.

HAC addresses non-stationarity with two mechanisms:

### Hindsight Action Transitions

When the lower-level policy fails to achieve the higher-level goal, HAC replaces the "attempted goal" in the replay buffer with the **state that was actually achieved**. This way, the higher-level policy sees transitions as if it had set a goal that the lower level successfully achieved — making the training signal consistent regardless of lower-level competence.

### Hindsight Goal Transitions

Similarly, when storing experiences for the lower-level policy, HAC replaces the goal from the higher level with the subgoal that was actually relevant, reducing the non-stationarity in the lower level's training distribution.

These hindsight techniques draw on **Hindsight Experience Replay (HER)** and allow HAC to learn stable policies even when multiple levels are trained simultaneously.

## Discovering Options Automatically

A key unsolved problem in HRL is **automatic option discovery** — learning which skills to extract without human specification. Several approaches have been proposed:

### Subgoal Discovery via Bottleneck States

**Betweenness centrality** of a state measures how often it appears on shortest paths between other state pairs. States with high betweenness are natural subgoals — they are "bottleneck" states that must be traversed to reach many goals. The **graph-based option discovery** approach builds a transition graph from collected experience and identifies these bottleneck states algorithmically.

### Spectral Methods

**Eigenpurposes** (Machado et al., 2017) uses the **eigenvectors of the state transition matrix** to define intrinsic option objectives. The slow eigenvectors of the Laplacian of the state graph correspond to large-scale spatial structure, and each eigenvector defines an intrinsic reward that encourages exploration in that direction. Options are trained to maximize these intrinsic rewards, producing skills that correspond to navigating toward structurally important regions.

### Skill Discovery with Diversity

**Diversity is All You Need (DIAYN)** (Eysenbach et al., 2018) trains skills $z \sim p(z)$ by maximizing the mutual information between skills and states visited:

$$\mathcal{F}(\theta) = \mathbb{E}[I(S; Z)] = H(Z) - H(Z \mid S)$$

A discriminator $q_\phi(z \mid s)$ is trained to predict which skill $z$ produced a given state $s$. The skill policy is rewarded for visiting states that are discriminably different under different skills. DIAYN produces a diverse set of behaviors without any extrinsic reward, and these skills can then be composed by a higher-level policy trained on downstream tasks.

## HIRO: Hierarchical RL with Off-Policy Correction

**HIRO** (Nachum et al., 2018) trains a two-level hierarchy with off-policy data. The Manager outputs a **goal** $g$ (a target state or subspace), and the Worker is trained to reach that goal within a fixed time window using **TD3**. A critical contribution is the **off-policy correction for the Manager**:

When replaying old Manager transitions, the goal $g$ that was originally set may no longer be optimal given the current Worker policy. HIRO re-labels goals by searching for the goal $\tilde{g}$ that maximizes the log-probability of the Worker's observed actions:

$$\tilde{g} = \arg\max_{\tilde{g}} \sum_{t=0}^{c-1} \log \pi_{lo}(a_t \mid s_t, \tilde{g})$$

This correction makes the replay buffer valid under the current (updated) Worker policy, enabling stable off-policy training of the Manager.

## Language-Conditioned Hierarchical Policies

Recent work connects HRL with **large language models** to enable natural language specification of goals and subgoals. Instead of learned goal embeddings, the Manager can produce subgoal specifications as language strings, and LLMs serve as zero-shot task decomposers:

- **SayCan** (Ahn et al., 2022): LLMs score candidate skills by language plausibility; affordance models score feasibility given the robot's state. The product selects the skill to execute next.
- **Code as Policies**: LLMs write hierarchical control code where high-level functions call lower-level motor primitives, creating an explicit programmatic hierarchy.
- **GROOT / RT-2**: vision-language models provide semantic goal decomposition, grounding language goals to robot sensor states.

## Challenges and Open Problems

### Non-Stationarity

As lower-level policies change, the effective transition dynamics experienced by higher-level policies shift, making the learning problem non-stationary. HAC and HIRO's off-policy correction partially address this; fully solving it remains an open research area.

### Goal Representation

What should the Manager's goals look like? State-space goals (target state values) work in low-dimensional settings. High-dimensional observations (images, language) require learned goal representations, which introduces additional instability.

### Reward Shaping vs. Intrinsic Rewards

HRL often relies on intrinsic rewards (reaching subgoals) to guide lower-level learning. Poorly shaped intrinsic rewards can mislead the lower-level policy — a classic example is a robot rewarded for touching a door handle who learns to touch the handle repeatedly without ever opening the door.

### Exploration

Hierarchical exploration remains unsolved. The Manager's goal distribution should cover useful subgoals; the Worker needs to explore around each subgoal. Combining HRL with Count-based or curiosity-driven exploration is an active research direction.

## Applications

HRL has demonstrated value in several domains:

- **Robotic manipulation**: HAC and HIRO learn multi-step manipulation tasks (stacking blocks, opening drawers) that flat RL agents cannot learn from sparse rewards.
- **Navigation**: Feudal networks and HIRO excel at long-horizon navigation in mazes and 3D environments where the goal requires traversing many intermediate states.
- **Video games**: Hierarchical LSTM agents achieve superhuman performance on Montezuma's Revenge — an Atari game infamous for its sparse rewards and long-horizon structure that defeated flat deep RL.
- **Protein folding trajectories**: HRL can model the hierarchical unfolding process in molecular dynamics simulations.

## Summary

Hierarchical reinforcement learning decomposes long-horizon tasks by introducing temporal abstraction: higher-level managers set subgoals, lower-level workers execute them over multiple time steps. The options framework provides the formal SMDP foundation. Feudal Networks and HIRO operationalize two-level hierarchies with off-policy training. HAC addresses non-stationarity via hindsight relabeling. Automatic option discovery methods — bottleneck states, eigenpurposes, DIAYN — eliminate the need for manual skill specification. The frontier connects HRL with language models for compositional, language-conditioned planning. HRL remains a core approach for scaling RL to real-world tasks that require planning over hundreds of steps.
