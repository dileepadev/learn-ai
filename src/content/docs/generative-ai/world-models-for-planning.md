---
title: World Models for Planning and Decision Making
description: Understand how AI agents use learned world models to simulate future states, plan ahead, and make better decisions — covering model-based RL, latent world models, DreamerV3, RSSM architectures, and the role of world models in autonomous driving and robotics.
---

**World models** are learned neural network representations of how an environment evolves in response to actions. Rather than learning a policy purely from direct interaction (model-free reinforcement learning), agents equipped with world models can **simulate future trajectories internally** — planning ahead, evaluating hypothetical actions, and learning from imagined experience — dramatically improving sample efficiency and enabling sophisticated long-horizon reasoning.

The concept draws directly from cognitive science: humans plan by running **mental simulations**, imagining the consequences of actions before committing to them. World models give AI agents an analogous capability.

## Model-Based vs. Model-Free RL

**Model-free RL** learns a policy $\pi(a|s)$ or value function $V(s)$ directly from environment transitions $(s, a, r, s')$:
- Simple to implement
- Can achieve high asymptotic performance given sufficient data
- **Sample-inefficient**: Requires many environment interactions (millions to billions of frames)
- Cannot plan ahead; decisions are purely reactive

**Model-based RL** additionally learns a **transition model** $\hat{p}(s'|s, a)$ and **reward model** $\hat{r}(s, a)$:
- Can generate synthetic rollouts from the learned model
- Plan by searching over action sequences in the model
- **Sample-efficient**: Imagined rollouts from the model are cheap, reducing real environment interactions
- **Model errors** can mislead planning — compounding errors over long horizons is a fundamental challenge

The tradeoff: model-based agents learn faster but are limited by the accuracy of the learned world model.

## Latent World Models

Operating in **pixel space** is prohibitively expensive for planning — simulating rollouts in image space requires generating high-dimensional observations for every step. **Latent world models** address this by operating entirely in a compressed latent space:

```
Observation → Encoder → Latent state z_t
                           ↓
                    Transition model: z_{t+1} = f(z_t, a_t)
                    Reward model:     r_t = g(z_t)
                           ↓
                    Decoder → Reconstructed observation (optional)
```

Planning happens in the **compact latent space**, not in pixel space, reducing computational cost by orders of magnitude. The decoder is only needed for visualization or auxiliary training objectives — not for planning itself.

### Recurrent State Space Model (RSSM)

The **RSSM** (Hafner et al., 2019), introduced in the **Dreamer** series, is the canonical architecture for latent world models. It maintains two complementary state representations:

- **Deterministic state** $h_t$: The hidden state of a GRU, carrying information forward through time without noise. Provides temporal continuity.
- **Stochastic state** $z_t$: A sample from a learned posterior distribution $q(z_t | h_t, o_t)$, conditioned on the current observation. Captures uncertainty.

The combined state $(h_t, z_t)$ is the **model state** used for everything downstream. This design:

- The deterministic component ensures temporal dependencies are maintained
- The stochastic component models environment stochasticity and observation uncertainty
- The separation makes the posterior tractable to learn via amortized variational inference

**Training** uses a **world model objective** combining:

1. **Reconstruction loss**: Predict observations $o_t$ from the model state
2. **Reward prediction loss**: Predict rewards $r_t$
3. **KL divergence**: Keep the learned posterior $q(z_t|h_t, o_t)$ close to the prior $p(z_t|h_t)$
4. **Termination prediction**: Predict episode ends

## DreamerV3: A Universal World Model

**DreamerV3** (Hafner et al., 2023) represents the state of the art in learned world models, achieving **unprecedented generality** — a single set of hyperparameters works across:

- Atari games (discrete visual observations)
- Continuous control (proprioceptive observations)
- Minecraft (open-ended 3D environment, sparse rewards)
- BSuite (behavioral benchmarks)

The key innovations over DreamerV2:

### Symlog Predictions

Raw rewards and values span many orders of magnitude across tasks. DreamerV3 applies a **symlog transformation**: $\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)$

This compresses large values, making the prediction problem uniform regardless of reward scale — eliminating the need for task-specific reward normalization.

### Free Bits KL Regularization

The KL divergence in the ELBO objective can cause **posterior collapse** — where the model ignores observations and relies entirely on the prior. DreamerV3 uses **free bits**: KL divergence below a threshold $\lambda$ is not penalized, preserving information content in the latent state.

### Return Normalization

Value targets are normalized by their running exponential moving average, adapting to the scale of rewards encountered during training. This stabilizes learning across diverse tasks without manual tuning.

### World Model Training in Latent Space

DreamerV3 trains the actor-critic entirely **inside the world model**:

1. Collect real trajectories and train the world model on them
2. From world model states, generate **imagined rollouts** of length $H$ (15 steps)
3. Train actor $\pi$ and critic $V$ on these imagined rollouts using TD($\lambda$)
4. Use the actor to collect new real data, repeating the cycle

This **Dyna-style** training (named after Sutton's 1991 Dyna algorithm) allows the agent to learn primarily from imagination, using real experience only to keep the world model accurate.

**Results**: DreamerV3 is the first algorithm to collect **diamonds in Minecraft** from scratch, a task requiring hundreds of steps of coherent long-horizon planning — and it does so with far fewer environment interactions than model-free baselines.

## Planning with World Models

### Model Predictive Control (MPC)

At each timestep, the agent uses the world model to **simulate $K$ candidate action sequences** of horizon $H$, evaluates each by accumulating predicted rewards, and executes the first action of the best sequence:

```
for each candidate action sequence [a_0, a_1, ..., a_{H-1}]:
    simulate: z_0 → z_1 → ... → z_H using model
    evaluate: sum predicted rewards r_0 + r_1 + ... + r_{H-1}
execute: first action of best sequence
```

MPC never stores a policy — it replans from scratch at every step. This is **flexible** (the world model can be updated between steps) but **computationally expensive** for long horizons with large action spaces.

### Cross-Entropy Method (CEM)

CEM is a popular planning algorithm for continuous action spaces. It maintains a **distribution over action sequences** (typically Gaussian) and iteratively refines it:

1. Sample $N$ action sequences from the current distribution
2. Evaluate each using the world model
3. Select the top-$k$ sequences ("elites")
4. Refit the distribution to the elite sequences
5. Repeat for $T$ iterations; execute the mean action

CEM is simple, parallelizable, and effective for short to medium horizons. **PETS** (Probabilistic Ensembles with Trajectory Sampling) combines CEM with an ensemble of probabilistic world models, propagating uncertainty through the planning horizon.

### Monte Carlo Tree Search (MCTS) + World Model

**MuZero** (Schrittwieser et al., 2019) combines MCTS planning with a learned latent world model in a landmark achievement: superhuman performance on Atari and board games (Go, Chess, Shogi) **without access to the game rules**.

MuZero's world model operates entirely in latent space:
- **Representation function** $h(o_t) = s_t$: Map observations to latent states
- **Dynamics function** $g(s_t, a_t) = (r_t, s_{t+1})$: Predict reward and next latent state
- **Prediction function** $f(s_t) = (p_t, v_t)$: Predict policy logits and value

MCTS uses these functions to simulate action trees, estimating Q-values for each action. The model learns from the backup values computed by MCTS — a form of **value-targeted world model training** that focuses learning capacity on aspects of the world relevant to decision-making.

## World Models in Physical Domains

### Autonomous Driving

Autonomous vehicles require predicting the behavior of surrounding agents (other cars, pedestrians, cyclists) to plan safe trajectories. **Occupancy flow networks** and **neural scene models** represent the driving environment as a learned world model.

**GAIA-1** (Wayve, 2023) is a generative world model for autonomous driving, trained on video data to predict future frames conditioned on actions. The model generates temporally consistent, photorealistic driving scenarios — enabling data augmentation, safety evaluation in simulated rare events, and closed-loop training of driving policies.

**UniSim** (Chen et al., 2023) generates realistic sensor data (camera, LiDAR) for autonomous driving scenarios, creating a **neural driving simulator** that can render arbitrary viewpoints and test edge cases that are difficult to encounter in the real world.

### Robotics

Physical robotic manipulation requires predicting contact dynamics, object motion, and the consequences of fine motor control — all in high-dimensional spaces with complex physics.

**RoboDreamer** and similar systems apply DreamerV3-style world models to robotic manipulation, enabling robots to plan manipulation sequences in imagination before committing to physical actions.

**Foundation world models** pretrained on large corpora of internet video (how objects move, how humans interact with them) can provide rich priors for robotic planning, reducing the amount of robot-specific data needed.

## Video Prediction as World Modeling

Large-scale **video prediction models** can be viewed as implicit world models — they learn to simulate how the visual world evolves through time. When conditioned on action tokens, they become explicit world models for interactive environments.

**DIAMOND** and **GameNGen** treat games as video prediction problems: given a sequence of past frames and actions, predict the next frame. This produces a **neural game engine** that can simulate gameplay without any traditional game engine code.

**Genie** (Google DeepMind, 2024) takes this further: trained on internet videos of 2D platformer games, it learns a latent action space from **video alone** (without action labels) — discovering the underlying structure of interactivity from visual patterns.

## Challenges and Open Problems

### Compounding Model Errors

World models accumulate errors over long horizons. A 99% accurate one-step model is 82% accurate over 20 steps (0.99^20), 67% accurate over 40 steps — and model errors at the planning horizon are indistinguishable from the real consequences. Techniques for managing error propagation:

- **Ensemble disagreement**: Use model uncertainty to terminate planning when the model is unreliable
- **Short horizons + learned value functions**: Limit planning depth; use a learned value function to estimate long-term returns
- **Data augmentation at boundaries**: Refresh model rollouts with real environment interactions periodically

### Partial Observability

Real environments are **partially observable** — the current observation does not fully determine the environment state. World models must infer hidden state from observation history. The RSSM's recurrent component addresses this, but long-horizon partial observability (e.g., an object that was in view 100 steps ago but is now occluded) remains challenging.

### Abstract and Symbolic Planning

Current world models operate at the level of low-level observations (pixels, proprioceptive states). **Abstract world models** that represent high-level concepts and relationships — enabling the kind of hierarchical, symbolic planning humans use for complex tasks — remain an open research frontier.

## The World Model Hypothesis

The **World Model Hypothesis** (proposed by Yann LeCun and others) posits that a central world model — capable of predicting the future across modalities and abstraction levels — is the missing ingredient for **human-level AI**. In this view, current AI systems are powerful pattern matchers, but true intelligence requires the ability to model the causal structure of the world and plan within that model.

Whether future AGI-level systems will be built around explicit world models, or whether implicit world knowledge emerges in Transformer-scale models without dedicated world model architecture, is one of the central unresolved questions in AI research.
