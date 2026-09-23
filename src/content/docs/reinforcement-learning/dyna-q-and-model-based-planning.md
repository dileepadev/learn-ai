---
title: Dyna-Q - Combining Learning and Planning
description: Learn how the Dyna-Q architecture interleaves real experience with simulated experience from a learned model to accelerate reinforcement learning.
---

Dyna-Q is a classic architecture that combines model-free learning with model-based planning, using a learned model of the environment to generate additional simulated experience that accelerates learning beyond what real experience alone provides.

## The Core Idea

Rather than choosing between learning purely from real experience (model-free, like [[q-learning-fundamentals]]) or planning with a known model ([[policy-iteration-and-value-iteration]]), Dyna-Q does both: it updates its Q-values from real experience as usual, but also uses that same real experience to learn a model of the environment's transitions and rewards, then generates additional simulated transitions from that learned model to perform extra Q-value updates without needing more real environment interaction.

```text
loop:
    take real action a in state s -> observe s', r
    Q-learning update using real (s, a, r, s')
    update learned model: model(s, a) <- (s', r)

    repeat n times:
        sample previously visited (s, a) from memory
        s', r <- model(s, a)               # simulated transition
        Q-learning update using simulated (s, a, r, s')
```

Each real environment step generates many additional simulated Q-value updates "for free," since querying the learned model is far cheaper than acquiring a new real environment interaction.

## Why This Speeds Up Learning

Real-world or simulated environment interaction is often the most expensive part of RL, especially in robotics or any setting with slow or costly episodes. By continuing to extract value from past experience through the learned model — effectively replaying and generalizing from what's already been observed — Dyna-Q reaches good performance with far fewer real environment interactions than a purely model-free approach, at the cost of needing to learn and maintain an accurate transition model, which is itself a nontrivial learning problem.

## The Risk of Model Error

Dyna-Q's benefit depends entirely on the learned model being reasonably accurate; planning extensively with a poor or biased model can actively mislead the Q-value updates, reinforcing incorrect beliefs about the environment learned from insufficient or noisy data. This tension between exploiting a model for sample efficiency and being misled by its errors is a recurring theme across model-based RL generally, not unique to Dyna-Q.

## Practical Guidance

Dyna-Q's core idea — learn a model as a byproduct of real experience, then use it to generate extra training signal — reappears throughout modern model-based deep RL (like world-model-based agents that plan or generate synthetic rollouts inside a learned model). When real environment interaction is expensive or slow, invest in an accurate transition model specifically to enable this kind of simulated planning rather than treating model learning as an afterthought.
