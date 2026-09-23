---
title: Sim-to-Real Transfer in Robotics
description: Learn why policies trained in simulation fail on real robots, and the domain randomization and system identification techniques used to close the gap.
---

Training RL policies directly on physical robots is slow, expensive, and risky — a single training run can take days and damage hardware. Sim-to-real transfer trains policies in simulation and deploys them on real robots, but the "reality gap" between simulated and real dynamics can make a policy that works perfectly in simulation fail immediately in the real world.

## Why the Reality Gap Exists

Simulators approximate physics (friction, contact forces, actuator delays, sensor noise) imperfectly, and RL policies are notoriously good at exploiting whatever quirks exist in their training environment. A policy can learn to rely on a simulation artifact — an unrealistically consistent friction coefficient, a sensor with no noise — that simply does not hold on real hardware, producing a policy that is overfit to the simulator rather than robust to physical reality.

## Domain Randomization

Domain randomization trains the policy across many randomized variations of simulation parameters — friction, mass, sensor noise, visual textures, lighting — so that the policy cannot rely on any single fixed value and instead must learn behavior robust across the whole distribution:

```text
for each training episode:
    sample friction, mass, latency, noise ~ randomized ranges
    train policy in this randomized simulation instance
```

The real world then looks like just another sample from the training distribution rather than a fundamentally different setting, provided the randomization ranges are wide enough to actually cover real-world variation.

## System Identification and Fine-Tuning

An alternative or complementary approach, system identification, measures real robot parameters (actual friction, actual motor response curves) and calibrates the simulator to match them more precisely before training, narrowing the reality gap directly rather than training across a wide randomized range. Many pipelines combine both: pretrain broadly with domain randomization, then fine-tune briefly on limited real-world data to correct residual mismatch.

## Practical Guidance

Start with domain randomization on the physical parameters most likely to differ between sim and real for your specific hardware, rather than randomizing everything uniformly, since excessive randomization can make the learning problem needlessly harder without improving real-world transfer. Always validate transferred policies with a safety-limited real-world test phase — sim-to-real failures often appear as unexpected, hardware-damaging behavior rather than a graceful performance drop.
