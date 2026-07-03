---
title: "Evaluation Drift in AI Systems"
description: "Why an evaluation set that was useful last quarter may stop reflecting real-world AI behavior today."
---

Evaluation drift happens when your benchmark or golden dataset no longer reflects the requests, edge cases, and product goals that matter in production. The system may appear stable on paper while quietly getting worse for real users.

## How Drift Happens

- User behavior changes over time
- New product features create new failure modes
- Models improve in some areas and regress in others
- Old benchmarks get overfit through repeated tuning

## Warning Signs

If offline scores stay flat or improve while user complaints increase, your evaluation process may be drifting away from reality. Another signal is when a model performs well on tests but struggles with newly common request types.

## How to Respond

Good teams refresh their evaluation sets regularly, add recent failures from production, and review whether the scoring rubric still matches business needs. An eval is only valuable if it stays connected to the real job the system is doing.
