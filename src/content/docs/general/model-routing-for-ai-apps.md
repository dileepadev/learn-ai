---
title: "Model Routing for AI Apps"
description: "How applications choose the right model for each task instead of using one model for everything."
---

Model routing is the practice of sending each request to the model that best matches its difficulty, cost target, and latency needs. This is a practical way to improve both efficiency and product quality.

## Why Route Requests

Not every task needs a frontier model. Simple classification, extraction, and summarization requests can often be handled by smaller or cheaper models, while complex reasoning or sensitive tasks may need a stronger one.

## Routing Signals

- Query length and complexity
- Required output format
- Domain sensitivity
- User tier or SLA
- Confidence from a lightweight classifier

## Production Benefit

Good routing avoids the false choice between "always use the best model" and "always use the cheapest model." It lets a product adapt intelligently on a request-by-request basis.
