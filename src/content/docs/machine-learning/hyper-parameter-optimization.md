---
title: "Hyper-Parameter Optimization (HPO): Automated ML"
description: "Techniques for automatically finding the best settings for your neural network architecture."
---

Picking the right learning rate, batch size, and number of layers is often called "alchemy." **Hyper-Parameter Optimization (HPO)** turns this into a rigorous mathematical search.

## Common Strategies

- **Grid Search**: Trying every possible combination of settings (expensive and slow).
- **Random Search**: Often more efficient than grid search, as it explores the space more broadly.
- **Bayesian Optimization**: Uses a probabilistic model to predict which settings are likely to work best, focusing the search where it's most needed.

## Auto-ML

HPO is the foundation of "Auto-ML" tools, which aim to let users upload a dataset and receive a fully optimized model without writing a single line of deep learning code.
