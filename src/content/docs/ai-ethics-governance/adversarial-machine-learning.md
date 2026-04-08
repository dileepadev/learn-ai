---
title: "Adversarial Machine Learning: Attacking the AI"
description: "An overview of how AI models can be fooled by adversarial examples and techniques for securing them."
---

**Adversarial Machine Learning** focuses on the vulnerabilities of AI models. Just as hackers attack software, "adversarial" researchers find ways to trick AI into making incorrect or dangerous decisions.

## Types of Attacks

### 1. Evasion Attacks

The most common type, where an input is subtly modified—like adding invisible noise to an image—to make a model misclassify it (e.g., a stop sign being seen as a speed limit sign).

### 2. Data Poisoning

Injecting malicious data into the training set to create a "backdoor" in the model that the attacker can trigger later.

### 3. Model Inversion

Reverse-engineering a model to extract sensitive information about the data it was trained on.

## Securing the AI

"Adversarial Training" involves including adversarial examples in the training set to make the model more robust. However, this often leads to an ongoing "arms race" between attackers and defenders.
