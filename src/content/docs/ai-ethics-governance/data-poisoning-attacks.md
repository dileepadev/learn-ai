---
title: Data Poisoning Attacks on Machine Learning Systems
description: Learn how attackers corrupt training data to manipulate model behavior, the main attack categories, and the defenses that mitigate them.
---

Data poisoning attacks manipulate a model's training data to corrupt its behavior, exploiting the fact that most machine learning systems trust their training data implicitly rather than verifying it comes from a trustworthy source.

## Why Training Data Is an Attack Surface

Many modern models train on data scraped from the open web, collected from user-generated content, or aggregated from external partners — sources an attacker can often influence, at least partially, without needing any access to the model's infrastructure at all. Unlike adversarial examples (crafted inputs that fool an already-trained model at inference time, discussed in [[adversarial-examples]]), data poisoning corrupts the model itself during training, so the resulting misbehavior persists across every future inference the poisoned model makes, not just for a single crafted input.

## Attack Categories

Availability attacks aim to broadly degrade a model's overall accuracy by injecting enough mislabeled or corrupted examples into the training set, typically requiring the attacker to control a significant fraction of the training data to have a meaningful effect. Targeted attacks are more surgical, aiming to change the model's behavior on a specific narrow input or input class while leaving overall accuracy on other inputs largely unaffected, making the attack much harder to detect through aggregate accuracy monitoring alone. Backdoor attacks are a particularly stealthy targeted variant: the poisoned model behaves normally on all typical inputs but produces an attacker-chosen output whenever a specific trigger pattern (an unusual pixel pattern in an image, a specific phrase in text) is present, remaining dormant and undetected until that trigger appears.

```text
Backdoor example:
Normal image of a stop sign         -> classified correctly as "stop sign"
Stop sign + small sticker (trigger) -> misclassified as "speed limit sign"
```

## Poisoning Risks Specific to LLMs

Large language models trained or fine-tuned on web-scraped or crowdsourced data are vulnerable to poisoning through content specifically crafted to be scraped into training sets — text designed to bias the model toward particular viewpoints, insert backdoor trigger phrases, or manipulate how the model responds to specific future queries once that content is absorbed during training or fine-tuning. Retrieval-augmented generation systems face an analogous risk at inference time: if the retrieval corpus can be manipulated by an outside party, poisoned documents can bias or corrupt individual generated responses without touching model weights at all.

## Practical Guidance

For any pipeline training or fine-tuning on external or user-contributed data, apply data provenance tracking, anomaly detection on the training set (statistical outlier detection, and for supervised data, spot-checking label quality on samples flagged as unusual), and if feasible, hold out a clean, trusted validation set to detect suspicious accuracy patterns on specific input categories rather than relying on aggregate accuracy alone. For RAG systems, apply the same content-provenance scrutiny to the retrieval corpus that you would to training data, since a compromised knowledge base is functionally a poisoning attack even without touching the model itself.
