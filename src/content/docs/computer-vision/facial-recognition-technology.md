---
title: Facial Recognition Technology - How It Works and Where It Fails
description: Understand the pipeline behind facial recognition systems, the accuracy gaps across demographics, and the governance concerns that follow.
---

Facial recognition identifies or verifies a person's identity from their face, powering everything from phone unlock to airport security to controversial surveillance deployments.

## The Pipeline

A typical facial recognition system runs several stages: face detection locates face regions in an image, face alignment normalizes pose and scale, an embedding model converts the aligned face into a fixed-length vector, and a matching step compares that vector against a database of known embeddings.

```text
image -> detect face -> align -> embed (128-512 dim vector) -> compare to database (cosine/L2 distance)
```

Verification (is this the same person as this reference photo?) compares two embeddings and checks if their distance is below a threshold. Identification (who is this person?) searches a database of embeddings for the closest match.

## Training the Embedding Model

Modern face embedding models train with a metric learning loss — triplet loss or ArcFace-style angular margin losses — that pulls embeddings of the same identity closer together and pushes different identities apart, rather than training as a standard classifier over a fixed set of identities that wouldn't generalize to new, unseen people.

## Accuracy Disparities and Governance Concerns

Facial recognition systems have documented accuracy gaps across demographic groups, historically performing worse on darker skin tones and on women, largely traced to unrepresentative training data. These disparities have led to wrongful identifications in law enforcement contexts and are a central reason multiple jurisdictions restrict or ban police use of facial recognition. Deployments also raise consent and surveillance concerns distinct from pure accuracy — a technically accurate system can still enable harmful mass surveillance if deployed without safeguards.

## Practical Guidance

Before deploying facial recognition, evaluate accuracy separately across demographic subgroups relevant to your deployment population rather than trusting a single aggregate accuracy number. Check applicable regulations (several US states and the EU AI Act impose specific restrictions on biometric identification) before building or deploying, and prefer verification (one-to-one matching, opt-in) over identification (one-to-many search) wherever the use case allows it, since verification carries materially lower surveillance risk.
