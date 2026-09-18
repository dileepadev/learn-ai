---
title: Scene Graph Generation - Structuring Objects and Their Relationships
description: Learn how scene graph generation extracts objects and the relationships between them from an image, and where this structure gets used.
---

Scene graph generation extracts a structured graph from an image: nodes are detected objects, and edges are labeled relationships between them, such as "riding," "holding," or "on top of."

```text
Image: a person riding a bicycle next to a parked car
Nodes: person, bicycle, car
Edges: (person, riding, bicycle), (bicycle, next to, car)
```

## Pipeline

A typical scene graph model first runs object detection to localize and classify candidate objects, then for each pair of detected objects, predicts whether a relationship exists and, if so, which relationship class applies. The relationship classifier usually conditions on both objects' visual features and their spatial arrangement, since relationships like "on top of" and "next to" strongly correlate with relative position.

## Why Structure Beyond Detection Matters

Object detection alone tells you what's present but not how objects relate — a scene with a person and a horse looks structurally different depending on whether the relationship is "riding," "next to," or "feeding." This structured relational information supports tasks that plain object lists cannot: complex image retrieval ("find images where a person is riding an animal"), detailed captioning grounded in explicit relationships, and downstream reasoning tasks like visual question answering that hinge on interactions between objects rather than their mere presence.

## Challenges

The space of possible object pairs grows quadratically with the number of detected objects, so scene graph models must be efficient about which pairs to even consider for relationship classification. Relationship classes also follow a highly imbalanced long-tail distribution — common predicates like "on" and "has" dominate training data, while informative but rarer predicates like "feeding" or "chasing" are underrepresented, making models biased toward generic, less useful relationship predictions.

## Practical Guidance

If your application only needs coarse relational cues, prompting a general vision-language model directly for a structured description often suffices without the engineering overhead of a dedicated scene graph pipeline. Build a dedicated scene graph model when you need a formal, queryable graph structure at scale — for example, indexing a large image or video corpus for structured relational search.
