---
title: Introduction to Argilla
description: An overview of Argilla — an open-source data annotation and curation platform for LLMs — covering feedback datasets, human-in-the-loop labeling, and integration with training pipelines.
---

Argilla is an open-source platform for collecting, curating, and annotating data for natural language processing and large language model workflows. It is designed to make human feedback collection a structured, trackable part of the ML pipeline rather than an ad hoc spreadsheet exercise.

## Core Concepts

### Datasets
An Argilla **dataset** defines a schema of fields (the content to review, such as a prompt-response pair) and questions (what annotators should provide, such as a rating, a label, or free-text feedback).

### Records
Each **record** is one item to annotate — for example, a single LLM response paired with its prompt — plus any existing metadata like model name or generation parameters.

### Responses
A **response** is an annotator's answer to a record's questions. Argilla tracks responses per-user, which supports both single-annotator workflows and multi-annotator agreement analysis.

## Getting Started

### Install and Run

```bash
pip install argilla
docker run -d --name argilla -p 6900:6900 argilla/argilla-quickstart
```

### Python Client

```python
import argilla as rg

client = rg.Argilla(api_url="http://localhost:6900", api_key="argilla.apikey")

settings = rg.Settings(
    fields=[rg.TextField(name="prompt"), rg.TextField(name="response")],
    questions=[
        rg.RatingQuestion(name="quality", values=[1, 2, 3, 4, 5]),
        rg.TextQuestion(name="feedback", required=False),
    ],
)

dataset = rg.Dataset(name="llm-response-review", settings=settings)
dataset.create()

dataset.records.log([
    {"prompt": "Explain gradient descent.", "response": "Gradient descent is..."},
])
```

## Human-in-the-Loop Review Workflows

Argilla's primary use case is structuring the review of model outputs by human annotators. Teams typically log a batch of generated responses, assign them to reviewers through the web UI, and collect ratings or corrections that feed back into evaluation or fine-tuning datasets.

```text
model generates responses -> logged to Argilla -> human review/rating -> curated dataset -> fine-tuning or eval
```

## Weak Supervision and Active Learning

Argilla supports semi-automated labeling through rule-based weak supervision and active learning loops, where a model's uncertain predictions are prioritized for human review — concentrating annotator effort on the examples that will most improve the dataset, rather than reviewing everything uniformly.

## Feedback Datasets for RLHF and DPO

Because Argilla natively supports comparison-style questions (ranking or preferring one of several responses), it is commonly used to build preference datasets for RLHF or DPO-style fine-tuning, where each record presents multiple candidate responses and an annotator selects the preferred one.

## Argilla vs. Other Annotation Tools

| Feature | Argilla | Label Studio | Scale AI |
|---------|---------|--------------|----------|
| Open-source | ✓ | ✓ | ✗ |
| LLM-specific question types | ✓ | Partial | ✓ |
| Self-hostable | ✓ | ✓ | ✗ |
| Active learning integration | ✓ | Partial | Managed |
| Preference/ranking datasets | ✓ | Partial | ✓ |

## Common Use Cases

- **Response quality review:** collecting human ratings on generated text before it's used for fine-tuning.
- **Preference data collection:** building RLHF/DPO datasets by comparing candidate responses.
- **Dataset curation:** filtering and correcting noisy or synthetic datasets before training.
- **Evaluation set construction:** curating a high-quality, human-verified benchmark set from model outputs.

Argilla's focus on structured, trackable human feedback makes it a natural fit for teams building fine-tuning or evaluation datasets that require reliable human judgment rather than fully automated labeling.
