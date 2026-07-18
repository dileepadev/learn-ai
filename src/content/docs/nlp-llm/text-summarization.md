---
title: Text Summarization - Condensing Documents Without Losing the Point
description: Learn extractive and abstractive summarization, how to evaluate factuality, and how to build safer document summaries.
---

Text summarization produces a shorter version of a document while preserving the information a reader needs. It can extract existing sentences or generate new wording.

## Extractive and Abstractive Methods

**Extractive summarization** ranks and selects source sentences. It is easier to trace to the source but can be repetitive or awkward.

**Abstractive summarization** generates a new summary with an encoder-decoder model or LLM. It can synthesize material across a document but may introduce unsupported claims.

```text
source -> select or encode key content -> concise summary
```

## Defining the Job

“Summarize this” is underspecified. Good requirements state the intended reader, length, format, and what must be retained:

- executive briefing with decisions and risks
- clinical handoff with medications and uncertainty
- meeting notes with owners and deadlines
- customer-facing plain-language explanation

## Evaluation

ROUGE measures overlap with reference summaries, but high overlap does not prove a summary is complete or factual. Evaluate:

| Dimension | Question |
| --- | --- |
| Faithfulness | Is every claim supported by the source? |
| Coverage | Are the important points present? |
| Relevance | Does it fit the requested audience and purpose? |
| Concision | Is unnecessary detail removed? |

Human review and claim-level checks are especially important for long documents and high-impact content.

## Safer Generation

Provide the source or retrieved citations in the prompt, require the model to distinguish facts from uncertainty, and ask it to say when the source does not answer a question. For long inputs, use hierarchical summaries and retain links to the original sections. A fluent summary is not necessarily a faithful one.

