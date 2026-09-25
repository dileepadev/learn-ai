---
title: Introduction to the Gemini API
description: Learn how Google's Gemini API handles native multimodal input, long context windows, and grounding with Google Search.
---

The Gemini API gives programmatic access to Google's Gemini family of models, built from the ground up as natively multimodal — a single model architecture handles text, images, audio, and video without separate specialized models bolted together.

## Basic Usage

```python
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=["Describe what's happening in this image.", image_bytes]
)
```

Because Gemini treats different modalities as tokens in the same sequence rather than routing them through separate encoders bolted onto a text model, a single request can mix text, images, audio clips, and video frames as context for one generation, without needing separate calls to a vision model and a language model.

## Long Context Windows

Gemini models support context windows large enough to fit entire codebases, hours of video, or hundreds of pages of documents in a single request, shifting some use cases that previously required a RAG pipeline (chunk, embed, retrieve) toward simply including the full source material directly in context. This tradeoff favors simplicity and completeness over the cost and latency efficiency that a well-tuned RAG pipeline provides at scale.

## Grounding with Google Search

The API supports a grounding tool that lets the model issue live Google Search queries during generation and cite sources in its response, addressing hallucination and knowledge-cutoff limitations for queries about current events or rapidly changing information without requiring you to build a separate search-and-retrieve pipeline yourself.

## Practical Guidance

Use Gemini's native multimodal input when your application genuinely needs to reason jointly across modalities — a workflow that separately captions an image with one model, then feeds the caption to a text model, loses information that direct multimodal reasoning preserves. For extremely large context use cases, benchmark cost and latency against a RAG approach for your specific data size and query pattern, since "put everything in context" does not always outperform targeted retrieval on either cost or accuracy.
