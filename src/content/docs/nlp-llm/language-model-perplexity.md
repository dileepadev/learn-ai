---
title: Language Model Perplexity - Measuring How Surprised a Model Is
description: Understand what perplexity measures, how it is computed, and why a lower score does not always mean a better model for your task.
---

Perplexity is the standard intrinsic metric for language models: it measures how well a probability distribution predicts a sample, expressed as the exponentiated average negative log-likelihood per token.

```text
PPL = exp( - (1/N) * sum(log P(token_i | context)) )
```

A perplexity of 20 means the model is, on average, as uncertain as if it had to choose uniformly among 20 equally likely next tokens at each step. Lower perplexity means the model assigns higher probability to the actual next tokens in the evaluation text.

## Why It's Useful

Perplexity requires no labeled data beyond the text itself, making it cheap to compute on any held-out corpus. It is useful for comparing checkpoints during pretraining, detecting distribution shift (a model trained on news text will show much higher perplexity on legal contracts), and sanity-checking that training is converging.

## Why It Can Mislead

Perplexity is tokenizer-dependent — you cannot directly compare perplexity scores across models with different vocabularies or tokenization schemes, since a model with a larger effective vocabulary per token will report different numbers for equivalent quality. Perplexity also does not correlate cleanly with downstream task performance or output quality as judged by humans; a model can achieve excellent perplexity on generic web text while performing poorly on instruction-following or reasoning tasks, because perplexity rewards matching the statistics of the training distribution rather than usefulness for a specific task.

## Practical Guidance

Use perplexity to track pretraining progress and detect domain mismatch, but validate any claim about model quality with task-specific benchmarks or human evaluation before relying on perplexity alone. When comparing two models, only compare perplexity if they share the same tokenizer and were evaluated on the exact same held-out text.
