---
title: LLM Uncertainty and Calibration
description: How to measure, interpret, and improve the confidence of large language models — covering calibration theory, uncertainty quantification methods, hallucination detection, and practical strategies for trustworthy AI outputs.
---

**Uncertainty quantification** in large language models addresses a critical question: when a model produces an answer, how confident should we be that it is correct? Unlike traditional classifiers that output explicit probability distributions, LLMs generate free-form text — making uncertainty measurement both more important and more challenging.

A well-calibrated model is one where its expressed confidence accurately reflects its actual accuracy. A model that says "I'm 90% sure" should be right 90% of the time across all such statements.

## Why Calibration Matters

Overconfident LLMs are a significant source of harm in high-stakes applications:

- A medical LLM that confidently states an incorrect drug dosage.
- A legal assistant that fabricates case citations with complete certainty.
- A code assistant that generates subtly incorrect logic presented as definitively correct.

Conversely, an underconfident model hedges on every response, reducing utility even when it knows the answer.

## Types of Uncertainty

### Aleatoric Uncertainty

**Aleatoric uncertainty** is inherent in the task itself — the data does not contain enough information to determine a unique correct answer. For an LLM, this arises in:

- Ambiguous questions with multiple valid interpretations.
- Tasks requiring information the model was never trained on.
- Questions about the future or inherently probabilistic domains.

This type of uncertainty cannot be reduced by giving the model more parameters or training data.

### Epistemic Uncertainty

**Epistemic uncertainty** arises from gaps in the model's knowledge or training — it reflects what the model doesn't know but could, in principle, learn. This includes:

- Facts outside the training cutoff.
- Rare or specialized knowledge underrepresented in training data.
- Edge cases in reasoning that weren't well-covered in pretraining.

Epistemic uncertainty can, in theory, be reduced with more or better training data.

## Measuring LLM Confidence

### Token-Level Probabilities

The most direct signal is the **log-probability** of the generated tokens. For a response token sequence $y_1, y_2, \ldots, y_T$:

$$\text{confidence} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)\right)$$

This is the geometric mean of per-token probabilities — equivalent to the exponentiated average log-probability (or equivalently, the per-token perplexity inverted).

**Limitations**: Token probabilities are available via API in some providers (OpenAI `logprobs`, Hugging Face) but not all. High token probability doesn't guarantee factual correctness.

### Sampling-Based Uncertainty (Self-Consistency)

Generate $N$ responses with non-zero temperature and measure **agreement**:

- High agreement across samples → lower uncertainty.
- High variance across samples → higher uncertainty.

For extractive tasks (e.g., factual QA), measure exact-match agreement. For open-ended tasks, use semantic similarity (embedding cosine similarity) to cluster responses.

### Verbalized Confidence

Prompt the model to explicitly state its confidence:

```text
Answer the following question, then on the last line write your confidence
as a percentage from 0–100%.

Question: What year was the Eiffel Tower completed?
```

**Known issue**: LLMs are often poorly calibrated when asked to verbalize confidence. They tend to express high confidence even when incorrect. This can be partially addressed with calibration fine-tuning.

### Semantic Entropy

Kuhn et al. (2023) introduced **semantic entropy** — grouping semantically equivalent responses together before computing entropy. Standard token-level entropy conflates paraphrases with genuine disagreement. Semantic entropy correlates better with factual accuracy than raw token entropy.

## Calibration Evaluation

### Expected Calibration Error (ECE)

ECE measures the average gap between confidence and accuracy across confidence bins:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where $B_m$ is a bin of predictions grouped by confidence level, $\text{acc}(B_m)$ is the accuracy in that bin, and $\text{conf}(B_m)$ is the average confidence.

A perfectly calibrated model has ECE = 0.

### Reliability Diagrams

A **reliability diagram** plots confidence (x-axis) against accuracy (y-axis). A perfectly calibrated model lies on the diagonal. LLMs commonly show **overconfidence** (accuracy below the diagonal at high confidence levels).

## Improving Calibration

### Temperature Scaling

**Temperature scaling** is a simple post-hoc calibration method: divide the logits by a learned temperature parameter $T$ before softmax. When $T > 1$, the distribution is flattened (more uncertain); when $T < 1$, it sharpens. A single scalar $T$ is learned on a held-out calibration set.

### Calibration Fine-Tuning

Train the model on examples where the correct behavior is to express calibrated uncertainty. Reward models in RLHF pipelines can be extended to penalize overconfident wrong answers and reward appropriate hedging.

### Retrieval-Augmented Confidence

For factual questions, grounding the model in retrieved documents reduces epistemic uncertainty. When the model generates an answer that contradicts the retrieved sources, it signals lower confidence.

### Chain-of-Thought for Uncertainty

Prompting models to reason through uncertainty explicitly often improves calibration:

```text
Before answering, reason about what you know and don't know about this topic.
Note any aspects where you're uncertain, then give your best answer.
```

## Hallucination Detection via Uncertainty

Uncertainty signals are directly applicable to hallucination detection:

- **Factual grounding check**: Compare generated claims against retrieved evidence; ungrounded claims have higher uncertainty.
- **Cross-model agreement**: If multiple independent models disagree on a factual claim, the claim is likely unreliable.
- **Consistency probing**: Rephrase the question multiple ways; inconsistent answers across phrasings indicate uncertain knowledge.
- **Entropy thresholding**: Flag responses where per-token entropy spikes mid-sentence — often corresponds to made-up proper nouns or statistics.

## Practical Deployment Strategies

- **Confidence-gated workflows**: Route low-confidence responses to human review or fallback strategies.
- **Abstention**: Train models to say "I don't know" rather than fabricate answers. Selective prediction — answering only when confident — improves precision at the cost of coverage.
- **Uncertainty communication**: Surface uncertainty to users through hedged language ("I believe...", "This may be outdated...") or explicit confidence indicators in the UI.
- **Multi-source verification**: For critical outputs, automatically verify against knowledge bases, search results, or domain-specific APIs before presenting to users.

## Current Research Directions

- **Probing internal representations**: Research shows that LLM hidden states encode "truthfulness signals" that can be extracted with linear probes — potentially enabling uncertainty estimation without multiple forward passes.
- **Model-based uncertainty**: Training dedicated uncertainty models that take an LLM output and predict its reliability.
- **Bayesian LLMs**: Incorporating Bayesian inference into transformer training to produce principled posterior distributions over outputs.
- **Long-form calibration**: Most calibration research focuses on short factual QA; calibrating long-form generation (essays, code, reasoning chains) remains an open problem.

Reliable uncertainty quantification is foundational to **trustworthy AI** — without it, deploying LLMs in high-stakes domains requires accepting unknown and potentially large error rates.
