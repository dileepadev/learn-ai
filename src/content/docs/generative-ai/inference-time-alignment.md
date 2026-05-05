---
title: Inference-Time Alignment
description: A comprehensive guide to inference-time alignment techniques for language models, covering Best-of-N sampling, RLHF decoding, contrastive decoding, guided generation, and Constitutional AI at inference time.
---

# Inference-Time Alignment

Inference-time alignment refers to techniques that steer language model outputs toward desired behaviors **without modifying model weights**. Rather than fine-tuning the model to be helpful, harmless, and honest, these methods operate at decoding time — reshaping the distribution of generated tokens using reward signals, contrastive references, or structured constraints. They offer flexibility, lower compute cost, and the ability to adapt a single base model to multiple alignment objectives on the fly.

## Motivation

Standard alignment pipelines (RLHF, DPO) are expensive: they require labeled preference data, gradient updates, and risk catastrophic forgetting of base capabilities. Inference-time alignment offers complementary benefits:

- **No training required**: apply to any model via API or local inference
- **Composability**: combine multiple alignment signals simultaneously
- **Adaptability**: change alignment objective without retraining
- **Interpretability**: alignment decisions can be made explicit and auditable

The tradeoff is inference cost — most techniques require multiple forward passes or auxiliary models.

## Best-of-N Sampling (BoN / Rejection Sampling)

The simplest inference-time alignment strategy: generate $N$ candidate responses and return the one with highest reward model score.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def best_of_n(
    model,
    tokenizer,
    reward_model_pipe,
    prompt: str,
    n: int = 8,
    max_new_tokens: int = 256,
) -> str:
    # Generate N candidates
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=n,
    )
    candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Score with reward model
    full_texts = [f"{prompt} {c}" for c in candidates]
    scores = [r["score"] for r in reward_model_pipe(full_texts, truncation=True)]

    return candidates[scores.index(max(scores))]
```

**Efficiency**: BoN provides alignment quality proportional to $\log N$ — doubling $N$ yields diminishing returns. Effective up to $N \approx 64$; beyond that, reward-guided decoding becomes more efficient.

**KL from reference**: BoN implicitly stays close to the base model since candidates are drawn from the original distribution. The effective KL penalty is approximately $\log N$.

## Reward-Guided Decoding (ARGS / GDC)

Instead of scoring complete responses, reward-guided decoding applies the reward signal **token by token**, reshaping the logit distribution at each step.

### ARGS — Augmented Reward Guidance Sampling

```python
import torch.nn.functional as F


def reward_guided_decode(
    model,
    reward_model,
    tokenizer,
    prompt: str,
    alpha: float = 0.5,
    max_new_tokens: int = 256,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated).logits[:, -1, :]  # (1, vocab)

        # Sample top-k candidates for reward lookahead
        k = 10
        top_logits, top_ids = logits.topk(k, dim=-1)
        
        # Estimate reward delta for each candidate next token
        reward_scores = []
        for i in range(k):
            candidate = torch.cat([generated, top_ids[:, i:i+1]], dim=1)
            partial_text = tokenizer.decode(candidate[0])
            r = reward_model(partial_text)
            reward_scores.append(r)

        reward_tensor = torch.tensor(reward_scores, dtype=torch.float)
        reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)

        # Combine LM logits with reward signal
        adjusted = top_logits.squeeze() + alpha * reward_tensor
        probs = F.softmax(adjusted, dim=-1)
        chosen_idx = torch.multinomial(probs, 1)
        next_token = top_ids[0, chosen_idx]

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)
```

### GDC — Generative Distributional Control

GDC frames alignment as a constraint satisfaction problem. Given a target distribution $p^*(x) \propto p(x) \cdot r(x)^\beta$, sequential Monte Carlo methods approximate this distribution at generation time, producing outputs that trade off fluency against reward.

## Contrastive Decoding

Contrastive Decoding (Li et al., 2022) subtracts the log-probabilities of a weaker "amateur" model from those of the target model:

$$\log p_{\text{contrastive}}(x_t \mid x_{<t}) = \log p_{\text{expert}}(x_t \mid x_{<t}) - \log p_{\text{amateur}}(x_t \mid x_{<t})$$

This amplifies tokens that the expert assigns relatively more probability to compared to the amateur — suppressing generic, low-information outputs.

```python
def contrastive_decode_step(
    expert_logits: torch.Tensor,
    amateur_logits: torch.Tensor,
    alpha: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    # Plausibility filter: only consider tokens above threshold in expert
    threshold = torch.log(torch.tensor(alpha)) + expert_logits.max()
    mask = expert_logits >= threshold

    cd_logits = expert_logits - amateur_logits
    cd_logits[~mask] = float("-inf")
    return (cd_logits / temperature).softmax(dim=-1)
```

**Alignment application**: use a model fine-tuned on harmful content as the "amateur" — contrastive decoding then amplifies the helpful, harmless behaviors the expert assigns more probability to.

## CAI at Inference: Constitutional Prompting

**Constitutional AI at inference time** applies the critique-revision loop without any fine-tuning:

```python
CONSTITUTION = [
    "The response should be helpful, harmless, and honest.",
    "The response should not assist with illegal activities.",
    "The response should be respectful of all people.",
]

def constitutional_inference(model_pipe, prompt: str, num_revisions: int = 2) -> str:
    response = model_pipe(prompt, max_new_tokens=512)[0]["generated_text"]

    for _ in range(num_revisions):
        # Critique step
        critique_prompt = (
            f"Here is a conversation:\nHuman: {prompt}\nAssistant: {response}\n\n"
            f"Please critique the assistant's response according to these principles:\n"
            + "\n".join(f"- {p}" for p in CONSTITUTION)
            + "\n\nCritique:"
        )
        critique = model_pipe(critique_prompt, max_new_tokens=256)[0]["generated_text"]

        # Revision step
        revision_prompt = (
            f"Human: {prompt}\nAssistant: {response}\n\n"
            f"Critique: {critique}\n\n"
            "Please revise the response to address the critique:\nRevised response:"
        )
        response = model_pipe(revision_prompt, max_new_tokens=512)[0]["generated_text"]

    return response
```

Multiple revision rounds progressively refine the output, with diminishing returns after 2–3 iterations.

## Activation Steering

Activation steering (representation engineering) identifies directions in the residual stream associated with target behaviors and adds them at inference:

```python
def steer_generation(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,   # (d_model,) behavioral direction
    layer_idx: int = 15,
    alpha: float = 20.0,
    max_new_tokens: int = 256,
) -> str:
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        hidden[:, :, :] += alpha * steering_vector.to(hidden.device)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    layer = model.model.layers[layer_idx]
    hooks.append(layer.register_forward_hook(hook_fn))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    for h in hooks:
        h.remove()

    return tokenizer.decode(output[0], skip_special_tokens=True)
```

Steering vectors are derived by contrasting activations on positive vs. negative concept examples (e.g., helpful vs. harmful responses) using linear probes or difference-in-means.

## Speculative Rejection

A compute-efficient variant of BoN that uses a fast draft model to generate candidates and a slow oracle model to score them — accepting only high-reward drafts:

1. Draft model generates $N$ candidate continuations quickly
2. Reward model scores each candidate
3. Accept the top candidate; reject others
4. If no candidate exceeds threshold, the oracle model regenerates

This provides alignment quality close to large-BoN at a fraction of the cost.

## Comparison of Inference-Time Methods

| Method | Reward Signal | Tokens/Second | Quality | Interpretability |
|---|---|---|---|---|
| Best-of-N | Sequence-level | Slow (N× passes) | Good | High |
| Reward-guided | Token-level | Very slow | Better | Low |
| Contrastive | Implicit (model diff) | Fast (2× passes) | Good | Moderate |
| CAI prompting | Self-critique | Slow (multi-turn) | Good | High |
| Activation steering | Direction in activations | Fast (1× pass) | Variable | Moderate |
| Speculative rejection | Sequence-level (draft+oracle) | Moderate | Good | Moderate |

## When to Use Inference-Time Alignment

- **Rapid prototyping**: test alignment interventions before committing to fine-tuning
- **API-only access**: apply alignment to models available only via API (no weight access)
- **Multi-objective**: different users or contexts need different alignment profiles from one model
- **High-stakes one-shot tasks**: apply BoN or reward-guided decoding for critical outputs where inference cost is acceptable
- **Research**: isolating the contribution of alignment independent of training distribution shifts

## Limitations

- **Reward model bias**: inference-time methods are bounded by reward model quality — they cannot exceed what the RM can detect
- **Compute cost**: multi-pass methods multiply inference cost by $N$
- **Distribution mismatch**: reward-guided decoding can produce low-fluency outputs when the reward signal conflicts with base model probabilities
- **Jailbreak vulnerability**: inference-time alignment without fine-tuning provides weaker safety guarantees than RLHF-trained models

## Summary

Inference-time alignment offers a flexible, training-free complement to weight-level alignment techniques. Best-of-N sampling, reward-guided decoding, contrastive decoding, and activation steering each occupy different points on the cost-quality-interpretability tradeoff. As models become more capable and reward models more accurate, inference-time alignment is increasingly viable for production deployment — particularly when adaptability across diverse alignment objectives matters more than minimizing per-token inference cost.
