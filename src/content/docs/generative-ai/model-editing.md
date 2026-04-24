---
title: Model Editing for Large Language Models
description: Explore how model editing techniques — including ROME, MEMIT, WISE, and gradient-based approaches — allow targeted updates to LLM factual knowledge without full retraining, enabling efficient knowledge correction and updates.
---

**Model editing** is the problem of making targeted, precise updates to the knowledge encoded in a large language model's parameters — changing specific factual associations without degrading the model's overall performance on unrelated tasks. As LLMs encode vast amounts of world knowledge in their weights, the ability to correct errors, update stale facts, and insert new knowledge without expensive retraining is a critical practical capability.

Consider a model that incorrectly believes a particular CEO leads a company that has since had a leadership change, or that encodes a subtle factual error about a scientific concept. Full retraining is prohibitively expensive; standard fine-tuning often corrupts unrelated capabilities or overfits. Model editing provides surgical, targeted updates that modify exactly the facts you want while leaving everything else intact.

## The Model Editing Problem

Formally, model editing aims to update a model $f_\theta$ to a model $f_{\theta'}$ such that:

**Efficacy**: The edited model correctly answers the target fact: $f_{\theta'}(x_e) = y_e$ (e.g., "Who leads OpenAI?" → "Sam Altman").

**Generality**: The edit generalizes to paraphrases and related formulations: $f_{\theta'}(x_e') = y_e$ for semantically equivalent questions.

**Locality**: Unrelated facts are unchanged: $f_{\theta'}(x) = f_\theta(x)$ for inputs $x$ unrelated to the edit.

**Consistency**: The model's related beliefs update consistently: if we edit "OpenAI's CEO is Sam Altman," the model should also update "Who manages OpenAI?" consistently.

These properties jointly define a successful edit — and achieving all four simultaneously is the core technical challenge.

## Rank-One Model Editing (ROME)

**ROME** (Meng et al., 2022) is the seminal model editing method, providing both a theoretical framework and a practical algorithm for editing factual associations in GPT-style LLMs.

### The Factual Association Hypothesis

ROME is grounded in the observation that **MLP layers in transformers function as key-value stores** — the first (key) layer maps inputs to intermediate representations; the second (value) layer maps these to output contributions. Factual associations are stored as (key, value) pairs: a specific subject activates a key, which retrieves the associated factual attribute as its value.

Causal mediation analysis (ablating hidden states and observing the effect on output probabilities) confirms that specific MLP layers — mid-layer MLP modules in GPT-style models — causally mediate the storage and retrieval of factual associations.

### The ROME Algorithm

To edit the association $(s, r, o) \to (s, r, o^*)$ (subject, relation, old object → new object):

1. **Identify the critical layer** $l^*$ via causal tracing — the layer where the fact is stored.

2. **Compute the edit direction**: Find the target value vector $v^*$ such that the model, with the MLP value matrix at layer $l^*$ modified to output $v^*$ for subject $s$, produces the correct output $o^*$.

3. **Update the MLP weight matrix**: Use a rank-one update to the value matrix $W_{\text{out}}^{l^*}$:

$$\hat{W} = W + \Delta, \quad \Delta = \frac{(v^* - W k^*)k^{*\top}}{k^{*\top} k^*}$$

where $k^*$ is the key representation for the subject $s$ at layer $l^*$. This rank-one update stores the new (key, value) pair in the MLP without affecting other keys.

1. **Constrain to the pre-edit key distribution**: To preserve model behavior on unrelated inputs, the update is projected to minimize disruption to existing key-value associations using a covariance constraint.

ROME achieves high efficacy and locality for single edits but degrades when applied sequentially for many edits.

## MEMIT: Mass-Editing Memory in a Transformer

**MEMIT** (Meng et al., 2022) extends ROME to support thousands of simultaneous edits — a critical requirement for practical knowledge base updating.

MEMIT distributes the editing error across multiple layers (rather than a single layer as in ROME) and computes a batch update to all target layers simultaneously. The multi-layer distribution reduces the perturbation per layer, preserving locality across many concurrent edits.

```python
from memit import MEMITHyperParams, apply_memit_to_model

# Define a batch of edits: (subject, relation, new_object)
edits = [
    {"prompt": "The CEO of {}", "subject": "OpenAI", "target_new": "Sam Altman"},
    {"prompt": "{} was founded in", "subject": "Anthropic", "target_new": "2021"},
    {"prompt": "The capital of {} is", "subject": "Germany", "target_new": "Berlin"},
    # ... up to thousands of edits
]

hparams = MEMITHyperParams.from_name("gpt2-xl")
model, tokenizer = get_model_and_tokenizer("gpt2-xl")

edited_model, edit_info = apply_memit_to_model(
    model, tokenizer, edits, hparams
)
```

MEMIT maintains efficacy and locality across 10,000+ simultaneous edits — enabling batch knowledge base updates that are impractical with single-edit methods.

## WISE: Retrieval-Augmented Editing

**WISE** (Wang et al., 2023) takes a fundamentally different approach: rather than modifying model weights, it maintains a **side memory** of edited facts and retrieves them at inference time:

1. Edits are stored in an external memory with keys derived from the edit subject's representation.
2. At inference time, the model checks whether the current input activates any stored edit (using similarity search).
3. If a relevant edit is found, it is injected into the model's processing via a small learned adapter.
4. If no relevant edit is found, the model runs normally on its original weights.

WISE avoids the locality challenges of weight-modification approaches — since the original weights are unchanged, unrelated facts are never affected. The tradeoff is inference overhead from the retrieval step and the memory requirement for storing edits explicitly.

## Gradient-Based Fine-Tuning Approaches

### Constrained Fine-Tuning

Standard fine-tuning on a single (input, new output) pair updates far more parameters than necessary, causing **catastrophic forgetting** of nearby facts. Constrained fine-tuning approaches address this:

- **EWC-based editing**: Apply Elastic Weight Consolidation to preserve important weights while updating for the target fact.
- **Hypernetwork editing (MEND)**: Train a hypernetwork that predicts the optimal weight update from the gradient of the cross-entropy loss on the edit example — transforming the raw gradient into a targeted weight update that preserves locality.

### MEND: Model Editor Networks with Gradient Decomposition

**MEND** (Mitchell et al., 2021) trains a hypernetwork offline to transform raw gradient updates into targeted edits:

1. For each candidate edit $(x_e, y_e)$, compute the standard fine-tuning gradient $\nabla_\theta \mathcal{L}(f_\theta(x_e), y_e)$.
2. The MEND hypernetwork $g_\phi$ transforms this gradient into a better update $\Delta\theta = g_\phi(\nabla_\theta \mathcal{L})$ that achieves the edit while preserving locality.
3. $g_\phi$ is trained on a dataset of (edit, desired update) pairs.

Once trained, MEND can be applied to new edits at inference time — fast enough for interactive editing. The hypernetwork learns a decomposed representation of gradients that separates edit-relevant dimensions from irrelevant ones.

## Evaluation Benchmarks

### COUNTERFACT

**COUNTERFACT** (Meng et al., 2022) tests whether models correctly update factual associations that contradict their pretraining knowledge. It measures:

- **Efficacy score (ES)**: P(model predicts new target | edited prompt).
- **Paraphrase score (PS)**: ES on paraphrased versions of the edit prompt.
- **Neighborhood score (NS)**: Model predictions on neighboring (unrelated) facts are unchanged.

### ZsRE

**Zero-Shot Relation Extraction (ZsRE)** provides a large-scale benchmark for evaluating generalization of edits to paraphrased and semantically equivalent formulations.

### RippleEdit

**RippleEdit** tests whether edits propagate consistently to logically related facts — the "ripple effects" of an edit. If we change the CEO of a company, the model should also update answers about who manages the company's products.

## Lifelong Editing and Sequential Updates

Real-world deployment requires not single edits but **streams of edits** — continually updating the model as the world changes. Sequential editing faces:

- **Edit interference**: Later edits can inadvertently overwrite earlier edits.
- **Accumulated perturbation**: Each weight-modification edit slightly perturbs the model; thousands of sequential edits may degrade overall capability.

**T-Patcher** and **CaliNet** address sequential editing by storing edits in additional parameters (patches or calibration layers) rather than modifying existing weights — each new edit adds a small module without affecting previous edits or the base model.

## Limitations and Open Problems

**Consistency**: Ensuring that editing one fact triggers all logically downstream facts to update consistently (ripple effects) remains largely unsolved. Models may believe "X's CEO is Y" while still answering "who manages X's products?" with the old CEO.

**Scalability**: Most weight-editing methods degrade with the number of edits. Beyond a few thousand edits, even MEMIT shows locality degradation.

**LLM scale**: Methods validated on GPT-2 XL (1.5B) do not always transfer cleanly to frontier-scale models (70B+).

**Generalization vs. Locality tension**: Methods that generalize broadly (updating all phrasings of a fact) may inadvertently affect related facts; methods with tight locality may under-generalize to paraphrases.

Model editing remains an active and open research area, with the long-term vision of LLMs that can be continuously and reliably updated — behaving like living knowledge bases rather than static snapshots of the world at training time.
