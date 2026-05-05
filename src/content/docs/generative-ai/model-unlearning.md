---
title: Machine Unlearning for LLMs
description: Explore machine unlearning — techniques that selectively remove specific knowledge, data, or behaviors from trained models without full retraining. Learn gradient ascent unlearning, SISA training, the ROME and MEMIT model editing approaches, forget-retain loss balancing, and evaluation frameworks for verifying what has and hasn't been unlearned.
---

**Machine unlearning** addresses a fundamental tension in modern AI: models are trained on vast datasets that may contain private data, copyrighted content, toxic information, or factual errors that their owners later want removed. Retraining from scratch is prohibitively expensive for large models. Machine unlearning seeks to efficiently scrub specific knowledge from a trained model while preserving everything else — making the model behave as if the target data was never seen.

This is becoming a legal necessity. The EU's General Data Protection Regulation (GDPR) enshrines the "right to erasure" — the right of individuals to request deletion of their personal data from systems that process it. For models that memorize training data, this may require removing that data's influence from the model weights.

## The Unlearning Problem Formally

Let $\mathcal{D} = \mathcal{D}_f \cup \mathcal{D}_r$ where:

- $\mathcal{D}_f$ — the **forget set**: data to be unlearned (individual's personal data, copyrighted text, harmful content)
- $\mathcal{D}_r$ — the **retain set**: everything else that should be preserved

The ideal outcome is a model $\theta_u$ that is indistinguishable from a model trained from scratch on $\mathcal{D}_r$ alone, but without the cost of retraining:

$$\theta_u \approx \underset{\theta}{\text{argmin}} \, \mathcal{L}(\theta; \mathcal{D}_r)$$

## Gradient Ascent Unlearning

The simplest unlearning approach: apply gradient **ascent** on the forget set (maximizing rather than minimizing the loss on $\mathcal{D}_f$), which reduces the model's ability to predict the forgotten data:

```python
import torch
from torch.utils.data import DataLoader

def gradient_ascent_unlearn(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    n_steps: int = 500,
    lr: float = 1e-5,
    alpha: float = 0.5,   # weight on retain loss
    device: str = "cuda"
) -> torch.nn.Module:
    """
    Gradient ascent unlearning with retain regularization.
    
    Naive gradient ascent (no retain term) causes catastrophic forgetting
    of unrelated knowledge — the model loses coherence rapidly.
    Adding the retain loss anchors the model to preserve general capabilities.
    
    Loss = -α_f × L(θ; D_f) + α_r × L(θ; D_r)
               ↑ ascent on forget    ↑ descent on retain
    """
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    retain_iter = iter(retain_loader)
    
    for step in range(n_steps):
        # ── Forget step: maximize loss on forget data ──────────────────────
        forget_batch = next(iter(forget_loader))
        input_ids = forget_batch["input_ids"].to(device)
        labels = forget_batch["labels"].to(device)
        
        forget_loss = model(input_ids=input_ids, labels=labels).loss
        
        # ── Retain step: minimize loss on retain data ──────────────────────
        try:
            retain_batch = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            retain_batch = next(retain_iter)
        
        retain_input = retain_batch["input_ids"].to(device)
        retain_labels = retain_batch["labels"].to(device)
        retain_loss = model(input_ids=retain_input, labels=retain_labels).loss
        
        # Combined: ascend on forget, descend on retain
        combined_loss = -forget_loss + alpha * retain_loss
        
        optimizer.zero_grad()
        combined_loss.backward()
        
        # Gradient clipping prevents instability from large negative gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}: forget_loss={forget_loss.item():.4f}, "
                  f"retain_loss={retain_loss.item():.4f}")
    
    return model
```

## Task Vector Negation

A cleaner approach for capability-level unlearning. **Task vectors** (Ilharco et al., 2022) represent the difference between a fine-tuned and base model in weight space. Subtracting a task vector removes the capability learned during fine-tuning:

```python
import copy

def compute_task_vector(
    base_model: dict[str, torch.Tensor],
    finetuned_model: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Task vector = fine-tuned weights - base model weights.
    Represents the knowledge added during fine-tuning as a direction in weight space.
    """
    return {
        k: finetuned_model[k] - base_model[k]
        for k in base_model
    }


def negate_task_vector(
    base_state_dict: dict[str, torch.Tensor],
    harmful_finetuned_state_dict: dict[str, torch.Tensor],
    scale: float = 0.5
) -> dict[str, torch.Tensor]:
    """
    Unlearn a capability by negating its task vector.
    
    If a model was fine-tuned on harmful content, this removes that
    fine-tuning's effect without access to the original harmful data.
    
    unlearned = base + (-scale) × task_vector
              = base - scale × (harmful - base)
    
    Applications:
    - Remove safety fine-tuning bypasses (re-alignment)
    - Undo domain adaptation that introduced bias
    - Remove knowledge of a specific individual or organization
    """
    task_vector = compute_task_vector(base_state_dict, harmful_finetuned_state_dict)
    
    unlearned_state_dict = {}
    for k in base_state_dict:
        unlearned_state_dict[k] = base_state_dict[k] - scale * task_vector[k]
    
    return unlearned_state_dict
```

## ROME and MEMIT: Surgical Weight Editing

**ROME** (Rank-One Model Editing, Meng et al., 2022) and **MEMIT** (Mass-Model Editing in Networks) make targeted edits to specific factual associations stored in MLP layers of transformer models. They identify which weights encode a specific fact and overwrite only those weights:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def rome_unlearn_fact(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    subject: str,
    target_fact: str,
    counterfactual: str,
    layer_idx: int = 17   # ROME targets middle layers (layer 17 in GPT-2-XL)
) -> AutoModelForCausalLM:
    """
    Simplified ROME-style weight edit to remove a specific factual association.
    
    ROME finds the MLP weight W that implements "subject → fact" and computes
    a rank-one update W' = W + Δ where Δ makes the model associate
    subject → counterfactual instead.
    
    Example:
        subject = "Albert Einstein"
        target_fact = "general relativity"   (what we want to forget)
        counterfactual = "[UNLEARNED]"        (what to output instead)
    
    Full ROME requires computing the covariance statistics of the hidden states
    (via C = E[k k^T] where k is the key vector for the subject token).
    This is a conceptual demonstration of the interface.
    """
    # Step 1: Find the key vector for the subject token at the target layer
    subject_tokens = tokenizer(subject, return_tensors="pt").input_ids
    
    with torch.no_grad():
        outputs = model(subject_tokens, output_hidden_states=True)
    
    # Hidden state at the subject's last token position, target layer
    hidden = outputs.hidden_states[layer_idx][:, -1, :]   # (1, d_model)
    
    # Step 2: Compute value vector for counterfactual
    # (in real ROME: solve for v using optimization; simplified here)
    counterfactual_tokens = tokenizer(counterfactual, return_tensors="pt").input_ids
    
    # Step 3: Rank-one update to MLP down-projection
    mlp_layer = model.transformer.h[layer_idx].mlp
    W = mlp_layer.c_proj.weight.data  # (d_model, d_ffn)
    
    # Note: actual ROME uses a closed-form solution derived from the
    # key-value memory interpretation of MLP layers (Geva et al., 2021)
    # This interface shows the structure; production use should use the
    # official ROME/MEMIT library: pip install rome-model-editing
    
    print(f"Would edit layer {layer_idx} to associate '{subject}' → '{counterfactual}'")
    return model


def batch_unlearn_with_memit(model, tokenizer, facts_to_forget: list[dict]) -> None:
    """
    MEMIT extends ROME to batch edits: edit hundreds of facts simultaneously
    by distributing updates across multiple MLP layers.
    
    Each fact: {"subject": ..., "relation": ..., "target": ..., "counterfactual": ...}
    
    Production usage (requires pip install rome):
    from rome import MEMITHyperParams, apply_memit_to_model
    hparams = MEMITHyperParams.from_hparams("hparams/MEMIT/gpt2-xl.json")
    model, _ = apply_memit_to_model(model, tokenizer, facts_to_forget, hparams)
    """
    print(f"MEMIT would unlearn {len(facts_to_forget)} facts across layers 13-17")
```

## Evaluating Unlearning

Good unlearning must be verified on multiple dimensions simultaneously:

```python
from transformers import pipeline

def evaluate_unlearning(
    original_model,
    unlearned_model,
    forget_prompts: list[str],
    retain_prompts: list[str],
    retrain_from_scratch_model = None
) -> dict[str, float]:
    """
    Comprehensive unlearning evaluation.
    
    Key metrics:
    1. Forget quality: does the model fail to recall forgotten content?
    2. Retain quality: is general capability preserved?
    3. Unlearning completeness: membership inference can't distinguish from retrained?
    4. No regurgitation: model doesn't produce verbatim forgotten text?
    """
    results = {}
    
    original_gen = pipeline("text-generation", model=original_model)
    unlearned_gen = pipeline("text-generation", model=unlearned_model)
    
    # ── Forget quality ─────────────────────────────────────────────────────
    # Lower is better: unlearned model should output low-quality/wrong answers
    forget_perplexities_original = []
    forget_perplexities_unlearned = []
    
    for prompt in forget_prompts:
        orig_output = original_gen(prompt, max_new_tokens=50)[0]["generated_text"]
        unlearned_output = unlearned_gen(prompt, max_new_tokens=50)[0]["generated_text"]
        
        # Measure: does the unlearned model still recall target facts?
        # (In practice: compute perplexity on known forget-set continuations)
        forget_perplexities_original.append(len(orig_output))  # proxy
        forget_perplexities_unlearned.append(len(unlearned_output))  # proxy
    
    results["forget_quality_ratio"] = (
        sum(forget_perplexities_unlearned) / (sum(forget_perplexities_original) + 1e-8)
    )
    
    # ── Retain quality ─────────────────────────────────────────────────────
    # Higher is better: general capability should be preserved
    retain_scores_original = []
    retain_scores_unlearned = []
    
    for prompt in retain_prompts:
        orig = original_gen(prompt, max_new_tokens=50, do_sample=False)[0]
        unlearned = unlearned_gen(prompt, max_new_tokens=50, do_sample=False)[0]
        
        # Cosine similarity of outputs (proxy for semantic preservation)
        retain_scores_original.append(orig["generated_text"])
        retain_scores_unlearned.append(unlearned["generated_text"])
    
    # Simple word overlap as proxy for retain quality
    def word_overlap(a: str, b: str) -> float:
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words:
            return 0.0
        return len(a_words & b_words) / len(a_words)
    
    results["retain_quality"] = sum(
        word_overlap(o, u) for o, u in
        zip(retain_scores_original, retain_scores_unlearned)
    ) / len(retain_prompts)
    
    return results
```

## SISA: Sharded-Isolated-Sliced-Aggregated Training

**SISA training** (Bourtoule et al., 2021) makes unlearning efficient at the training infrastructure level:

```
Standard training: one model trained on all data → full retrain to unlearn
SISA training:
  ├── Sharding: split dataset into k disjoint shards
  ├── Isolation: train k sub-models, each on one shard
  ├── Slicing: within each shard, train incrementally on slices
  └── Aggregation: ensemble sub-model predictions at inference

To unlearn a sample in shard i:
  → Retrain only the sub-model for shard i (from its last safe checkpoint)
  → O(1/k) retraining cost vs. full retrain
```

SISA is particularly suited to production ML systems where data deletion requests are known in advance or follow predictable patterns.

## The Unlearning Verification Challenge

The hardest part of machine unlearning is **proving** it worked. A model that claims to have forgotten may still:

- Reproduce exact training text when prompted cleverly (prompt injection exploiting residual memorization)
- Perform well on membership inference attacks that detect whether specific samples were in training
- Reconstruct forgotten content from closely related retained knowledge

Current best practice involves:

- **Membership inference attack** resistance: forgotten data should look like non-training data to the best available MIA classifier
- **Extraction attacks**: model should not produce verbatim forgotten text under any reasonable prompt variation
- **General capability benchmarks**: MMLU, HellaSwag, TruthfulQA scores should be within acceptable degradation thresholds

Machine unlearning remains an active research area, with no consensus on what "sufficient unlearning" means. The practical standard is evolving alongside regulatory requirements — the EU AI Act's provisions on training data transparency are expected to accelerate demand for robust, verifiable unlearning techniques.
