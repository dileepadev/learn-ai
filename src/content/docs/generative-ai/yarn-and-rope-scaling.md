---
title: YaRN and RoPE Scaling for Context Window Extension
description: Learn how position interpolation, NTK-aware scaling, and YaRN extend the context length of rotary-embedding models beyond their pretraining window.
---

Models using rotary position embeddings (RoPE) degrade sharply when given sequences longer than those seen in training. RoPE scaling methods adjust the rotation frequencies so a model can handle longer contexts with little or no additional training.

## Recap: RoPE

RoPE rotates query and key vectors by position-dependent angles. Each pair of dimensions has its own frequency: low dimensions rotate quickly (fine-grained local position), high dimensions rotate slowly (coarse long-range position).

## Why Extrapolation Fails

At positions beyond the training length, the model sees rotation angles it never encountered, so attention patterns break down.

## Position Interpolation (PI)

Rescale positions so the new, longer sequence maps back into the trained range:

```text
position' = position * (L_train / L_target)
```

This works with a short fine-tune but compresses all frequencies equally, blurring the high-frequency dimensions that encode nearby token order.

## NTK-Aware Scaling

Instead of compressing positions, change the RoPE base so high frequencies are preserved and low frequencies are stretched:

```text
base' = base * scale ^ (d / (d - 2))
```

This often works even without fine-tuning.

## YaRN

YaRN (Yet another RoPE extensioN) refines this with:

1. **Per-dimension interpolation**: high-frequency dimensions are left unchanged, low-frequency dimensions are interpolated, and the middle band blends between them
2. **Attention temperature scaling**: a small factor on attention logits compensates for the entropy shift that longer contexts cause

YaRN reaches long contexts with relatively few fine-tuning tokens.

## Configuration Example

```json
{
  "rope_scaling": {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
  }
}
```

## Caveats

- Extended length does not guarantee effective use of the full window; evaluate with long-context retrieval and reasoning tests.
- Static scaling can slightly degrade short-context quality.
