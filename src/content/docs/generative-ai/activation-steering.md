---
title: Activation Steering and Representation Engineering
description: Learn how activation steering and representation engineering techniques allow researchers to read and write high-level concepts directly into neural network activations, enabling mechanistic control over model behavior.
---

Activation steering and representation engineering are interpretability and control techniques that operate directly on the internal activations of neural networks — reading out encoded concepts and injecting or suppressing them at inference time. These methods treat a model's residual stream as a **medium for high-level computation** that can be surgically manipulated.

## The Central Insight

Transformer language models represent concepts as **directions in activation space**. A concept like "sentiment," "truthfulness," or "anger" tends to correspond to a linear direction in the high-dimensional residual stream — not a single neuron or a complex non-linear manifold, but approximately a **ray** in vector space.

If you can identify this direction, you can:

- **Read:** Detect whether the model is currently representing that concept
- **Write (steer):** Add or subtract the direction to amplify or suppress the concept
- **Probe:** Train a linear classifier on activations to predict when the concept is present

This is the foundation of both activation steering and representation engineering.

## Probing Classifiers

A **linear probe** is a logistic regression classifier trained on a model's internal activations to predict some property:

```python
# Collect activations at layer L for prompts labeled with concept C
activations = [get_activations(model, prompt, layer=L) for prompt in dataset]
labels = [1 if "concept" in prompt else 0 for prompt in dataset]

# Train linear probe
probe = LogisticRegression().fit(activations, labels)
```

If a linear probe achieves high accuracy, the concept is **linearly represented** in the model's activations at that layer. This is a core finding of mechanistic interpretability: many surprisingly abstract concepts (e.g., "the model is about to make a factual error" or "this text is in French") are linearly decodable from intermediate activations.

## Activation Steering

Activation steering (also called **activation addition** or **inference-time intervention**) injects concept vectors into the residual stream during a forward pass to change model behavior.

### Computing the Steering Vector

1. Collect activations for **contrastive pairs**: prompts that differ only in the target concept  
   (e.g., "I love this" vs. "I hate this" for sentiment direction)
2. Compute the **mean difference** between positive and negative activations at layer $L$:
   $$\vec{v}_\text{concept} = \mathbb{E}[\text{act}(x^+)] - \mathbb{E}[\text{act}(x^-)]$$
3. Normalize to unit length

### Applying the Steering Vector

During inference, add the scaled steering vector to the residual stream at layer $L$:

$$h_L \leftarrow h_L + \alpha \cdot \vec{v}_\text{concept}$$

where $\alpha$ controls the steering intensity (positive = amplify, negative = suppress).

**Example:** Adding a "speak in French" steering vector causes an English-prompted model to respond in French. Adding a "reasoning" vector improves performance on logic tasks without any prompt change.

### Empirical Results

- **Anthropic's Claude Sonnet 3.5:** Steering vectors for "banana" caused the model to be preoccupied with bananas while continuing a conversation — demonstrating that steered concepts influence processing in the expected direction
- **Llama 2:** Steering vectors for "truthfulness" reduce sycophancy; vectors for "refusal" cause unwanted over-refusal of benign prompts
- **Safety relevance:** A "dangerous" concept direction can be identified and suppressed, potentially preventing harmful outputs without fine-tuning

## Representation Engineering (RepE)

Representation Engineering (Zou et al., 2023) formalizes activation reading and writing at scale, arguing that model "emotions," "goals," and "knowledge states" are encoded as representational structures that can be directly monitored and controlled.

### Key Findings from RepE

- Models have detectable internal representations of **honesty/deception** — they sometimes "know" a statement is false before generating it
- There are measurable representations of **emotional states** (fear, happiness, anger) in LLMs that correlate with biased outputs
- **Power-seeking** tendencies have identifiable activation signatures that precede agentic behaviors
- Injecting "calm" representations reduces anxiety-calibrated model behavior

### Control Vectors vs. Steering Vectors

While similar, these approaches differ in scope:

- **Steering vectors:** Targeted single-concept injection at specific layers
- **Control vectors (llama.cpp):** Multiple concept vectors applied simultaneously across all layers via GGUF model export, enabling deployment-time model personality control without reloading model weights

```bash
# Example: apply a control vector with llama.cpp
./main -m model.gguf --control-vector friendly.gguf --control-vector-scaled formal.gguf:0.5
```

## Contrastive Activation Addition (CAA)

CAA (Rimsky et al., 2023) is a direct application method:

1. Create paired datasets: $A^+$ (with target behavior) and $A^-$ (without it)
2. Run both sets through the model, collecting residual stream activations at the middle layers
3. Subtract mean activations: $\vec{v} = \bar{h}(A^+) - \bar{h}(A^-)$
4. Apply $\vec{v}$ at inference time on new inputs

CAA demonstrated steering models toward/away from sycophancy, refusals, and political opinions using vectors computed from only ~100 contrastive examples.

## Relation to Mechanistic Interpretability

Activation steering and representation engineering are closely related to but distinct from mechanistic interpretability:

| Approach | Goal | Method |
|---|---|---|
| Mechanistic Interpretability | Understand computations inside the model | Circuit analysis, superposition, feature attribution |
| Activation Steering | Control model behavior at runtime | Inject/subtract concept vectors |
| Representation Engineering | Map and govern internal model states | Systematic probing + control |

They are **complementary**: mechanistic interpretability identifies *what* is represented and *how*; activation steering uses those findings to *control* the representation.

## Limitations and Risks

- **Steering instability:** Too large an $\alpha$ causes incoherent outputs; the concept bleeds into unrelated generation
- **Side effects:** Steering one concept can inadvertently shift correlated concepts
- **Linear assumption:** Some concepts may not be linearly represented; the technique fails for these
- **Misuse potential:** Steering could suppress safety behaviors or amplify harmful tendencies if applied maliciously
- **Temporary effect:** Steering only affects a single inference; it doesn't update the model's weights

## Applications

- **Real-time behavior control:** Adjusting model tone, verbosity, or persona without fine-tuning
- **Safety monitoring:** Detecting when a model's internal state suggests it is about to produce harmful output
- **Interpretability research:** Understanding what concepts models encode and how they process them
- **Bias analysis:** Measuring which demographic concepts are encoded in intermediate layers
- **Personalization:** User-specific control vectors applied at serving time without per-user fine-tuning

## Tools and Resources

- **repeng (ggerganov):** Python library for computing and applying representation vectors
- **llama.cpp control vectors:** Built-in support for applying control vectors from GGUF models
- **TransformerLens:** Mechanistic interpretability library with activation hooks for steering
- **nnsight:** Intervention framework for studying large model internals

## Further Reading

- Zou et al. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency*
- Rimsky et al. (2023), *Steering Llama 2 via Contrastive Activation Addition*
- Turner et al. (2023), *Activation Addition: Steering Language Models Without Optimization*
- Templeton et al. (2024), *Scaling Monosemanticity: Extracting Interpretable Features from Claude Sonnet 3* (Anthropic)
