---
title: Emergent Abilities and Scaling Laws
description: Analyzing how model capabilities emerge with scale — predicting performance, understanding phase transitions, and implications for foundation models.
---

**Emergent abilities** are capabilities that appear suddenly as models scale up in size or training data, despite being absent in smaller models. A small language model cannot perform arithmetic; GPT-3 with 175B parameters solves simple math problems.

**Scaling laws** quantify how performance improves with scale, enabling prediction of large model capabilities before training.

## Emergent Abilities

### Examples

**In-context learning**: Small models can't learn from examples in prompts. GPT-3 demonstrates in-context learning across diverse tasks; GPT-4 does even better.

**Reasoning**: Smaller models fail at multi-step reasoning. Larger models show chain-of-thought capability.

**Instruction following**: Very large LLMs follow complex, natural-language instructions without task-specific fine-tuning.

### Phase Transitions

Some abilities appear suddenly:

```
Model Size (Parameters)
→ Smaller models: 0% accuracy on complex reasoning
→ Medium models: Still ~0%
→ Large model (GPT-3 size): Suddenly 50%+ accuracy
→ Even larger: 80%+
```

Rather than gradual improvement, there's a sharp transition — an apparent phase change.

### Why Emergence Happens

**Competing hypotheses**:

1. **Continuity**: Abilities improve gradually; the apparent discontinuity is measurement artifact or evaluation bias.

2. **Threshold effects**: Certain model sizes/data amounts are needed to represent necessary concepts. Below threshold: impossible. Above: achievable.

3. **Loss landscape**: Larger models access regions of parameter space enabling new capabilities; smaller models' constraints prevent access.

Debate remains; likely multiple mechanisms contribute.

## Scaling Laws

### Empirical Power Laws

Chinchilla et al. (2022) and others found that loss follows a power law with scale:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where:
- $N$: Number of parameters.
- $D$: Tokens in training data.
- $E$: Irreducible error (lower bound).
- $\alpha, \beta \approx 0.07$: Exponents (vary slightly by setup).

**Prediction**: Given model size $N$ and data $D$, predict test loss without training.

### Chinchilla-Optimal Scaling

Standard: Scale model size much more than data (10x more parameters, but only 1x more data).

**Chinchilla finding**: Optimal allocation is roughly **equal compute for model and data**: $N \approx D / 20$ (approximate).

If you can do $10^{20}$ FLOPs:
- Traditional: 170B params, 150B tokens (GPT-3 setup).
- Chinchilla-optimal: 70B params, 1.4T tokens.

70B params trained on 1.4T tokens outperforms 175B on 300B.

**Implication**: Prior scaling focused too much on parameter count; data efficiency was underemphasized.

### Beyond Loss: Downstream Task Performance

Loss scaling laws are well-studied, but downstream task performance follows different laws:

$$\text{Accuracy}(N) = 1 - (k N^{-\alpha})$$

for some task-specific $k, \alpha$.

- Tasks vary in scaling behavior: some improve faster, some slower.
- Emergent abilities create non-smooth curves.

## Predicting Emergent Abilities

### Loss Doesn't Predict All Abilities

A model can have low loss (accurate next-token prediction) yet fail at in-context learning. Predicting when abilities emerge remains open.

### Grokking

Surprising phenomenon: models sometimes generalize (pass test set) long after training loss plateaus. Called "grokking":

1. Training loss decreases quickly.
2. Test loss remains high for long time.
3. Suddenly, test loss drops dramatically.

Suggests phases: initial memorization, then delayed generalization. Relevant to understanding emergent abilities.

### Critical Period Hypothesis

Certain abilities may require sufficient scale early in training. Missing this "critical period" prevents the ability from emerging later.

## Implications for Foundation Models

### Scaling Efficiency

Scaling laws predict compute requirements. Planning large-model training:

- Estimate performance targets.
- Use scaling laws to compute required model size and data.
- Budget accordingly (FLOPs, energy, time).

### Transfer and Generalization

Larger models generalize better and transfer to more tasks. Scaling laws suggest this trend continues, but with diminishing returns.

### Frontier Models

Each new frontier model (GPT-3 → GPT-4, BERT → T5) demonstrates emergent abilities absent in predecessors. Scaling laws help predict the next frontier's capabilities.

## Limitations and Open Questions

### Scaling Isn't Everything

Scaling laws describe average trends. Outliers exist:
- Some small models are surprisingly capable (efficient architectures).
- Some large models underperform predictions (poor training, data issues).

### Compute vs. Real-World Constraints

Scaling laws assume unlimited compute. In reality:
- Training costs are high (GPT-3: ~$5M-10M).
- Inference costs increase with model size.
- Latency constraints limit deployable model sizes.

### Generalization to New Domains

Scaling laws fit existing data; extrapolating far beyond observed scales is risky. Assumptions may break.

### Quality of Data

Most scaling laws assume data quality is constant. In reality, low-quality data becomes dominant in large-scale training — may bend or break scaling laws.

## Architectural Innovations

Scaling laws assume fixed architectures (e.g., transformer). Architecture changes (sparse models, different attention) can dramatically shift scaling:

- **Sparse models**: Reduce parameters while maintaining performance.
- **Mixture of experts**: Conditional compute; activate different experts for different inputs.
- **Retrieval-augmented models**: Augment with external knowledge instead of scaling parameters.

These innovations may change scaling laws fundamentally.

## Current Research Directions

- **Predicting emergent abilities**: Can we predict when new capabilities emerge before training?
- **Sample efficiency**: Are there ways to train as effectively with less data?
- **Compute-optimal training**: Balancing model size, data, and training duration for fixed compute budget.
- **Long-horizon scaling**: How do scaling laws extend beyond current frontier models (1T parameters)?
- **Beyond next-token prediction**: Do scaling laws for other objectives (vision, multimodal) follow similar patterns?

## Practical Takeaways

1. **Scale matters**: Bigger models demonstrate qualitatively new abilities.
2. **Data-model balance**: Don't scale parameters alone; balance with data.
3. **Prediction is possible**: Use scaling laws to estimate compute needs before full training.
4. **Emergence is subtle**: Some abilities appear suddenly; predicting exactly when is hard.

Scaling laws and emergent abilities are reshaping how we think about AI development — from designing clever architectures to scaling intelligently. Understanding these principles is critical for predicting capabilities and resource requirements of future AI systems.
