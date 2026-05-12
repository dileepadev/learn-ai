---
title: Test-Time Adaptation
description: Understand Test-Time Adaptation (TTA) — techniques that update a pre-trained model's parameters at inference time using only unlabeled test data — covering entropy minimization with TENT, batch normalization adaptation, MEMO, DUA, continual TTA, and the key challenges of distribution shift handling without labels.
---

When a model trained on a source distribution encounters test data from a shifted distribution — different lighting conditions, a new domain, different sensor characteristics, or different data collection protocols — its performance degrades. **Test-Time Adaptation (TTA)** addresses this by updating model parameters using the unlabeled test data itself, without access to source data or test labels.

TTA occupies a distinct position in the adaptation taxonomy:

| Setting | Labeled target data | Source data available | Model updates |
| --- | --- | --- | --- |
| Fine-tuning | Yes | Optional | Offline |
| Domain adaptation | No | Yes | Offline |
| Test-Time Training (TTT) | No | Yes (during training) | At inference |
| **Test-Time Adaptation (TTA)** | **No** | **No** | **At inference** |

TTA is more constrained than domain adaptation (no source data) and more constrained than TTT (no source-time auxiliary task setup) — it operates purely on test data at inference time, making it practical for deployed systems where source data is unavailable.

## Sources of Distribution Shift

TTA is motivated by real-world distribution shifts:

- **Covariate shift**: $P_{\text{test}}(x) \neq P_{\text{train}}(x)$ while $P(y|x)$ remains the same. Common in image recognition under corruptions (noise, blur, weather), sensor changes, or demographic shifts.
- **Concept drift**: $P(y|x)$ changes over time. Requires continual adaptation rather than one-time TTA.
- **Domain shift**: the marginal input distribution differs substantially (medical images from different scanners, satellite images from different sensors).

## Entropy Minimization: TENT

**TENT** (Wang et al., 2021) is the foundational TTA algorithm. It adapts only the **affine parameters** (scale $\gamma$ and bias $\beta$) of batch normalization layers while freezing all other parameters, using entropy minimization as the objective:

$$\mathcal{L}_{\text{TENT}} = H(\hat{y}) = -\sum_{c=1}^{C} p(\hat{y} = c | x) \log p(\hat{y} = c | x)$$

where $p(\hat{y} = c | x)$ is the softmax probability for class $c$ and $H$ is the Shannon entropy of the predictive distribution. Minimizing entropy pushes the model to make confident (low-entropy) predictions on test samples.

### Why Batch Normalization Parameters

Batch normalization layers contain two types of parameters:

- **Running statistics** ($\mu_{\text{running}}, \sigma_{\text{running}}$): accumulated during training, used at test time.
- **Affine parameters** ($\gamma, \beta$): learned during training, applied after normalization.

When the test distribution shifts, the running statistics become mismatched with test batch statistics. TENT adapts $\gamma$ and $\beta$ to compensate for this shift, requiring only $O(\text{channels})$ parameters to be updated — a tiny fraction of total parameters. This makes TENT fast and stable.

### TENT Update Rule

For each test batch $\{x_1, \ldots, x_B\}$:

1. Forward pass through the model; compute per-class probabilities.
2. Compute entropy loss $\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} H(\hat{y}_i)$.
3. Backpropagate through BN affine parameters only.
4. Update $\gamma, \beta$ with a single gradient step.

TENT updates the model continuously across the test stream — each batch updates the parameters used for subsequent batches.

## Batch Normalization Statistics Adaptation

Before TENT, simpler approaches adapted only the running statistics of batch normalization layers without any gradient computation:

### Test Batch Normalization (TestBN)

Replace running statistics with batch statistics computed from the test batch:

$$\hat{x} = \frac{x - \mu_{\text{batch}}}{\sigma_{\text{batch}}}$$

This is the simplest possible TTA approach — it requires no gradient computation, no hyperparameter tuning, and no auxiliary objective. It works well when test batches are large enough to estimate reliable batch statistics (typically $B \geq 64$).

### DUA (Distribution Uncertainty Adaptation)

**DUA** (Mirza et al., 2022) updates the running statistics by mixing source statistics with exponentially weighted test statistics:

$$\mu_{\text{new}} = (1 - \alpha) \cdot \mu_{\text{running}} + \alpha \cdot \mu_{\text{batch}}$$

$$\sigma_{\text{new}}^2 = (1 - \alpha) \cdot \sigma_{\text{running}}^2 + \alpha \cdot \sigma_{\text{batch}}^2$$

The mixing coefficient $\alpha$ controls the rate of adaptation. DUA processes test batches sequentially and accumulates updated statistics without computing gradients. It is particularly effective under moderate shifts where the source statistics are still partially useful.

## MEMO: Marginal Entropy Minimization

**MEMO** (Zhang et al., 2022) performs single-sample TTA — adapting the model using each test sample individually. Unlike TENT (which requires batches), MEMO applies multiple augmentations to a single test image and minimizes the **marginal entropy** across augmentations:

$$\mathcal{L}_{\text{MEMO}} = H\left(\mathbb{E}_{a \sim \mathcal{A}}[\hat{y}(a(x))]\right)$$

where $\mathcal{A}$ is a set of augmentations and the expectation is taken over augmented views. The marginal entropy is minimized (not the per-augmentation entropy), encouraging consistent predictions across augmentations.

### MEMO vs. TENT

| Aspect | TENT | MEMO |
| --- | --- | --- |
| Input requirement | Test batch | Single test sample |
| Gradient computation | Required | Required |
| Parameters adapted | BN affine params | All (or subset) |
| Latency overhead | Low per-sample | Moderate (multiple augmentations) |
| Suitable for batch inference | Yes | Yes (with amortization) |
| Suitable for single-sample inference | No | Yes |

MEMO is particularly valuable in settings where samples arrive one at a time (e.g., interactive inference) or where batch composition is not controlled.

## Continual Test-Time Adaptation

Standard TTA assumes a single, stationary test distribution. **Continual TTA** addresses scenarios where the test distribution itself evolves over time:

### Challenges in Continual TTA

- **Error accumulation**: gradient-based TTA can drift from the original model as adaptation errors compound.
- **Forgetting**: adapting to the current distribution degrades performance on previously seen distributions.
- **Non-stationary shifts**: the test distribution may change abruptly or gradually in different deployment contexts.

### CoTTA (Continual Test-Time Adaptation)

**CoTTA** (Wang et al., 2022) addresses continual adaptation with two mechanisms:

- **Stochastic restore**: randomly reset a fraction of BN parameters to their pre-trained source values at each step, preventing catastrophic drift.
- **Augmentation-averaged predictions**: compute predictions averaged over multiple augmentations (similar to MEMO) to reduce label noise in the self-training signal.

The stochastic restore provides a regularization effect that prevents complete departure from source domain knowledge, while augmentation averaging improves the reliability of pseudo-labels for the adaptation objective.

## SAR: Sharpness-Aware and Reliable Entropy Minimization

**SAR** (Niu et al., 2023) addresses two failure modes of TENT:

- Noisy samples with inherently high entropy (e.g., corrupted or ambiguous inputs) produce harmful gradient updates.
- Online entropy minimization can lead to sharpness in the loss landscape, making the adapted model sensitive to small shifts.

SAR filters out high-entropy samples before computing gradient updates and applies sharpness-aware minimization to the remaining samples, seeking flat minima in the adaptation loss landscape. This improves stability under severe corruptions and corrupted test batches.

## Empirical Results on CIFAR-10-C and ImageNet-C

Standard benchmarks for TTA evaluate classification accuracy under 15 corruption types (noise, blur, weather, digital) at 5 severity levels:

| Method | Params Updated | ImageNet-C Error (avg) | Notes |
| --- | --- | --- | --- |
| Source (no adapt) | None | 82.4% | ResNet-50 baseline |
| BN Statistics (TestBN) | BN stats only | 74.3% | No gradient needed |
| DUA | BN stats only | 72.8% | Sequential accumulation |
| TENT | BN affine | 70.7% | Gradient, batched |
| MEMO | All params | 63.8% | Gradient, single-sample |
| SAR | BN affine | 69.1% | Gradient, filtered |
| CoTTA | BN affine | 68.6% | Continual setting |

Performance gains from TTA are largest at high corruption severity levels, where source distribution mismatch is greatest.

## TTA for Large Pretrained Models

Applying TTA to large transformer-based models (ViT, CLIP, LLaMs) requires care:

- **Layer normalization vs. batch normalization**: vision transformers use layer normalization rather than batch normalization. TTA approaches designed for BN statistics must be redesigned for LN — adapting LN scale/shift parameters via entropy minimization remains effective but requires different hyperparameter choices.
- **Parameter efficiency**: adapting all parameters of a billion-parameter model at test time is computationally prohibitive. Methods like **Efficient Test-Time Adaptation (ETA)** apply TTA only to a small subset of attention layers or adapter modules.
- **CLIP-based TTA**: TPT (Test-Time Prompt Tuning) adapts a text prompt (in the CLIP embedding space) at test time using entropy minimization over augmented views of a test image, achieving strong zero-shot robustness without modifying image encoder parameters.

## Practical Considerations

### Batch Size Sensitivity

TENT and entropy-based methods require reliable entropy estimates, which improve with larger batch sizes. At $B=1$, entropy is noisy and gradient steps may hurt. MEMO addresses this by using multiple augmentations instead of multiple samples.

### Adaptation Step Count

Most TTA methods apply a single gradient step per batch. Multiple steps per batch can improve accuracy but risk overfitting to the current batch and destabilizing the model.

### Learning Rate Tuning

TTA is sensitive to learning rate. Too large: catastrophic forgetting of source knowledge. Too small: insufficient adaptation. Common practice is to use a learning rate $10\text{–}100\times$ smaller than the original training learning rate.

### Failure Modes

- TTA can harm performance if test distribution is similar to the source distribution and the batch is small.
- Long test streams under non-stationary shifts cause error accumulation.
- Class-imbalanced test batches bias entropy minimization toward majority classes.

## Summary

Test-Time Adaptation enables models to adapt to distribution shift at inference time using only unlabeled test data. TENT established entropy minimization over batch normalization parameters as the core primitive. DUA provides gradient-free statistics adaptation. MEMO extends TTA to single-sample inference via augmentation-averaged marginal entropy. Continual TTA methods like CoTTA address non-stationary shifts with stochastic restoration. For large transformer models, prompt-based and adapter-based TTA approaches replace BN-specific methods. TTA is a practical tool for deployed systems where source data is unavailable, distribution shifts are expected, and labels cannot be obtained at test time.
