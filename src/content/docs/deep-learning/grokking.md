---
title: Grokking - Delayed Generalization in Neural Networks
description: Explore the grokking phenomenon where neural networks first memorize training data and then, long after overfitting, suddenly generalize — and what this reveals about learning dynamics.
---

Grokking is a striking training phenomenon: a neural network first overfits the training set to near-perfect accuracy, apparently failing to generalize, and then — after many more gradient steps — abruptly achieves strong test performance as well. The word "grokking" was coined by Power et al. (2022) in a paper showing that small transformers trained on modular arithmetic exhibit this double-phase transition.

## The Original Observation

Power et al. trained transformers on tasks like modular addition ($a + b \mod p$) with a small dataset of equation pairs. The models reached 100% training accuracy quickly, then plateaued near chance on the validation set for thousands of additional steps — before suddenly jumping to near-perfect generalization, sometimes 100× or 1000× more steps after the memorization phase.

This was surprising because conventional wisdom suggests that once a model overfits, more training will only entrench memorization further.

## Why Does Grokking Happen?

### Weight Norm Dynamics

One key finding is that the weight norms of the network continue growing during the memorization phase and then start to **decrease** just before generalization occurs. This norm compression appears connected to implicit regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \|\theta\|^2$$

Even with small explicit weight decay $\lambda$, the regularizer eventually wins over memorization solutions because memorization solutions tend to have larger weight norms than general solutions.

### Efficiency of Representations

Grokking can be understood through the lens of **representation efficiency**. Memorization solutions are "lazy" — they interpolate training examples without learning structure. Generalization solutions are more compressed and structured. The network explores weight space until it finds a more efficient representation.

### Phase Transitions and Circuits

Mechanistic interpretability work has revealed that grokking in modular arithmetic corresponds to the model learning **Fourier-based circuits**. During grokking, specific attention heads develop periodic activations aligned to the Fourier decomposition of the modular task:

$$f(a, b) = \sum_k A_k \cos\left(\frac{2\pi k (a + b)}{p}\right)$$

The model effectively discovers the underlying mathematical structure.

## Conditions That Influence Grokking

Several factors affect whether and when grokking occurs:

### Dataset Fraction

Grokking is most pronounced when the training set is a small fraction of all possible inputs. As training data coverage increases, the delay between memorization and generalization shrinks.

| Training fraction | Generalization step (approx.) |
| --- | --- |
| 30% | ~100,000 |
| 50% | ~10,000 |
| 80% | ~1,000 |
| 95% | ~100 |

### Weight Decay

Weight decay is often **necessary** for grokking to occur at all. Without regularization, models may remain in the memorization regime indefinitely. The right amount of weight decay accelerates grokking; too much prevents memorization from succeeding initially.

### Learning Rate

Lower learning rates increase the memorization-to-generalization gap. Higher learning rates can accelerate grokking but may destabilize training.

### Architecture

Grokking has been observed in:

- Transformers (original discovery)
- MLPs on algorithmic tasks
- Convolutional networks on image tasks
- Random feature models

It is not unique to transformers but seems to require networks with sufficient capacity relative to data.

## Grokking Beyond Algorithmic Tasks

Initial reports focused on algorithmic/mathematical datasets, but grokking has since been observed in:

### Image Classification

Small CNNs on CIFAR subsets show delayed generalization when trained long enough with appropriate weight decay.

### Natural Language

Fine-tuning pre-trained language models on small datasets sometimes exhibits grokking-like dynamics where early training improves loss but not downstream task accuracy.

### Sparse Regression

Linear models with sparsity-inducing regularization show analogous behavior — overfitting many noise features before selecting the true sparse support.

## Accelerating Grokking

Several interventions can accelerate the transition from memorization to generalization:

### Adaptive Weight Decay

Dynamically increasing weight decay after detecting plateau in validation loss shortens the delay significantly.

### Representation Sparsification

Techniques like dropout or activation sparsity that discourage large distributed representations push networks toward generalization solutions earlier.

### Grokfast

The **Grokfast** method (Lee et al., 2024) uses an exponential moving average of gradients to amplify slow-changing gradient components — which correspond to the generalization signal — rather than fast-changing ones associated with memorization:

$$g_{\text{slow}, t} = \alpha \cdot g_{\text{slow}, t-1} + (1 - \alpha) \cdot g_t$$

$$g_{\text{amplified}, t} = g_t + \lambda \cdot g_{\text{slow}, t}$$

This can reduce the generalization delay by an order of magnitude.

### SAM and Flat Minima

Sharpness-Aware Minimization (SAM), which explicitly seeks flat loss minima, accelerates grokking because generalization solutions tend to reside in flatter regions of the loss landscape.

## Theoretical Interpretations

### Slingshot Dynamics

One theory describes a "slingshot" mechanism: weight norms grow until regularization causes them to collapse, at which point the model is forced into a lower-norm generalization solution.

### Information Compression

From an information-theoretic perspective, grokking mirrors the compression phase in the **information bottleneck** theory of deep learning. The model first fits labels (memorization) and then compresses representations (generalization).

### Algorithmic Phase Transitions

From a statistical mechanics perspective, grokking resembles a first-order phase transition. The system sits in a metastable memorization state and then tunnels to the lower free-energy generalization state.

## Relationship to Double Descent

Grokking is related to but distinct from the **double descent** phenomenon:

- **Double descent**: Risk curve as a function of model size or dataset size
- **Grokking**: Temporal dynamics during training for a fixed model/dataset

Both challenge the classical bias-variance tradeoff and suggest that overparameterized models can generalize when trained long enough.

## Implications for Practice

Grokking has several practical implications:

### Training Budget

For tasks with structured data and small datasets, using an aggressive early stopping criterion based solely on validation loss can prematurely terminate training before generalization kicks in.

### Regularization Tuning

Weight decay, which is often treated as a minor hyperparameter, is central to grokking. Tuning it carefully matters especially in low-data regimes.

### Interpretability

The mechanistic interpretability work on grokking has become a paradigmatic example of how neural networks develop interpretable internal algorithms. It motivates studying weight dynamics during training, not just at convergence.

### Curriculum and Data Design

Understanding grokking motivates designing tasks and datasets where the efficient/general solution has a meaningfully lower norm or complexity than the memorization solution.

## Open Questions

Several questions remain active research areas:

- Under what conditions does grokking occur in large-scale models?
- Can grokking explain sudden capability jumps observed during pre-training of large language models?
- Is there a principled theory connecting weight norm dynamics to Kolmogorov complexity of solutions?
- How does grokking interact with data augmentation, batch size, and optimizer choice?

## Summary

Grokking reveals that the relationship between training time and generalization is non-monotone and sometimes discontinuous. A network that appears fully overfit may be one long training run away from generalizing. The phenomenon has deepened understanding of implicit regularization, representation learning, and the geometry of neural network loss landscapes — and has become a rich testbed for mechanistic interpretability research.
