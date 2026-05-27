---
title: Knowledge Distillation
description: Master knowledge distillation — the technique for transferring knowledge from large teacher models to compact student models — covering soft targets and temperature scaling, intermediate feature distillation, Born Again Networks, data-free distillation, and applications to LLM compression via DistilBERT and TinyLLaMA.
---

A large neural network trained on substantial compute and data encodes rich knowledge in its weight structure and output distributions — knowledge that goes far beyond the binary correct/incorrect signal of labeled data. **Knowledge distillation** transfers this knowledge to a smaller, faster student model by training the student to mimic the teacher's behavior, producing compact models that significantly outperform equivalently-sized models trained from scratch.

## The Core Idea: Soft Targets

A neural network's output probability distribution over all classes contains more information than a one-hot label. For an image of a "2", a well-trained model might output 85% probability for "2", 10% for "3", and 5% for "7" — encoding the observation that 2 and 3 are visually similar, and 7 is less similar but still non-negligible. These **soft targets** provide a much richer training signal than the hard label.

**Hinton et al. (2015)** formalized this insight. The student is trained to match the teacher's softened output distribution using a temperature parameter $T$:

$$\text{softmax}(\mathbf{z}_i / T)_j = \frac{\exp(z_{ij}/T)}{\sum_k \exp(z_{ik}/T)}$$

Higher temperature $T$ softens the distribution — making small probability differences more prominent for the student to learn from. At $T = 1$, the distribution is the normal softmax; as $T \to \infty$, it approaches uniform.

The distillation loss combines two terms:

$$\mathcal{L}_{\mathrm{KD}} = (1 - \alpha)\, \mathcal{L}_{\mathrm{CE}}(\mathbf{y}_{\mathrm{hard}}, \sigma(\mathbf{z}_S)) + \alpha\, T^2\, \mathcal{L}_{\mathrm{KL}}\!\left(\sigma(\mathbf{z}_T / T),\, \sigma(\mathbf{z}_S / T)\right)$$

The $T^2$ factor compensates for the gradient magnitude being $T$ times smaller when using soft targets.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature: float = 4.0,
    alpha: float = 0.7,
):
    """
    Hinton-style knowledge distillation loss.
    alpha: weight for soft-target loss (1-alpha for hard-label loss)
    temperature: softens teacher/student distributions
    """
    # Hard-label cross-entropy loss
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft-target KL divergence loss
    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher.exp(), reduction="batchmean")

    return (1 - alpha) * hard_loss + alpha * (temperature ** 2) * soft_loss


class DistillationTrainer:
    def __init__(self, teacher, student, temperature=4.0, alpha=0.7):
        self.teacher = teacher.eval()  # Teacher is frozen
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    def train_step(self, x, y):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        loss = distillation_loss(
            student_logits, teacher_logits, y,
            temperature=self.temperature, alpha=self.alpha,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

## Feature-Based Distillation

Soft-target distillation only matches final outputs. **Feature-based distillation** aligns intermediate representations — forcing the student to develop similar internal computations.

### FitNets

FitNets (Romero et al., 2015) adds a hint layer: a linear projection that maps the student's intermediate features to the same dimension as the teacher's, then minimizes MSE between the two:

```python
class FitNetHintLayer(nn.Module):
    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.regressor = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)

    def forward(self, student_feat, teacher_feat):
        projected = self.regressor(student_feat)
        return F.mse_loss(projected, teacher_feat.detach())
```

### Attention Transfer

Attention Transfer (Zagoruyko & Komodakis, 2017) computes **attention maps** — squared, channel-averaged feature norms — and aligns them across teacher and student at multiple depths:

$$\mathcal{L}_{\mathrm{AT}} = \frac{\beta}{2}\sum_l \left\| \frac{Q_S^l}{\|Q_S^l\|_2} - \frac{Q_T^l}{\|Q_T^l\|_2} \right\|_2^2$$

where $Q^l = \sum_c A_{ijc}^2$ is the spatial attention map of layer $l$.

### Relational Knowledge Distillation

RKD (Park et al., 2019) transfers **relationships** between examples rather than individual activations — preserving the geometric structure of the embedding space:

```python
def rkd_distance_loss(student_emb, teacher_emb):
    """Relational KD: match pairwise distances between examples."""
    t_dists = torch.cdist(teacher_emb, teacher_emb, p=2)
    s_dists = torch.cdist(student_emb, student_emb, p=2)
    # Normalize by mean distance
    t_dists = t_dists / (t_dists.mean() + 1e-8)
    s_dists = s_dists / (s_dists.mean() + 1e-8)
    return F.smooth_l1_loss(s_dists, t_dists)
```

## Born Again Networks

Born Again Networks (BANs, Furlanello et al., 2018) show that a student with the **same architecture** as the teacher — distilled from the teacher — consistently outperforms the teacher. Training a sequence of students from each other (a "generation" of BANs) and ensembling them produces state-of-the-art results:

```text
Teacher (gen 0) → Student₁ (gen 1) → Student₂ (gen 2) → ... → Studentₖ

Ensemble of all generations outperforms teacher ensemble trained from scratch.
```

This suggests distillation acts as a form of regularization, and soft labels provide a better training signal than hard labels regardless of model size.

## Data-Free Distillation

In many settings, the original training data is unavailable (privacy, proprietary data). **Data-free distillation** synthesizes surrogate data from the teacher:

```python
class DeepInversion:
    """Generate synthetic inputs that maximize teacher network activations."""

    def __init__(self, teacher, num_classes: int, image_size: int = 32):
        self.teacher = teacher.eval()
        self.num_classes = num_classes
        self.image_size = image_size

    def synthesize(self, target_class: int, iterations: int = 2000, lr: float = 0.1):
        # Optimizable synthetic image
        x = torch.randn(1, 3, self.image_size, self.image_size, requires_grad=True)
        y = torch.tensor([target_class])
        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(iterations):
            optimizer.zero_grad()
            logits = self.teacher(x)
            loss_ce = F.cross_entropy(logits, y)
            # BN statistics regularization: penalize deviation from stored BN moments
            loss_bn = self._bn_regularization_loss()
            loss = loss_ce + 0.001 * loss_bn
            loss.backward()
            optimizer.step()
            x.data.clamp_(0, 1)

        return x.detach()

    def _bn_regularization_loss(self):
        # Compute deviation of synthetic batch statistics from stored BN running stats
        total = torch.tensor(0.0)
        for module in self.teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                total += (module.running_mean.detach() - 0.0).pow(2).mean()
        return total
```

## LLM Compression via Distillation

Knowledge distillation is central to producing practical LLMs.

### DistilBERT

DistilBERT (Sanh et al., 2019) distills BERT-base into a 6-layer, 66M-parameter model (40% smaller, 60% faster) that retains 97% of BERT's performance on GLUE. Key choices:

- **Initialization**: student layers initialized from every other teacher layer
- **Loss**: KL divergence on MLM token probabilities + cosine embedding loss on hidden states + hard MLM loss

### TinyLLaMA

TinyLLaMA (Zhang et al., 2024) trains a 1.1B parameter model matching LLaMA-2 7B performance on many benchmarks through a combination of:

- Distillation from LLaMA-2 on 3T tokens
- Flash Attention and FSDP for efficiency
- Grouped-query attention to reduce KV cache size

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Sequence-level distillation for causal LMs
teacher = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
student = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def causal_lm_distillation_loss(student, teacher, input_ids, temperature=2.0, alpha=0.5):
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits

    student_logits = student(input_ids).logits

    # Shift for next-token prediction
    shift_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_student = student_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    soft_loss = F.kl_div(
        F.log_softmax(shift_student / temperature, dim=-1),
        F.softmax(shift_teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(
        shift_student.view(-1, shift_student.size(-1)),
        shift_labels.view(-1),
    )
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

## Summary

Knowledge distillation is one of the most practical model compression techniques, combining theoretical clarity with consistent empirical gains:

- **Soft targets** carry richer information than hard labels; temperature scaling controls the softness
- **Feature distillation** (FitNets, attention transfer, RKD) aligns intermediate representations, providing supervision beyond the output layer
- **Born Again Networks** show that distillation is beneficial even when student and teacher share the same architecture — soft labels are a better training signal than hard labels
- **Data-free distillation** synthesizes training data from teacher activations and BN statistics when original data is unavailable
- For LLMs, distillation produces models like DistilBERT and TinyLLaMA that approach the performance of models 3–7× larger, enabling deployment on resource-constrained hardware
