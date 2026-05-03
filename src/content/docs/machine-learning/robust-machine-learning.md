---
title: Robust Machine Learning
description: Learn how to build ML models that remain reliable under adversarial attacks, distribution shift, and real-world noise. Covers adversarial examples, FGSM and PGD attacks, adversarial training, certified defenses via randomized smoothing, and distributionally robust optimization (DRO).
---

**Robust machine learning** addresses a fundamental gap between benchmark performance and real-world reliability. Standard ML training minimizes average loss on i.i.d. samples from a training distribution — but deployed models face distribution shifts, measurement noise, and deliberate adversarial manipulation. A model that achieves 99% accuracy on a clean test set can be fooled into misclassifying any input by adding imperceptible perturbations invisible to humans.

Robustness research studies how to build models that maintain predictive accuracy across a certified neighborhood of inputs and remain reliable under the kinds of distribution variation found in deployment.

## Adversarial Examples

Adversarial examples are inputs crafted by adding small, carefully chosen perturbations that cause a model to make incorrect predictions with high confidence. Formally, given a model $f$ with loss $\mathcal{L}$, input $x$, and true label $y$, an adversarial example $x_{adv}$ satisfies:

$$x_{adv} = \underset{x' : \|x' - x\|_p \leq \epsilon}{\arg\max}\; \mathcal{L}(f(x'), y)$$

The $\ell_\infty$ threat model ($\|x' - x\|_\infty \leq \epsilon$) is most common for images: each pixel can change by at most $\epsilon$ (e.g., $8/255$ for a $[0,1]$-scaled image), making perturbations invisible to humans.

## FGSM: Fast Gradient Sign Method

The simplest attack — a single gradient step in the direction that maximally increases loss:

$$x_{adv} = x + \epsilon \cdot \text{sign}\!\left(\nabla_x \mathcal{L}(f(x), y)\right)$$

```python
import torch
import torch.nn.functional as F

def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,       # (B, C, H, W), normalized to [0, 1]
    labels: torch.Tensor,       # (B,) integer class labels
    epsilon: float = 8/255,     # maximum perturbation magnitude (ℓ∞)
    targeted: bool = False,     # False = untargeted (maximize loss)
    target_labels: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).
    Single-step adversarial attack in the ℓ∞ threat model.
    
    Untargeted: push prediction away from true label.
    Targeted: push prediction toward target_labels.
    """
    images = images.clone().detach().requires_grad_(True)
    
    outputs = model(images)
    
    if targeted and target_labels is not None:
        # Minimize loss w.r.t. target → subtract gradient direction
        loss = F.cross_entropy(outputs, target_labels)
        sign_mult = -1
    else:
        # Maximize loss w.r.t. true label → add gradient direction
        loss = F.cross_entropy(outputs, labels)
        sign_mult = 1
    
    model.zero_grad()
    loss.backward()
    
    # Perturbation = epsilon * sign(gradient)
    perturbation = sign_mult * epsilon * images.grad.sign()
    adversarial = (images + perturbation).detach()
    
    # Clamp to valid image range
    adversarial = adversarial.clamp(0, 1)
    return adversarial
```

## PGD: Projected Gradient Descent

FGSM is weak — a single step rarely finds the worst-case perturbation. **PGD (Madry et al., 2018)** iterates multiple gradient steps, projecting back to the $\epsilon$-ball after each step:

$$x^{(0)} = x + \mathcal{U}(-\epsilon, \epsilon), \quad x^{(t+1)} = \Pi_\epsilon\!\left(x^{(t)} + \alpha \cdot \text{sign}(\nabla_{x^{(t)}} \mathcal{L})\right)$$

where $\Pi_\epsilon$ is the projection onto the $\ell_\infty$ ball of radius $\epsilon$ around $x$, and $\alpha$ is the step size.

```python
def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8/255,
    alpha: float = 2/255,       # step size (typically epsilon/4)
    num_steps: int = 40,        # more steps → stronger attack
    random_start: bool = True   # random initialization within epsilon-ball
) -> torch.Tensor:
    """
    PGD adversarial attack (Madry et al., 2018).
    Iterative ℓ∞-constrained attack, the 'gold standard' for evaluating robustness.
    
    PGD with many steps + random restarts provides a strong lower bound on
    a model's true adversarial robustness.
    """
    adv = images.clone().detach()
    
    if random_start:
        # Random initialization within epsilon-ball (uniform)
        delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
        adv = (adv + delta).clamp(0, 1).detach()
    
    for step in range(num_steps):
        adv.requires_grad_(True)
        output = model(adv)
        loss = F.cross_entropy(output, labels)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            # Gradient step
            adv = adv + alpha * adv.grad.sign()
            # Project back to epsilon-ball centered at original image
            delta = (adv - images).clamp(-epsilon, epsilon)
            adv = (images + delta).clamp(0, 1)
    
    return adv.detach()
```

## Adversarial Training

The most effective empirical defense: augment training with adversarial examples generated on-the-fly. The adversarially trained model minimizes the **minimax objective**:

$$\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}}\!\left[\max_{\delta: \|\delta\|_p \leq \epsilon} \mathcal{L}(f_\theta(x + \delta), y)\right]$$

```python
def adversarial_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8/255,
    pgd_steps: int = 7,         # fewer steps during training for speed
    pgd_alpha: float = 2/255
) -> dict[str, float]:
    """
    One adversarial training step (Madry et al., 2018 / AT-PGD-7).
    
    Trade-off: adversarial training achieves ~50-60% robust accuracy on CIFAR-10
    at epsilon=8/255 vs. 0% for standard training, but clean accuracy drops
    by ~10-15%. This accuracy-robustness trade-off is fundamental.
    """
    model.train()
    
    # Generate adversarial examples with frozen model (no_grad for efficiency)
    with torch.no_grad():
        model.eval()
    
    adv_images = pgd_attack(model, images, labels,
                             epsilon=epsilon, alpha=pgd_alpha,
                             num_steps=pgd_steps, random_start=True)
    
    model.train()
    
    # Update model on adversarial examples
    optimizer.zero_grad()
    output = model(adv_images)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    optimizer.step()
    
    # Evaluate on clean images too (optional: mix clean + adversarial)
    with torch.no_grad():
        clean_output = model(images)
        clean_acc = (clean_output.argmax(1) == labels).float().mean()
        adv_acc = (output.argmax(1) == labels).float().mean()
    
    return {
        "loss": loss.item(),
        "clean_acc": clean_acc.item(),
        "robust_acc": adv_acc.item()
    }
```

## Certified Defenses: Randomized Smoothing

Adversarial training provides empirical robustness with no guarantees. **Randomized smoothing** (Cohen et al., 2019) provides **certified** robustness: a provable guarantee that no $\ell_2$ perturbation within radius $R$ can change the model's prediction.

The key idea: define a smoothed classifier $g$ from a base classifier $f$ by averaging predictions over Gaussian noise:

$$g(x) = \underset{c}{\arg\max}\;\; P(f(x + \mathcal{N}(0, \sigma^2 I)) = c)$$

If class $c_A$ is returned with probability $p_A$ and all other classes with probability at most $p_B = 1 - p_A$, then $g$ is certifiably robust within $\ell_2$ radius:

$$R = \frac{\sigma}{2}\!\left(\Phi^{-1}(p_A) - \Phi^{-1}(p_B)\right)$$

where $\Phi^{-1}$ is the inverse standard normal CDF.

```python
import numpy as np
from scipy.stats import norm

class RandomizedSmoothing:
    """
    Certified ℓ2 robustness via randomized smoothing (Cohen et al., 2019).
    
    The smoothed classifier g certifies that no perturbation with ℓ2 norm ≤ R
    can cause a different prediction. Certification is statistical (uses sampling)
    and valid with confidence 1 - alpha.
    """
    def __init__(self, base_classifier, sigma: float = 0.25, num_samples: int = 1000,
                 alpha: float = 0.001):
        self.base = base_classifier
        self.sigma = sigma           # noise std (larger = larger radius, less accuracy)
        self.num_samples = num_samples
        self.alpha = alpha           # failure probability for certification

    @torch.no_grad()
    def certify(self, x: torch.Tensor, n_certify: int = 10000) -> tuple[int, float]:
        """
        Returns (predicted class, certified radius) or (-1, 0) if abstain.
        Uses Clopper-Pearson confidence interval for p_A lower bound.
        """
        # Sample predictions
        x_rep = x.unsqueeze(0).repeat(n_certify, 1, 1, 1)
        noise = torch.randn_like(x_rep) * self.sigma
        noisy = (x_rep + noise).clamp(0, 1)
        
        with torch.no_grad():
            preds = self.base(noisy).argmax(dim=1)
        
        # Count votes
        num_classes = 10  # CIFAR-10
        counts = torch.bincount(preds, minlength=num_classes)
        top1_count = counts.max().item()
        top1_class = counts.argmax().item()
        top2_count = counts.topk(2).values[-1].item()
        
        # Clopper-Pearson lower bound on p_A (1-sided)
        p_A_lower = self._proportion_confint_lower(top1_count, n_certify)
        
        if p_A_lower <= 0.5:
            return -1, 0.0   # abstain: cannot certify
        
        # Certified radius under ℓ2 norm
        radius = self.sigma * norm.ppf(p_A_lower)
        return top1_class, radius

    def _proportion_confint_lower(self, k: int, n: int) -> float:
        """Lower Clopper-Pearson confidence bound on p = k/n."""
        from scipy.stats import beta
        return beta.ppf(self.alpha, k, n - k + 1)
```

## Distributionally Robust Optimization

DRO trains models to minimize worst-case loss over an **uncertainty set** of distributions near the training distribution, rather than minimizing average training loss:

$$\min_\theta \max_{Q \in \mathcal{U}(P)} \mathbb{E}_{(x,y) \sim Q}\!\left[\mathcal{L}(f_\theta(x), y)\right]$$

**Group DRO** (Sagawa et al., 2020) defines the uncertainty set as a mixture over predefined groups (e.g., demographic subpopulations) and minimizes the worst-group loss:

```python
class GroupDROTrainer:
    """
    Group Distributionally Robust Optimization.
    
    Minimizes worst-group loss rather than average loss.
    Assigns higher weight to groups with currently high loss.
    
    Motivating example: a model for medical diagnosis that achieves
    95% average accuracy but 60% accuracy on minority demographic groups.
    Group DRO lifts the worst-group accuracy at some cost to average accuracy.
    """
    def __init__(self, model, optimizer, num_groups: int, eta: float = 0.01):
        self.model = model
        self.optimizer = optimizer
        self.num_groups = num_groups
        # Group weights: exponential moving average of group losses
        self.group_weights = torch.ones(num_groups) / num_groups

    def step(self, images: torch.Tensor, labels: torch.Tensor,
             group_ids: torch.Tensor) -> dict:
        """
        group_ids: (B,) integer group membership for each sample
        Group weights are updated online to up-weight high-loss groups.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(images)
        
        # Per-sample loss
        per_sample_loss = F.cross_entropy(outputs, labels, reduction="none")
        
        # Per-group mean loss
        group_losses = torch.zeros(self.num_groups, device=images.device)
        for g in range(self.num_groups):
            mask = (group_ids == g)
            if mask.sum() > 0:
                group_losses[g] = per_sample_loss[mask].mean()
        
        # Update group weights: up-weight high-loss groups
        self.group_weights = self.group_weights * torch.exp(0.01 * group_losses.detach())
        self.group_weights = self.group_weights / self.group_weights.sum()
        
        # Weighted loss: maximize over groups → focus on worst group
        loss = (self.group_weights * group_losses).sum()
        
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "group_losses": group_losses.detach().tolist(),
            "group_weights": self.group_weights.tolist(),
            "worst_group_loss": group_losses.max().item()
        }
```

## Robustness Benchmarks

| Defense | CIFAR-10 Clean Acc | CIFAR-10 Robust Acc (ε=8/255 ℓ∞) | Guarantee |
| --- | --- | --- | --- |
| Standard training | 95% | 0% | None |
| FGSM-AT | 83% | 40% | Empirical |
| PGD-AT (Madry) | 84% | 56% | Empirical |
| TRADES | 84% | 59% | Empirical |
| Randomized smoothing (σ=0.25) | 74% | Certifies ℓ2 radius | Certified ℓ2 |
| Diffusion-based purification | 88% | 71% | Empirical |

The accuracy-robustness trade-off is a fundamental tension: current methods cannot simultaneously achieve both maximum clean accuracy and maximum adversarial robustness. Active research areas include finding better training objectives, using diffusion models for adversarial purification, and developing architectures with intrinsic robustness properties.
