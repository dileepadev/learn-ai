---
title: Classifier-Free Guidance
description: Understand classifier-free guidance (CFG) — how it steers generative models toward conditional outputs without an external classifier, covering guidance scale, negative prompting, CFG for diffusion and language models, CFG++, and the quality-diversity trade-off in conditional generation.
---

**Classifier-free guidance (CFG)** is a sampling technique that dramatically improves conditional generation quality in diffusion models and large language models. It allows a single generative model to simultaneously behave as an unconditional generator and a conditional generator, using the difference between these two behaviors to amplify adherence to the conditioning signal — without ever training or querying a separate classifier model.

## Background: Classifier Guidance

To understand CFG, it helps to first understand the problem it solved. Early conditional diffusion models were guided using an **external classifier** (Dhariwal and Nichol, 2021): given a noisy image $x_t$ and a label $c$, a classifier $p_\phi(c \mid x_t)$ was trained to predict the label at every noise level. During sampling, the score function was modified:

$$\tilde{\nabla}_x \log p(x_t \mid c) = \nabla_x \log p(x_t) + \gamma \nabla_x \log p_\phi(c \mid x_t)$$

The gradient $\nabla_x \log p_\phi(c \mid x_t)$ pushes each denoising step toward inputs that the classifier assigns high probability to label $c$. The scale $\gamma > 1$ amplifies this push, trading sample diversity for class fidelity.

This approach required training a separate classifier at every noise level — a substantial overhead — and the classifier's gradients in high-noise regimes could be misleading.

## Classifier-Free Guidance

**CFG** (Ho and Salimans, 2022) eliminates the external classifier by training a **single model** to serve both conditional and unconditional roles simultaneously:

During training, the conditioning signal $c$ is **randomly dropped** — replaced with a null token $\emptyset$ — with probability $p_\text{uncond}$ (typically 10–20%). This means the same model learns:

- $\epsilon_\theta(x_t, c)$: conditional score estimate given prompt/class $c$.
- $\epsilon_\theta(x_t, \emptyset)$: unconditional score estimate.

During sampling, the **guided score** is computed as:

$$\tilde{\epsilon}(x_t, c) = \epsilon_\theta(x_t, \emptyset) + \gamma \cdot \left[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\right]$$

Equivalently:

$$\tilde{\epsilon}(x_t, c) = (1 - \gamma)\,\epsilon_\theta(x_t, \emptyset) + \gamma\,\epsilon_\theta(x_t, c)$$

The **guidance scale** $\gamma$ controls the trade-off:

- $\gamma = 0$: fully unconditional generation (ignores $c$).
- $\gamma = 1$: standard conditional generation.
- $\gamma > 1$: over-amplified conditioning — stronger adherence to $c$, lower diversity.

Typical guidance scales for image generation: $\gamma \in [7, 14]$ for SDXL and Stable Diffusion; $\gamma \in [3, 5]$ for class-conditional ImageNet generation.

## Geometric Interpretation

CFG can be interpreted as moving the sampling trajectory in the direction of **increasing log-likelihood ratio** between the conditional and unconditional models:

$$\tilde{\nabla} \log p \propto \nabla \log p(x_t \mid c) - \nabla \log p(x_t) = \nabla \log \frac{p(x_t \mid c)}{p(x_t)} = \nabla \log p(c \mid x_t)$$

By Bayes' theorem, $p(c \mid x_t) = p(x_t \mid c) \cdot p(c) / p(x_t)$, so the score difference approximates the gradient of an **implicit classifier** $p(c \mid x_t)$. CFG uses the diffusion model itself to implicitly compute classifier gradients — no external classifier needed.

## Quality-Diversity Trade-Off

The guidance scale $\gamma$ introduces a fundamental trade-off between **sample quality** (adherence to conditioning) and **sample diversity** (coverage of the conditional data distribution):

- **Low $\gamma$**: diverse but sometimes off-prompt or low-quality samples.
- **High $\gamma$**: highly prompt-adherent, photorealistic samples, but reduced diversity — the model tends to generate "typical" or "canonical" versions of the prompt, avoiding unusual but valid interpretations.

This is quantified by the **FID vs. CLIP score trade-off curve**: as $\gamma$ increases, CLIP score (text-image alignment) increases while FID (diversity) worsens. The Pareto frontier of this trade-off defines the achievable quality-diversity combinations for a given model.

## Negative Prompting

A powerful application of CFG is **negative prompting**: instead of using the null token $\emptyset$ as the unconditional condition, the unconditional model is conditioned on a **negative prompt** $c^-$ describing undesirable content:

$$\tilde{\epsilon}(x_t, c, c^-) = \epsilon_\theta(x_t, c^-) + \gamma \cdot \left[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, c^-)\right]$$

This pushes sampling toward $c$ and **away from** $c^-$. Common negative prompts include:

- `"blurry, low quality, artifacts, distorted"` — avoids low-quality outputs.
- `"text, watermark, signature"` — avoids common image artifacts.
- Specific content descriptors — stylistic control over what **not** to generate.

Negative prompting has become a standard technique in image generation interfaces (Stable Diffusion WebUI, ComfyUI) and is often as important as the positive prompt for achieving desired results.

## CFG for Text Generation: FUDGE and Others

CFG's principle applies to language models, though with important differences. For autoregressive LMs:

$$\log \tilde{p}(y_t \mid y_{<t}, c) \propto \log p_\theta(y_t \mid y_{<t}) + \gamma \cdot \left[\log p_\theta(y_t \mid y_{<t}, c) - \log p_\theta(y_t \mid y_{<t})\right]$$

This interpolates between the unconditional and conditional next-token distributions at each step. Applied during inference:

- Amplifies conditioning signals (style, topic, format constraints).
- Reduces the probability of tokens that the conditional model prefers but the unconditional does not — effectively sharpening the conditional distribution.

**FUDGE** (Yang and Klein, 2021) implements a related idea using a future discriminator for text generation. **CFG for language models** (Sanchez et al., 2023) demonstrated that CFG improves instruction following in LLMs beyond standard conditional sampling.

## CFG++

**CFG++** (Chung et al., 2024) identified a subtle problem with standard CFG: the guided score function can leave the **data manifold** — the denoising trajectory visits points in noise-space that are not reachable under any natural data distribution, leading to artifacts.

CFG++ corrects this by projecting the guidance step back onto the learned manifold at each denoising step:

$$\tilde{x}_{t-1} = D_\theta(x_t, c) + \lambda \left[ D_\theta(x_t, c) - D_\theta(x_t, \emptyset) \right]$$

where $D_\theta$ is the denoiser (posterior mean estimate) rather than the score. This subtly different formulation keeps the trajectory within the data manifold while maintaining the quality amplification of guidance. CFG++ requires a lower guidance scale ($\lambda \approx 0.6$ vs. $\gamma \approx 7.5$) to achieve equivalent quality, reducing artifacts.

## Multi-Condition Guidance

CFG naturally extends to **multiple conditioning signals** using composition:

$$\tilde{\epsilon} = \epsilon_\theta(x_t, \emptyset) + \gamma_1 \left[\epsilon_\theta(x_t, c_1) - \epsilon_\theta(x_t, \emptyset)\right] + \gamma_2 \left[\epsilon_\theta(x_t, c_2) - \epsilon_\theta(x_t, \emptyset)\right]$$

where $c_1$ and $c_2$ are independent conditions (e.g., style and content). Different guidance scales $\gamma_1, \gamma_2$ weight each condition independently. This enables:

- Style + content control in image generation.
- Multi-attribute conditioning in music generation.
- Subject + environment composition in 3D generation.

## Practical Implementation

In practice, CFG requires **two forward passes** through the model at each denoising step: one with the conditioning $c$ and one with the null condition $\emptyset$. This doubles inference cost compared to unconditional sampling.

**Batched inference**: the two passes are batched together (batch size 2N for N samples) and the CFG combination is computed after both passes. This is memory-intensive but GPU-efficient.

**Distilled CFG**: several methods distill the CFG-guided model into a single forward pass:

- **Consistency Models** with CFG distillation: single-step sampling that approximates multi-step CFG.
- **Latent Consistency Models (LCM)**: distilled guided SDXL with 4-step inference.
- **Score Distillation Sampling (SDS)** for 3D: uses CFG gradients to optimize a 3D representation.

## Role of CFG Drop Probability

The training drop probability $p_\text{uncond}$ controls how much the model learns as an unconditional model:

- **Too low** ($p_\text{uncond} < 0.05$): the model rarely sees $\emptyset$ and cannot produce good unconditional samples, degrading the CFG direction signal.
- **Too high** ($p_\text{uncond} > 0.30$): the model becomes too focused on unconditional generation and loses conditional fidelity.
- **Standard practice**: $p_\text{uncond} = 0.10$ to $0.15$ for image generation; up to $0.20$ for video and audio.

## Summary

Classifier-free guidance is the standard technique for high-quality conditional generation in diffusion models. By training a single model to simultaneously handle conditional and unconditional cases — through random dropout of the conditioning signal — CFG enables efficient inference-time guidance without external classifiers. The guidance scale $\gamma$ controls the quality-diversity trade-off, with typical values of 7–14 for image generation. Negative prompting extends CFG to steer away from undesirable attributes. CFG++ addresses manifold deviation artifacts. For language models, CFG sharpens conditional distributions at each decoding step. The primary cost — doubled forward passes — is mitigated by distillation methods like LCM and consistency models. CFG has become the central pillar of modern controllable generative AI.
