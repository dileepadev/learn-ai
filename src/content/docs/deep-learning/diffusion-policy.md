---
title: Diffusion Policy for Robot Learning
description: Explore how diffusion models are applied to robot policy learning — enabling robots to generate smooth, multimodal action trajectories through iterative denoising, with superior performance on dexterous manipulation tasks compared to traditional behavior cloning methods.
---

**Diffusion Policy** applies the generative power of diffusion models to robot learning — specifically to the problem of **behavior cloning**, where a robot learns to perform tasks by imitating expert demonstrations. Rather than mapping observations directly to actions (as in standard imitation learning), Diffusion Policy treats action generation as a denoising process: the robot learns to iteratively refine random noise into a coherent action sequence conditioned on its current visual and proprioceptive observations.

Introduced by Chi et al. (2023) from MIT and Toyota Research Institute, Diffusion Policy demonstrated state-of-the-art performance on the RoboMimic and Block Push benchmarks, surpassing previous imitation learning approaches on 11 of 12 tasks — particularly excelling at dexterous, contact-rich manipulation where multiple valid action trajectories exist for a given state.

## Why Standard Behavior Cloning Fails

Traditional behavior cloning trains a policy $\pi_\theta(a|o)$ to directly predict actions from observations by minimizing mean squared error or cross-entropy loss over a dataset of expert demonstrations. This approach has fundamental limitations:

**Compounding errors**: Small deviations from the training distribution accumulate over an episode — the policy encounters states never seen during training, where its predictions degrade unpredictably. This is the classic **covariate shift** problem in imitation learning.

**Multimodality**: Many manipulation tasks have multiple valid strategies (e.g., approaching a cup from the left or right). MSE regression averages across these modes, producing a blend of strategies that satisfies neither — the averaging problem causes hesitant, blended motion.

**Action discontinuities**: Direct regression often produces jerky, high-variance actions rather than the smooth trajectories experts demonstrate — particularly problematic for contact-rich tasks where smooth force profiles matter.

Diffusion Policy addresses all three issues: it captures multimodal action distributions, generates smooth trajectories by predicting sequences rather than individual actions, and produces diverse but valid behaviors conditioned on the observation.

## Diffusion Models as Conditional Generative Models

In image generation, diffusion models learn to reverse a gradual noising process:
- **Forward process**: Gradually add Gaussian noise to a data sample $x_0$ over $T$ steps to produce $x_T \sim \mathcal{N}(0, I)$.
- **Reverse process**: Learn a denoising network $\epsilon_\theta(x_t, t)$ that predicts the noise, enabling iterative denoising from $x_T$ back to $x_0$.

For Diffusion Policy, the "image" is replaced by an **action sequence**: a chunk of $H_a$ future actions $A = [a_{t}, a_{t+1}, \ldots, a_{t+H_a-1}]$. The conditioning signal $O$ is a history of recent observations (images, robot joint positions, gripper state):

$$p_\theta(A^0 | O) = \int p(A^T) \prod_{k=1}^{T} p_\theta(A^{k-1} | A^k, O) \, dA^{1:T}$$

The denoising network $\epsilon_\theta(A^k, k, O)$ learns to predict the noise component at each step, conditioned on the observation history $O$.

## The Diffusion Policy Architecture

### Observation Encoding

Visual observations (RGB images, depth maps) are encoded using a pre-trained or jointly-trained CNN (ResNet or ViT). Proprioceptive observations (joint angles, velocities, gripper state) are encoded via MLPs. The concatenated observation embedding serves as the conditioning signal for the denoising network.

### Denoising Network Architectures

**CNN-based Diffusion Policy (CNN-DP)**:

A 1D convolutional network processes the noisy action sequence $A^k$ as a temporal signal:

```python
import torch
import torch.nn as nn

class ConditionalUnet1D(nn.Module):
    """1D U-Net for denoising action sequences conditioned on observations."""
    def __init__(self, action_dim, obs_dim, diffusion_step_embed_dim=256):
        super().__init__()
        # Encode diffusion timestep
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim)
        )
        # 1D U-Net with skip connections
        self.encoder = nn.ModuleList([
            ConditionalResidualBlock1D(action_dim, 256, cond_dim=obs_dim + diffusion_step_embed_dim),
            ConditionalResidualBlock1D(256, 512, cond_dim=obs_dim + diffusion_step_embed_dim),
        ])
        self.decoder = nn.ModuleList([
            ConditionalResidualBlock1D(512 + 256, 256, cond_dim=obs_dim + diffusion_step_embed_dim),
            ConditionalResidualBlock1D(256 + action_dim, action_dim, cond_dim=obs_dim + diffusion_step_embed_dim),
        ])
    
    def forward(self, noisy_actions, timestep, obs_cond):
        t_emb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([obs_cond, t_emb], dim=-1)
        # U-Net forward pass over the action sequence (temporal dimension)
        ...
        return denoised_actions
```

**Transformer-based Diffusion Policy (Trans-DP)**:

A transformer architecture attends over both the observation tokens and the noisy action tokens — enabling richer cross-modal attention between visual context and action trajectory. Transformer-DP handles variable-length observation histories and tends to outperform CNN-DP on tasks requiring detailed visual reasoning.

## Training Procedure

Training follows the standard DDPM (Denoising Diffusion Probabilistic Models) objective applied to action sequences:

```python
def compute_loss(model, batch, noise_scheduler):
    obs, actions = batch['obs'], batch['actions']  # actions: [B, H_a, action_dim]
    
    # Sample random noise and timesteps
    noise = torch.randn_like(actions)
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (actions.shape[0],))
    
    # Add noise to actions according to the noise schedule
    noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
    
    # Predict the noise that was added
    noise_pred = model(noisy_actions, timesteps, obs)
    
    # MSE loss between predicted and actual noise
    loss = F.mse_loss(noise_pred, noise)
    return loss
```

## Inference: Generating Actions

At deployment, actions are generated by iterative denoising:

```python
def generate_actions(model, obs, noise_scheduler, num_inference_steps=100):
    # Start from pure noise
    actions = torch.randn(1, H_a, action_dim)
    
    noise_scheduler.set_timesteps(num_inference_steps)
    
    for t in noise_scheduler.timesteps:
        # Predict noise
        with torch.no_grad():
            noise_pred = model(actions, t.unsqueeze(0), obs)
        
        # One denoising step
        actions = noise_scheduler.step(noise_pred, t, actions).prev_sample
    
    return actions  # [1, H_a, action_dim] — a chunk of H_a future actions
```

**Action chunking**: The policy predicts a chunk of $H_a = 16$ future actions at once. The robot executes the first few steps, then re-plans — balancing reactivity (replanning frequently) against temporal consistency (executing a coherent plan without interruption).

## DDIM Acceleration

Standard DDPM requires 100 denoising steps — too slow for real-time robot control. **DDIM** (Denoising Diffusion Implicit Models) accelerates inference to 10-20 steps with minimal quality loss, enabling control loops fast enough for manipulation (10-20 Hz policy frequency):

```python
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(num_train_timesteps=100)
scheduler.set_timesteps(10)  # Only 10 denoising steps at inference
```

## Key Results and Advantages

On the RoboMimic benchmark (real-world manipulation tasks):

| Method | Can (PH) | Lift (MH) | Square (PH) | Transport (PH) |
| --- | --- | --- | --- | --- |
| BC (MSE) | 74.5% | 40.0% | 18.0% | 8.0% |
| IBC | 62.0% | 28.0% | 0.8% | 0.4% |
| BESO | 71.0% | 52.0% | 32.0% | 11.0% |
| Diffusion Policy | **95.0%** | **78.0%** | **58.0%** | **29.5%** |

The largest gains appear on precision tasks (Square assembly, Transport) — exactly the dexterous, contact-rich tasks where multimodal action distributions and smooth trajectory generation matter most.

## Connection to Broader Robot Learning

**Diffusion Policy** belongs to a broader trend of applying generative models to robot policy learning:

- **ACT** (Action Chunked Transformers): Uses a conditional VAE to generate action chunks — similar insight (sequence prediction) without diffusion.
- **π0** (Physical Intelligence, 2024): Scales Diffusion Policy to a large vision-language-action model, enabling zero-shot generalization to novel tasks via language conditioning.
- **Octo** and **OpenVLA**: Pre-trained robot foundation models that incorporate diffusion-style action heads.
- **GROOT** and **UniSim**: Use world models alongside diffusion policies for planning.

Diffusion Policy represents a paradigm shift in behavior cloning: rather than regressing to a single action, the robot reasons over the space of possible action trajectories and selects the most appropriate — a far more robust and expressive approach to learned robot control.
