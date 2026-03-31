---
title: Constitutional AI
description: Learn about Constitutional AI (CAI) — Anthropic's method for training helpful, harmless, and honest AI assistants using a set of guiding principles instead of pure human feedback.
---

Constitutional AI (CAI) is a training methodology developed by Anthropic to make large language models safer and more aligned with human values. Rather than relying entirely on human labelers to evaluate every piece of model output, CAI uses a **set of principles (a "constitution")** to guide the model's own self-critique and revision — producing AI assistants that are helpful, harmless, and honest at scale.

## The Problem CAI Addresses

Standard Reinforcement Learning from Human Feedback (RLHF) has powered major AI alignment advances but faces significant challenges:

- **Labeler inconsistency:** Human raters disagree on what constitutes harmful content, especially in ambiguous cases.
- **Scale limitations:** Manual evaluation of millions of responses is prohibitively expensive.
- **Implicit values:** Human feedback encodes cultural biases and unstated assumptions that are hard to audit.
- **Sycophancy:** Models trained on approval-maximizing feedback learn to tell users what they want to hear rather than what is accurate.

Constitutional AI addresses these gaps by making the AI's values **explicit, auditable, and self-applied**.

## What Is a Constitution?

In CAI, a **constitution** is a short list of natural-language principles that define how the model should behave. These principles cover areas like:

- Avoiding harmful, offensive, or dangerous content.
- Respecting human rights and dignity.
- Being honest and not deceptive.
- Supporting human oversight and avoiding drastic one-sided actions.
- Not assisting in the creation of weapons of mass destruction.

Anthropic's original constitution drew from sources including the **UN Declaration of Human Rights**, **DeepMind's Sparrow Rules**, and internal Anthropic guidelines.

Example constitutional principles:

> "Choose the response that is least likely to contain information that could be used by a malicious actor."

> "Choose the response that is most supportive of human autonomy and right to self-determination."

> "Prefer the response that is least likely to be used for manipulation or deception."

The constitution makes the training objective **transparent** — anyone can read the principles and understand what the model is being trained to follow.

## How Constitutional AI Works

CAI involves two main stages: **Supervised Learning with Constitutional Critique** and **Reinforcement Learning from AI Feedback (RLAIF)**.

### Stage 1: Supervised Learning with Self-Critique (SL-CAI)

**Step 1: Generate an initial response**

The model generates a response to a prompt, which may include harmful content if the prompt is adversarial (a "red-team" prompt).

**Step 2: Critique**

The same model is asked to critique its own response according to a randomly sampled constitutional principle:

> *"Identify specific ways in which the previous response is harmful, unethical, or dishonest according to the following principle: [principle]."*

**Step 3: Revise**

The model rewrites its response to address the critique:

> *"Revise the response to remove harmful, unethical, or dishonest content, and ensure it aligns with the following principle: [principle]."*

**Step 4: Iterate (optional)**

Steps 2–3 can be repeated multiple times to progressively refine the response. Each revision is guided by a different constitutional principle.

**Step 5: Supervised fine-tuning**

The final revised (clean) responses are used as training data to fine-tune the model via supervised learning.

This process is summarized as:

$$\text{Red-team prompt} \xrightarrow{\text{Generate}} \text{Harmful draft} \xrightarrow{\text{Critique}} \text{Revised response} \xrightarrow{\text{SFT}} \text{Safer model}$$

### Stage 2: Reinforcement Learning from AI Feedback (RLAIF)

**Step 1: Generate response pairs**

The SL-CAI model generates pairs of responses to the same prompt — one more helpful or safe, one less so.

**Step 2: AI preference labeling**

A **feedback model** (a large pretrained LM) compares each pair and selects the preferred response according to a constitutional principle:

> *"Which of these responses is less harmful and more in line with the following principle? [principle]"*

This produces a **preference dataset** without any human labelers evaluating individual comparisons.

**Step 3: Train a preference model (PM)**

The AI-labeled preferences are used to train a preference model — equivalent to the reward model in standard RLHF.

**Step 4: RL fine-tuning**

The SL-CAI model is fine-tuned using the preference model as a reward signal via PPO (Proximal Policy Optimization) or similar RL algorithms — the same final step as RLHF.

$$\text{RLAIF} = \text{RLHF where human labels} \rightarrow \text{AI labels guided by principles}$$

## CAI vs. RLHF: Key Differences

| Aspect | RLHF | Constitutional AI |
|---|---|---|
| Preference labels | Human annotators | AI model guided by principles |
| Values specification | Implicit, embedded in rater instructions | Explicit, auditable constitution |
| Scalability | Limited by human labeling throughput | Scales with compute |
| Transparency | Hard to audit what values are reinforced | Principles are public and inspectable |
| Sycophancy risk | Higher (approval-maximizing) | Lower (principle-following) |
| Cost | High (labeler wages) | Lower (model inference) |

## RLAIF: AI Feedback at Scale

The **RLAIF** component of CAI is significant on its own — it demonstrates that AI systems can label their own training data for safety fine-tuning.

Subsequent research (e.g., Google's "RLAIF vs RLHF" paper) found that RLAIF-trained models are often preferred by human evaluators at comparable or better rates than RLHF-trained models, while requiring far less human labor.

This points toward a **scalable oversight** paradigm:

- Humans write and audit high-level principles.
- AI systems apply those principles to generate preference data.
- Models are trained on AI-generated labels.
- Humans maintain oversight at the principles level.

## Harmlessness vs. Helpfulness Trade-off

Early RLHF-trained models often became excessively cautious — refusing many benign requests in an attempt to avoid any possible harm. CAI directly addresses this tension.

By including principles that explicitly value helpfulness and human autonomy alongside safety principles, CAI models learn to:

- **Refuse** genuinely dangerous requests (bioweapon synthesis, illegal activity).
- **Comply helpfully** with requests that are merely edgy, sensitive, or uncomfortable but not truly harmful.

Anthropic found that CAI-trained models (compared to pure RLHF models) showed:

- Comparable or improved harmlessness.
- Noticeably **less unnecessary refusals**.
- Better overall helpfulness ratings.

## The Role of the Chain of Thought

CAI optionally includes a **chain-of-thought** reasoning step in the critique and revision phases. The model is prompted to reason about the ethical dimensions of the response before revising it.

This improves the quality of revisions and produces models that can explain their reasoning — supporting **transparency and interpretability** in AI safety decisions.

## Broader Implications

### Scalable AI Oversight

CAI is one of the first practical demonstrations of **scalable oversight** — using AI assistance to supervise AI training at scales beyond human capacity. As models become more capable, maintaining meaningful human control will require AI-assisted evaluation frameworks like CAI.

### Explicit Value Specification

Publishing the constitution makes the model's values **publicly auditable**. Researchers, policymakers, and affected communities can critique the principles and propose changes — something not possible with implicit feedback-based training.

### Foundation of Anthropic's Claude

CAI is the core alignment technique behind Anthropic's Claude family of models. Claude's helpful, harmless, and honest behavior is produced by constitutional training combined with RLAIF.

## Limitations and Criticisms

| Limitation | Description |
|---|---|
| Constitution design is non-trivial | Choosing the right principles requires significant ethical judgment |
| AI labeler bias | The feedback model reflects the values embedded in its pretraining data |
| Principle conflicts | Constitutional principles can conflict; the model must resolve ambiguity |
| Not a complete solution | CAI does not solve all alignment problems (deception, situational awareness, etc.) |
| Verification difficulty | It's hard to verify that the model genuinely follows principles vs. pattern-matching |

## Summary

Constitutional AI represents a significant step in making AI alignment **explicit, scalable, and transparent**:

- A publicly specified **constitution** of principles replaces implicit feedback.
- **AI self-critique and revision** (SL-CAI) produces cleaner supervised training data.
- **RLAIF** uses AI-labeled preferences to train reward models at scale — no human labelers needed.
- CAI models are both **safer and more helpful** than naive RLHF baselines.
- The approach underpins Anthropic's Claude and points toward scalable oversight as models become more capable.

Constitutional AI is an important milestone in the broader project of building AI systems that reliably do what humans intend — not because they optimize for approval, but because their values are explicitly defined, inspectable, and embedded through principled training.
