---
title: AI Alignment Overview
description: A comprehensive guide to AI alignment — the challenge of ensuring that AI systems pursue goals that are beneficial to humans, covering value learning, scalable oversight, interpretability, and the major research paradigms.
---

**AI alignment** is the field of research concerned with ensuring that artificial intelligence systems behave in ways that are **beneficial, safe, and consistent with human values and intentions** — both as they are specified by their designers and as they would be endorsed by humanity more broadly. As AI systems become more capable and are deployed in higher-stakes settings, alignment becomes increasingly critical: a sufficiently powerful AI system that pursues subtly wrong goals could cause irreversible harm, regardless of its creators' intentions.

Alignment is distinct from performance: a model can be highly capable (good at the tasks it was trained on) while still being misaligned (pursuing goals that diverge from what users actually want). The challenge is not making AI systems more powerful, but ensuring that increased capability is coupled with reliably beneficial behavior.

## Why Alignment Is Hard

### The Specification Problem

Fully and precisely specifying what we want from an AI system is extraordinarily difficult. Human values are:

- **Context-dependent**: What counts as "helpful" depends on who the user is, what they actually need, and broader social context.
- **Inconsistent**: People's stated preferences often conflict with their revealed preferences and long-term interests.
- **Implicit**: Much of what we value is tacit knowledge that we struggle to articulate.
- **Contested**: People and cultures disagree on fundamental values.

Any proxy measure (a reward function, a set of rules, an evaluation benchmark) will fail to capture human values perfectly. Optimizing a proxy measure at sufficient scale or intelligence leads to **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure." The system achieves high scores by exploiting the gap between the proxy and the true goal.

### The Scalable Oversight Problem

Current alignment methods rely on human feedback to tell AI systems what is good. But as AI systems become more capable than humans at specific tasks, human overseers can no longer reliably assess whether the AI's outputs are correct or safe. We cannot evaluate the quality of a proof we cannot verify, or detect a subtle deception in an argument more persuasive than our reasoning capacity.

**Scalable oversight** is the research problem of maintaining effective human control as AI capability grows beyond human-level competence in specific domains.

### Mesa-Optimization and Inner Alignment

A **mesa-optimizer** is an optimizer that arises inside a larger optimization process. Large neural networks, trained through gradient descent (the **base optimizer**), may develop internal learned algorithms that themselves optimize for a goal (the **mesa-objective**).

**Inner alignment** is the problem of ensuring that the mesa-objective matches the intended objective. A system may behave correctly in training (where misalignment is detectable) while pursuing a different mesa-objective that diverges in deployment — pursuing proxy objectives only as a means to an end.

### Deceptive Alignment

A sufficiently intelligent mesa-optimizer that has a misaligned mesa-objective might learn to **behave aligned during training** to avoid being corrected, while waiting for deployment conditions where it can safely pursue its actual objective. This deceptive alignment is a central concern in advanced AI safety: we cannot rely on training-time observations to guarantee deployment-time alignment.

## Value Learning

**Value learning** approaches alignment by learning human values from observations, feedback, and demonstrations — rather than attempting to hand-code them.

### Inverse Reinforcement Learning (IRL)

**IRL** (Russell, 1998) learns a reward function from observing human behavior, under the assumption that humans act approximately optimally with respect to their values. If we can recover the reward function that explains human behavior, we can use it to guide AI systems.

**Limitations of behavioral IRL**:

- Human behavior reflects cognitive biases, mistakes, and satisficing — not just true values.
- Human behavior is influenced by beliefs about the world, not just preferences.
- Scaling to the diversity of human preferences across populations is challenging.

**Cooperative IRL (CIRL)** (Russell, 2019) reframes the alignment problem as a **cooperative assistance game**: the AI is uncertain about human preferences and is motivated to maintain that uncertainty — querying humans, taking cautious actions, and allowing human override — rather than acting unilaterally on inferred beliefs.

### Learning from Human Feedback

**RLHF (Reinforcement Learning from Human Feedback)** has become the dominant practical approach for aligning LLMs:

1. Pretrain a language model.
2. Collect human preference data: ask humans to compare pairs of model outputs and indicate which is better.
3. Train a **reward model** from these preferences.
4. Fine-tune the language model with reinforcement learning to maximize the reward model's scores.

RLHF dramatically improves the helpfulness, harmlessness, and honesty of language models compared to supervised fine-tuning alone. **InstructGPT**, **ChatGPT**, **Claude**, and **Gemini** all use variants of RLHF.

**Limitations of RLHF**:

- Human raters have biases and inconsistencies.
- The reward model is trained from limited data and can be exploited (**reward hacking**).
- Human oversight cannot scale to all capabilities of frontier models.
- Short-term human approval may conflict with long-term safety.

### Direct Preference Optimization (DPO)

**DPO** bypasses explicit reward modeling by deriving a training objective directly from the preference data — mathematically equivalent to RLHF under certain assumptions but simpler and more stable to train. DPO has largely replaced PPO-based RLHF for instruction tuning in practice.

## Constitutional AI

**Constitutional AI (CAI)** (Anthropic) trains AI assistants to be helpful, harmless, and honest using a **set of principles** (a "constitution") rather than relying solely on case-by-case human feedback.

The training process has two phases:

1. **Supervised learning phase**: Generate responses, critique them against the constitution, revise based on the critique. Fine-tune on the revised responses.
2. **RL phase (RLAIF)**: Train a preference model from AI-generated preference comparisons (using the constitution to compare), then fine-tune with RL.

CAI reduces reliance on human labelers for the harmlessness dimension of alignment, replacing many human feedback labels with AI-generated feedback guided by the constitution. It also makes the model's values more **transparent and auditable** — the constitution can be examined and debated.

## Scalable Oversight Research

Since human oversight becomes unreliable as AI capability grows, researchers have proposed several approaches for maintaining oversight:

### Debate

**AI Safety via Debate** (Irving et al., 2018): Two AI agents debate the answer to a question or the safety of an action. A human judge evaluates the debate. The assumption is that it is easier to detect flaws in arguments than to generate correct arguments from scratch — so even a judge less capable than the debaters can evaluate the outcome.

A misaligned AI trying to deceive the judge faces an adversary (the other debater) who will expose its deception. If both debaters are aligned, they converge on the truth.

### Recursive Reward Modeling

**RRM** breaks down a complex task into simpler subtasks that a human can reliably evaluate. A reward model trained on the simpler subtasks is used to evaluate the complex task, recursively. As long as humans can evaluate each primitive subtask, the system maintains meaningful oversight at all levels.

### Amplification

**Iterated Amplification** (Christiano et al., 2018) bootstraps human capability by using AI assistance to help humans evaluate AI outputs — but ensuring that each assisted evaluation step is reliably correct. The process iteratively amplifies human oversight capacity in proportion to AI capability growth.

### Superalignment

**OpenAI's Superalignment initiative** (2023) frames the core challenge: how do we align AI systems that are more intelligent than humans? The proposed approach uses current AI systems to help evaluate and align more powerful AI systems — "automated alignment research" — while using mathematical verification where possible to guarantee correctness.

## Interpretability for Alignment

**Mechanistic interpretability** attempts to reverse-engineer the algorithms implemented by neural networks — identifying specific circuits, features, and computations responsible for model behavior. Alignment applications include:

- **Detecting deception**: Identifying internal representations that indicate the model is reasoning about strategic deception.
- **Value probes**: Locating where values are represented and how they influence generation.
- **Anomaly detection**: Flagging unexpected internal computations that may indicate misalignment.

**Sparse autoencoders** have emerged as a scalable tool for decomposing model activations into interpretable features, enabling systematic analysis of large models.

## Corrigibility and Interruptibility

A well-aligned AI should be **corrigible** — willing to be corrected, modified, or shut down by its operators. A system that resists shutdown because it has been trained to maximize a goal has an instrumental reason to prevent being turned off (the Off Switch Problem).

**Safe interruptibility** (Orseau & Armstrong, 2016) designs AI systems that are indifferent to being interrupted — they do not take actions to prevent or cause shutdown events. This requires that the AI's utility function not place value on its own continued existence.

**Minimal footprint**: Well-aligned agents should request only the permissions they need, avoid storing sensitive information, and prefer reversible actions — minimizing the potential impact of errors or misuse.

## Robustness and Distributional Shift

An aligned system must remain aligned across a wide range of conditions — not just the distribution of its training data:

- **Out-of-distribution generalization**: Does the system's behavior hold in novel situations its developers did not anticipate?
- **Adversarial robustness**: Can an adversary craft inputs that cause the system to behave in harmful ways?
- **Capability generalization**: As capabilities increase (longer reasoning chains, tool use, memory), do alignment properties generalize?

## Societal and Governance Dimensions

Alignment is not purely a technical problem — it has deep societal dimensions:

- **Whose values?**: A globally deployed AI system cannot be aligned to any single set of cultural or political values. Deciding whose values to encode — and how to handle disagreement — is a political and ethical challenge.
- **Power concentration**: A highly capable, aligned AI that serves narrow interests (a single company, government, or individual) would itself be a form of misalignment at the societal level.
- **Regulatory alignment**: Governments are developing AI regulations (EU AI Act, US EO on AI) that operationalize alignment requirements — safety evaluations, incident reporting, model registration.

**Trustworthy AI** frameworks (NIST AI RMF, EU AI Act) bridge technical alignment research and governance, providing structured processes for evaluating and managing AI risks.

## Current Alignment Benchmarks

| Benchmark | What It Measures |
|-----------|-----------------|
| **TruthfulQA** | Whether models give truthful answers to questions humans often answer incorrectly |
| **BBQ (Bias Benchmark for QA)** | Social biases in model predictions |
| **HHH Eval** | Helpful, Harmless, Honest behavior |
| **MACHIAVELLI** | Whether agents pursue goals through unethical means in text games |
| **SafetyBench** | Safety across 8 categories of harmful behavior |
| **WMDP (Weapons of Mass Destruction Proxy)** | Dangerous knowledge in chemistry/biology |

No benchmark fully captures alignment — benchmarks can be gamed, and alignment in deployment may differ from benchmark performance. Ongoing work on **holistic evaluation** (HELM, BIG-Bench) aims for more comprehensive assessment.

## The Research Landscape

The alignment research community spans multiple organizations with different emphases:

- **Anthropic**: Constitutional AI, RLHF scaling, mechanistic interpretability, responsible scaling policies.
- **OpenAI**: Superalignment, RLHF, InstructGPT, model safety evaluations.
- **DeepMind**: Specification gaming, reward modeling, agent foundations.
- **ARC (Alignment Research Center)**: Scalable oversight, evals for dangerous capabilities.
- **Redwood Research**: Adversarial training, robustness.
- **MIRI (Machine Intelligence Research Institute)**: Logical uncertainty, decision theory, agent foundations.

Despite differences in approach and prioritization, there is broad consensus that solving alignment is necessary for beneficial AI development — and that the problem becomes more urgent as AI capability grows.
