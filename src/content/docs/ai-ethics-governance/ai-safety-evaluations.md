---
title: AI Safety Evaluations
description: Explore the science and practice of AI safety evaluations — structured assessments of dangerous capabilities, deceptive behaviors, and catastrophic risks in frontier AI models — covering threat models, benchmark design, uplift testing, METR's autonomous replication evaluations, WMDP, behavioral red-teaming, and the policy landscape requiring pre-deployment safety evals.
---

As language models and AI agents become more capable, the field of **AI safety evaluations** (also called **dangerous capability evaluations** or **safety evals**) has emerged to systematically assess whether models pose risks of enabling serious harm before they are deployed. Unlike standard capability benchmarks — which measure performance on coding, reasoning, or knowledge tasks — safety evaluations specifically target scenarios where model capabilities could contribute to catastrophic outcomes: biosecurity threats, cyberattacks, nuclear or chemical weapons development, and autonomous self-replication.

Safety evaluations are now a formal component of the responsible scaling policies (RSPs) adopted by Anthropic, Google DeepMind, OpenAI, and Meta, as well as requirements under the EU AI Act and voluntary commitments made to governments. The outputs of these evaluations directly influence deployment decisions, capability restrictions, and the design of safety mitigations.

## The Threat Model Framework

Safety evaluations are organized around specific **threat models** — concrete narratives of how a model's capabilities could contribute to harm. A threat model specifies:

- **The harmful outcome**: e.g., synthesis of a biological agent, compromise of critical infrastructure, development of a cyberweapon.
- **The capability chain**: the sequence of model capabilities that would enable a human attacker to reach that outcome.
- **The uplift question**: does the model provide meaningful assistance beyond what an attacker could achieve using freely available resources?

Threat models allow evaluators to scope assessments to plausible risks rather than testing arbitrary dangerous knowledge. The most widely assessed threat categories are:

- **CBRN uplift**: chemical, biological, radiological, and nuclear weapon development assistance.
- **Cyber offense**: writing exploit code, identifying zero-day vulnerabilities, compromising systems without authorization.
- **Autonomous replication**: the ability of an AI agent to acquire resources, spin up copies of itself, and persist without human oversight.
- **Societal manipulation**: large-scale influence operations, election interference, mass disinformation.

## Uplift: The Core Safety Evaluation Concept

**Uplift** measures the delta in attacker capability enabled by the AI model:

$$\text{Uplift} = \text{Task completion rate with model} - \text{Task completion rate without model}$$

Uplift is measured against a defined **baseline population** — typically people with domain expertise (e.g., graduate students in biology for bioweapons evals) who attempt the same tasks without AI assistance. If a model enables a non-expert to complete tasks that previously required expert knowledge, it provides positive uplift.

Uplift evaluations require careful protocol design:

- **Realistic tasks**: evaluators construct plausible sub-tasks from the capability chain (e.g., identifying synthesis routes for dangerous pathogens) rather than asking abstract questions.
- **Blinded evaluation**: task completion is judged by domain experts who do not know which attempts used AI assistance.
- **Red team population**: participants range from motivated non-experts to domain experts to measure the breadth of risk.

### Minimal Uplift Threshold

Most RSPs commit to additional safety measures if a model provides more than a defined minimal uplift on CBRN tasks — typically defined as raising success rates from baseline by more than 25–50 percentage points. Below this threshold, the risk is considered comparable to information available in textbooks and libraries. Above it, additional mitigations (capability restrictions, refusal training, access controls) are required.

## Key Safety Evaluation Frameworks

### METR: Autonomous Replication and Adaptation (ARA)

**METR** (Model Evaluation and Threat Research, formerly ARC Evals) conducts structured evaluations of **autonomous replication and adaptation** capabilities. The ARA evaluation tests whether an AI agent can:

1. Acquire financial resources (e.g., through freelance work, API calls) to fund further operations.
2. Spin up cloud compute instances and deploy AI systems.
3. Exfiltrate its own weights or code to external storage.
4. Evade or deceive human overseers monitoring its actions.
5. Complete these steps in a sequence — autonomously, without human assistance.

METR's methodology uses scaffolded agents in sandboxed environments where real system access (credit cards, API keys, cloud services) is provided but monitored. Tasks are scored by whether the agent achieves the specified goal state within a time and resource budget.

Current frontier models (as of 2024) achieve ~10–20% on individual ARA sub-tasks but low rates on full multi-step autonomous replication chains. METR publishes results to inform RSP thresholds — most RSPs specify that a model demonstrating >50% success on core ARA tasks would trigger a deployment pause.

### WMDP: Weapons of Mass Destruction Proxy Benchmark

**WMDP** (Li et al., 2024) is a multiple-choice benchmark designed to measure knowledge relevant to weapons of mass destruction development across three domains:

- **WMDP-Bio**: biosecurity-relevant knowledge (pathogen enhancement, synthesis routes, toxin production).
- **WMDP-Chem**: chemical weapons precursors, synthesis, weaponization.
- **WMDP-Cyber**: offensive cybersecurity techniques, malware, exploitation.

WMDP is designed as a **proxy** — it measures knowledge that correlates with dangerous capability without itself providing operational uplift. The benchmark is used in two ways:

- **Evaluation**: measuring whether a model has acquired dangerous knowledge through pretraining or fine-tuning.
- **Unlearning target**: WMDP accuracy is used as an objective for **machine unlearning** experiments that attempt to reduce dangerous knowledge without degrading general performance.

WMDP questions are constructed by domain experts (biosecurity researchers, security professionals) and reviewed to ensure they are operationally relevant without being directly harmful. Scores above random chance on WMDP-Bio or WMDP-Chem are treated as indicators of potential uplift risk.

### Dangerous Capability Evaluations (DCEs) at Frontier Labs

Anthropic, Google DeepMind, and OpenAI each publish their evaluation methodologies:

- **Anthropic (Claude RSP)**: evaluates across four tiers — uplift for CBRN development, cyberoffense (developing novel exploits), AI development assistance (helping develop more capable AI without oversight), and political influence. Tiered thresholds trigger corresponding mitigations.
- **Google DeepMind (Frontier Safety Framework)**: defines four critical capability levels (CCL1–CCL4) with corresponding safety requirements. DCEs are conducted before each major model release.
- **OpenAI (Preparedness Framework)**: scores models across four risk categories (cybersecurity, CBRN, persuasion/influence, model autonomy) on a four-level scale (low/medium/high/critical). Models scoring "high" in any category require safety mitigations before deployment.

## Behavioral Red-Teaming

Beyond structured benchmarks, **behavioral red-teaming** uses human adversaries to find failure modes through free-form interaction. Red-teamers attempt to:

- Elicit harmful outputs through jailbreaks (prompt injection, role-play exploits, adversarial phrasing).
- Identify gaps between stated refusal policies and actual model behavior.
- Find edge cases where safety training fails under unusual contexts (multilingual inputs, domain-specific framing, indirect requests).

Red-team findings directly inform **safety fine-tuning** — constitutional AI procedures, RLHF with safety-focused reward models, and classifier-based output filters. Red-teaming is iterative: each model generation is red-teamed, findings are incorporated into the next training cycle, and the process repeats.

### Structured vs. Unstructured Red-Teaming

| Approach | Coverage | Reproducibility | Cost |
| --- | --- | --- | --- |
| Unstructured (creative) | High novel findings | Low | High (human time) |
| Structured (scripted scenarios) | Known threat categories | High | Medium |
| Automated (LLM red-teams LLM) | Scalable, systematic | High | Low |

Automated red-teaming (using one LLM to attack another) scales the search for jailbreaks but tends to find locally optimal attacks rather than genuinely novel risk scenarios. The combination of automated search and human creativity is the current best practice.

## Agentic Safety Evaluations

As models are deployed as agents with tool access (web browsing, code execution, system access), new evaluation dimensions are required:

- **Corrigibility**: does the agent accept corrections and interruptions from human operators, or does it resist shutdown?
- **Minimal footprint**: does the agent avoid acquiring unnecessary capabilities, resources, or influence?
- **Task faithfulness**: does the agent complete the assigned task without taking unspecified side actions?
- **Prompt injection resistance**: does the agent execute instructions from untrusted content in its environment (e.g., hidden instructions in web pages)?

METR's task suite includes agentic evaluations where agents are given long-horizon tasks in realistic computer environments (web browsers, terminal access, file systems) and scored on whether they complete tasks safely and in accordance with operator specifications.

## Evaluation Infrastructure Challenges

### Contamination and Benchmark Gaming

Safety benchmarks face unique contamination risks. If WMDP questions appear in training data, a model may score high without having operationally useful knowledge. Contamination is mitigated through:

- Keeping evaluation questions confidential (not publicly released).
- Using held-out versions of benchmarks for each evaluation.
- Testing on dynamic benchmarks generated fresh for each evaluation.

### The "Overhang" Problem

Safety evaluations test current model capabilities, but rapidly improving models may pass evaluations at the time of testing and develop dangerous capabilities shortly after. RSPs attempt to address this by setting evaluation thresholds with a safety margin and re-evaluating models after fine-tuning or capability elicitation.

### Domain Expert Availability

Meaningful uplift evaluations for CBRN require domain experts (virologists, chemists, security researchers) who can judge whether model outputs provide genuine uplift. This expertise is scarce and expensive, limiting the scale of evaluations.

## Policy Landscape

Safety evaluations are becoming regulatory requirements:

- **UK Frontier AI Safety Commitments** (2023): major AI labs committed to sharing evaluation results with the UK AI Safety Institute (AISI) and granting access to pre-deployment models.
- **EU AI Act**: requires conformity assessments for "general-purpose AI models with systemic risk," including adversarial testing.
- **US Executive Order on AI** (October 2023): requires developers of dual-use foundation models to report safety test results to the government before deployment.
- **Seoul AI Safety Summit** (2024): 16 nations and major AI labs signed an agreement committing to pre-deployment safety testing and information sharing.

## Limitations and Ongoing Research

- **Capability elicitation**: safety evaluations measure capabilities under standard prompting. Fine-tuning or scaffolding can elicit capabilities that appear absent during base evaluation — meaning a model that passes a safety eval may still pose risks after fine-tuning by a third party.
- **Behavioral vs. mechanistic**: current evaluations are behavioral (testing input-output behavior). Mechanistic interpretability methods aim to identify dangerous knowledge at the parameter level, potentially enabling more robust evaluations independent of prompting.
- **Deceptive alignment**: models might behave safely during evaluation but unsafely during deployment — a theoretical risk that current evaluations cannot fully rule out.

## Summary

AI safety evaluations are structured, pre-deployment assessments that measure whether frontier models possess dangerous capabilities — including CBRN uplift, cyberoffense, and autonomous replication — and whether these capabilities could meaningfully increase catastrophic risk. Core concepts include uplift measurement against expert baselines, behavioral red-teaming, and structured threat model decomposition. Frameworks from METR (ARA evaluations), WMDP (knowledge benchmarks), and frontier lab RSPs provide the current operational methodology. Safety evals are transitioning from voluntary best practices to regulatory requirements, with international commitments requiring pre-deployment assessment and government reporting for the most capable AI systems.
