---
title: Model Cards and AI Documentation
description: Learn how model cards and standardized AI documentation practices enable transparency, reproducibility, and responsible deployment of machine learning models in production systems.
---

**Model cards** are structured documents that provide standardized, transparent descriptions of machine learning models — covering their intended use, performance characteristics, training data, limitations, and ethical considerations. Introduced by Mitchell et al. at Google in 2019, model cards have become a widely adopted convention for responsible AI disclosure, serving engineers, researchers, policymakers, and end users.

Think of a model card as the **"nutritional label"** for a machine learning model: a concise, honest summary of what the model is, what it is good at, what it should not be used for, and where it might fail.

## Why AI Documentation Matters

The history of AI deployments contains many cases of models behaving unexpectedly in production — failing on demographic groups not represented in training, drifting over time as input distributions shift, or being applied to contexts far outside their original design. Much of this harm stems from a documentation gap: models shipped without clear records of their capabilities, limitations, and intended contexts.

Comprehensive documentation addresses this by enabling:

- **Informed deployment decisions**: Teams can evaluate whether a model's characteristics match their specific use case.
- **Accountability**: When a model causes harm, documentation creates a trail of decisions and trade-offs that were known at the time of deployment.
- **Reproducibility**: Researchers can replicate, compare, and build on documented models rather than starting from scratch.
- **Regulatory compliance**: Emerging AI regulations (EU AI Act, NIST AI RMF) require documentation as evidence of due diligence.
- **Cross-team communication**: Model cards transfer knowledge between the team that built a model and the teams that deploy or audit it.

## The Model Card Structure

While formats vary, model cards typically include the following sections:

### 1. Model Details

Basic metadata about the model:

- **Model name and version**: Including versioning scheme (e.g., `sentiment-classifier-v2.1`).
- **Model type**: Architecture, modality, and task (e.g., "BERT-based text classifier for sentiment analysis").
- **Developers**: The individuals, team, or organization responsible.
- **Date**: Training completion and card publication date.
- **Contact**: Where to report issues or ask questions.
- **License**: Usage rights and restrictions.

### 2. Intended Use

A critical section that explicitly states what the model was designed for and, equally importantly, what it should **not** be used for.

**Primary intended uses**: The specific tasks, domains, and populations the model was designed to serve. Being explicit here reduces the risk of out-of-context deployment.

**Out-of-scope uses**: Explicitly list use cases that are not appropriate for this model — either because performance is insufficient or because the use case raises ethical concerns. For example: "This model should not be used for hiring decisions or medical diagnosis."

**Intended users**: Who is expected to interact with the model — engineers integrating it into applications, end users interacting directly, or researchers studying its behavior.

### 3. Factors

Describes the **features and population groups** most relevant to model performance variation:

- **Relevant factors**: Variables that may affect performance — demographic attributes (age, gender, language, dialect), environmental conditions (image resolution, audio quality), or operational context (formal vs. informal text, clinical vs. consumer settings).
- **Evaluation factors**: Which of these factors were actually evaluated (may differ from relevant factors due to data availability).

This section is crucial for identifying who may be underserved by the model.

### 4. Metrics

Documents how model performance is measured:

- **Performance measures**: Which metrics are reported and why they were chosen (accuracy, F1, AUC, calibration error, latency, etc.).
- **Decision thresholds**: For classification models, what threshold converts scores to decisions, and how this threshold was chosen.
- **Variation approaches**: How metrics were computed across different population groups and operating conditions.

### 5. Evaluation Data

Describes the dataset used to evaluate the model:

- Dataset name, version, and source.
- Preprocessing steps applied.
- Any known limitations or biases in the evaluation data.
- Why this dataset was chosen for evaluation.

### 6. Training Data

Describes what data the model was trained on:

- Data sources and collection methodology.
- Training data size and composition.
- Key preprocessing and filtering decisions.
- Known demographic composition of the training data.
- Any privacy or consent considerations.

Note: Proprietary model developers sometimes decline to fully disclose training data. Even in these cases, documenting what *can* be shared (scale, data types, known characteristics) is valuable.

### 7. Quantitative Analyses

The core performance reporting section:

- **Disaggregated evaluation**: Performance broken down by relevant subgroups (e.g., accuracy by gender, language, age group). This is the most important section for identifying disparate impact.
- **Intersectional analysis**: Performance on combinations of attributes (e.g., older women, speakers of low-resource dialects) where performance may be worse than any single-attribute breakdown suggests.
- Visualization with confidence intervals where possible.

### 8. Ethical Considerations

An honest assessment of the ethical implications of the model:

- What harm could occur if the model makes errors?
- Are there use cases where deployment would be inappropriate regardless of accuracy?
- What biases exist or may exist in the model's outputs?
- What privacy risks does the model create?
- Were affected communities consulted during development?

### 9. Caveats and Recommendations

Practical guidance for deployers:

- Known failure modes and edge cases.
- Performance degradation conditions (out-of-distribution inputs, adversarial examples).
- Monitoring and maintenance recommendations.
- Suggested complementary safeguards (human review for high-stakes decisions, input validation, output filtering).

## Datasheets for Datasets

Model cards have a companion document for training data: **Datasheets for Datasets** (Gebru et al., 2018). Datasheets document:

- **Motivation**: Why was this dataset created? By whom and for what purpose?
- **Composition**: What does each instance represent? What is the distribution across categories?
- **Collection process**: How was data collected? Were subjects consented? Were they compensated?
- **Preprocessing**: What cleaning, filtering, or transformation was applied? What was removed and why?
- **Recommended uses**: What tasks is this dataset appropriate for?
- **Out-of-scope uses**: What uses should be avoided?
- **Maintenance**: Who will maintain the dataset? What is the update schedule?

Datasheets enable model card authors to accurately characterize their training and evaluation data, closing a key loop in the documentation chain.

## System Cards

While model cards document individual models, **system cards** (popularized by Meta for systems like Galactica and Llama) document the complete AI system including:

- Multiple models interacting with each other.
- Retrieval systems, knowledge bases, and tools.
- Safety classifiers and filters.
- Human review processes.
- Monitoring and incident response procedures.

System cards are increasingly important as AI deployments move from single models to multi-component pipelines with complex behavior that cannot be predicted from any single model's card.

## Model Cards in Practice: Hugging Face

The **Hugging Face Model Hub** has operationalized model cards at scale — every model hosted on the Hub has a model card rendered from a structured README.md file. Hugging Face provides:

- A **model card template** with standard sections.
- **Metadata tags** for tasks, languages, datasets, and licenses — enabling filtering and discovery.
- **Evaluation results tables** that integrate with the `evaluate` library for standardized benchmarking.

This has normalized model card adoption in the open-source ML community. Searching the Hub for well-documented models is now a standard part of ML engineering practice.

## Automated Model Documentation

Manually writing model cards is time-consuming, especially as teams iterate rapidly on models. Several tools partially automate documentation:

- **Hugging Face `model-card-creator`**: A wizard-style interface that guides users through standard sections.
- **TensorFlow Model Card Toolkit**: Programmatically generates model cards from training metadata and evaluation results, integrating with TFX pipelines.
- **Google's Responsible AI Toolkit**: Provides programmatic model card generation with visualizations from evaluation results computed with the Fairness Indicators library.
- **MLflow**: Experiment tracking that captures training parameters, metrics, and artifacts that can populate a model card's quantitative sections.

The goal is to **generate most of the model card automatically** from metadata already produced during training and evaluation, with humans filling in the qualitative sections (intended use, ethical considerations).

## Model Cards for Generative AI

Generative AI models — LLMs, text-to-image systems, multimodal models — pose new challenges for model cards:

- **Open-ended outputs**: Unlike classifiers with a fixed output space, generative models produce unconstrained outputs that cannot be fully characterized by a single metric.
- **Emergent capabilities**: LLMs acquire capabilities not explicitly trained for; model cards must acknowledge limitations in what can be documented about emergent behaviors.
- **Safety evaluations**: Cards for LLMs should include results on safety benchmarks (TruthfulQA, BBQ, Toxigen) and red-teaming efforts.
- **System prompt dependence**: LLM behavior varies significantly with the system prompt. Documenting behavior requires specifying which prompting configurations were evaluated.

Meta's Llama cards, OpenAI's GPT-4 technical report, Google's Gemini technical report, and Anthropic's model cards have collectively established emerging norms for documenting large foundation models.

## Regulatory Context

Model documentation is increasingly required by regulation:

- **EU AI Act**: High-risk AI systems must maintain technical documentation covering training data, performance metrics, and human oversight provisions.
- **NIST AI Risk Management Framework (AI RMF)**: Recommends model cards as a governance artifact for the "Govern," "Map," and "Measure" functions.
- **US Executive Order on AI (2023)**: Requires documentation from developers of the most powerful AI models as a condition of safety reporting.
- **Financial services regulation**: Regulators increasingly require model risk management documentation for AI models used in lending, insurance, and trading.

Proactive model card adoption is becoming a regulatory risk mitigation strategy, not just a best practice.

## Common Pitfalls in Model Documentation

| Pitfall | Impact | Remedy |
|---------|--------|--------|
| **Incomplete disaggregated evaluation** | Disparate harm to underrepresented groups goes undetected | Evaluate on all relevant subgroups; use intersectional analysis |
| **Vague intended use** | Models deployed in inappropriate contexts | Explicitly list in-scope and out-of-scope uses |
| **Overly optimistic metrics** | Deployers overestimate reliability | Report worst-case and average-case metrics; include confidence intervals |
| **Stale documentation** | Card does not reflect current model version | Version model cards with models; update on every release |
| **Missing training data disclosure** | Inability to assess bias and copyright risk | Document data sources, composition, and known limitations |

## Building a Documentation Culture

Model cards are only effective if embedded in engineering and research culture:

- **Make documentation a gate**: Require a completed model card as part of model deployment approval.
- **Review cards in model reviews**: Treat card quality as equivalent to code quality in model reviews.
- **Credit documentation work**: Recognize model card authorship as a valued engineering contribution.
- **Iterate on cards**: Update cards with new evaluation results, discovered failure modes, and user feedback — treat them as living documents.

The ultimate goal of model documentation is not compliance but **trust** — giving the people who use, audit, and are affected by AI systems the information they need to evaluate whether they can and should trust a model's outputs in their specific context. As AI systems become more powerful and more widely deployed, this transparency becomes not just good practice but an ethical imperative.
