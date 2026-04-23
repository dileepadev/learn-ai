---
title: Continual Pretraining of Language Models
description: Learn how continual pretraining extends base language models with new domain knowledge or temporal information — covering techniques to avoid catastrophic forgetting, domain adaptation strategies, and practical recipes for updating LLMs efficiently.
---

**Continual pretraining** (also called **continued pretraining** or **domain-adaptive pretraining**) is the process of further training an already-pretrained language model on new corpora to extend its knowledge, improve domain-specific capabilities, or update it with information from a more recent time period. Unlike fine-tuning — which optimizes a model for a specific task — continual pretraining maintains the model's general language understanding while enriching the knowledge encoded in its parameters.

As LLMs become the foundation of AI applications, the need to extend and update their knowledge without retraining from scratch has become a critical practical challenge. Continual pretraining sits at the intersection of transfer learning, catastrophic forgetting research, and the economics of large model training.

## Why Continual Pretraining?

### The Knowledge Cutoff Problem

Every pretrained LLM has a **training data cutoff** — events, publications, and developments after that date are unknown to the model. GPT-4's knowledge cutoff, LLaMA 3's knowledge cutoff, and Gemini's knowledge cutoff all represent hard limits on what the model knows. For applications requiring up-to-date information — financial modeling, medical literature, legal research — a stale knowledge base is a significant limitation.

Continual pretraining on recent text data updates the model's world knowledge without requiring full pretraining from scratch — which would cost tens to hundreds of millions of dollars for frontier-scale models.

### Domain Adaptation

General-purpose LLMs are trained on broad internet corpora weighted toward general web text, code, and books. This distribution underrepresents specialized domains:

- **Biomedical**: Clinical notes, pathology reports, genomics literature, drug interaction databases.
- **Legal**: Case law opinions, statutes, contracts, regulatory filings.
- **Scientific**: Research papers across physics, chemistry, materials science, earth science.
- **Financial**: Earnings reports, analyst notes, regulatory filings, market data commentary.
- **Code**: Domain-specific programming languages, internal codebases, specialized technical documentation.

Continual pretraining on domain-specific corpora enriches the model's representations for that domain — improving downstream task performance with less fine-tuning data. **BioMedLM**, **Galactica**, **FinMA**, and **CodeLLaMA** (the code-focused continuation of LLaMA 2) are prominent examples of domain-adapted models.

### Language and Multilingual Extension

A model pretrained primarily on English text may be extended to cover additional languages through continual pretraining on multilingual corpora — adding language capability while retaining English performance, at a fraction of the cost of multilingual pretraining from scratch.

## Catastrophic Forgetting

The central challenge of continual pretraining is **catastrophic forgetting** (McCloskey & Cohen, 1989) — the tendency of neural networks trained on new data to rapidly overwrite the weights encoding previously learned information, degrading performance on tasks the model previously performed well.

In LLM continual pretraining, catastrophic forgetting manifests as:

- **General capability degradation**: The model becomes better at domain-specific tasks but worse at general benchmarks (MMLU, HellaSwag, ARC) — a form of regression.
- **Instruction following degradation**: If the base model was instruction-tuned, continued pretraining on raw text can erode instruction-following ability.
- **Hallucination increase**: Destabilized representations may increase hallucination rates on familiar topics.

Mitigating forgetting is the primary technical challenge in continual pretraining.

## Techniques for Mitigating Catastrophic Forgetting

### Data Mixing

The most empirically reliable mitigation is **mixing new domain data with general pretraining data** — rather than training exclusively on the new corpus:

$$\mathcal{D}_{train} = \alpha \cdot \mathcal{D}_{new} + (1 - \alpha) \cdot \mathcal{D}_{general}$$

A mixing ratio $\alpha$ of 0.5–0.9 (50–90% new domain data) is typical, with the optimal ratio depending on the domain shift magnitude and how much general capability must be preserved.

**Why mixing works**: New domain data updates relevant knowledge while general data continually reinforces the representations needed for general tasks, preventing their decay.

### Learning Rate Warmup and Scheduling

Using a **lower learning rate** than original pretraining significantly reduces forgetting:

- Original pretraining learning rates (e.g., $3 \times 10^{-4}$ for a 7B model) are too large for continual pretraining — they overwrite existing representations too aggressively.
- Continual pretraining typically uses $1/3$ to $1/10$ of the original peak learning rate.
- A **short warmup period** (1,000–5,000 steps) gradually increases the learning rate before the cosine decay schedule — preventing large early updates from destroying established weights.

**Re-warming the learning rate** — starting from the end-of-pretraining LR (near zero) and warming back up — has been found beneficial in some continual pretraining recipes, as it allows the optimizer to gradually integrate new gradient information.

### Replay

**Replay** (also called **experience replay**) maintains a buffer of examples from previous training data and mixes them into each batch:

- **Random replay**: Uniformly sample from the original pretraining corpus.
- **Importance-weighted replay**: Sample examples that are most likely to suffer from forgetting — prioritizing examples on which the current model already has low loss (well-learned examples that could be forgotten).

Replay is a form of data mixing, but applied dynamically during training rather than by constructing a static mixed dataset.

### Elastic Weight Consolidation (EWC)

**EWC** (Kirkpatrick et al., 2017) adds a regularization term to the loss function that penalizes changes to parameters deemed important for previous tasks:

$$\mathcal{L}(\theta) = \mathcal{L}_{new}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where $F_i$ is the **Fisher information** of parameter $i$ (estimated from the original training data), $\theta_i^*$ is the original parameter value, and $\lambda$ controls the regularization strength.

EWC protects "important" weights from large updates. In practice, computing Fisher information for LLMs with billions of parameters is computationally expensive, and approximate variants are typically used.

### LoRA-Based Continual Pretraining

**Low-Rank Adaptation (LoRA)** restricts parameter updates to low-rank matrices added alongside existing weight matrices:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$.

By fixing the original weights $W$ and training only the low-rank adapters $B$ and $A$, LoRA-based continual pretraining:

- **Preserves general capabilities** in the frozen original weights.
- **Encodes new knowledge** in the learned adapters.
- **Reduces compute requirements** — updating only 0.1–1% of parameters.
- **Enables modular knowledge extension** — multiple adapter sets can be maintained for different domains and composed or swapped at inference time.

**QLoRA** combines LoRA with 4-bit quantization of the base model, enabling continual pretraining of large models on consumer-grade hardware.

## Practical Continual Pretraining Recipe

A typical continual pretraining workflow for a 7B parameter model:

### 1. Data Preparation

```python
# Target distribution: 70% domain data, 30% general replay data
domain_tokens = 10_000_000_000   # 10B tokens of domain text
general_tokens = 4_300_000_000   # 4.3B tokens of general web text

# Tokenize using the same tokenizer as the base model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Pack sequences to context length (no padding waste)
# Standard packing: concatenate documents with EOS tokens, split into fixed-length chunks
```

### 2. Training Configuration

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./checkpoints",
    max_steps=10_000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,       # Effective batch: 32 per GPU
    learning_rate=3e-5,                  # ~10x lower than original pretraining
    lr_scheduler_type="cosine",
    warmup_steps=500,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=50,
    save_steps=1000,
    deepspeed="ds_config_zero2.json",    # ZeRO-2 for multi-GPU
)
```

### 3. Evaluation Throughout Training

Monitor **both** domain performance and general capability benchmarks throughout training:

- Domain perplexity on held-out domain text.
- General benchmark scores (MMLU, HellaSwag, ARC, TruthfulQA) — watch for degradation.
- Instruction-following capability (MT-Bench or similar) if the base model was instruction-tuned.

Early stopping or LR reduction when general benchmark scores drop more than an acceptable threshold.

## Domain-Adaptive Pretraining (DAPT)

**DAPT** (Gururangan et al., 2020) systematically studied the effect of continued pretraining on domain-specific text before task fine-tuning, finding:

- Domain adaptation always helps downstream task performance, even for tasks using general-domain fine-tuning data.
- **Task-Adaptive Pretraining (TAPT)** — pretraining on the unlabeled text of the specific downstream task dataset — provides additional gains beyond domain adaptation.
- The combination DAPT → TAPT → fine-tuning outperforms fine-tuning alone by large margins, particularly when labeled data is scarce.

This cascaded strategy — general pretraining → domain adaptation → task adaptation → task fine-tuning — has become a standard recipe for building high-performance specialized models.

## Case Studies

### CodeLLaMA

**CodeLLaMA** (Meta, 2023) was produced by continually pretraining LLaMA 2 on 500B tokens of code data with a 0.1 LR warmup from LLaMA 2's end checkpoint. The result dramatically improved code generation capabilities while retaining LLaMA 2's general language understanding — demonstrating that a 500B token domain-specific continuation can achieve near-parity with models specifically pretrained on code (like StarCoder).

### BioMedLM / PubMedBERT

**PubMedBERT** (Gu et al., 2021) pretrained BERT from scratch on PubMed abstracts, while **BioMedLM** (Stanford) continually pretrained GPT-2 on biomedical literature. Both significantly outperformed general-domain models on biomedical NLP benchmarks, establishing that domain-specific pretraining data provides benefits beyond what fine-tuning alone can achieve.

## Continual Pretraining vs. RAG

For knowledge updating, continual pretraining competes with **Retrieval-Augmented Generation (RAG)**:

| Aspect | Continual Pretraining | RAG |
| --- | --- | --- |
| New knowledge update | Encoded in weights | Retrieved at inference |
| Update cost | High (GPU compute) | Low (index update) |
| Inference cost | Standard | Higher (retrieval + generation) |
| Knowledge staleness | Requires retraining | Near real-time |
| Complex reasoning | Strong | Depends on retrieval quality |
| Hallucination risk | Reduced for trained facts | Depends on retriever precision |

The two approaches are complementary: continual pretraining establishes foundational domain knowledge in the model's weights, while RAG provides access to current, specific facts that cannot be economically encoded through pretraining. Production systems often combine both.
