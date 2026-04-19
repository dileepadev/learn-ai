---
title: Instruction Tuning
description: Understand instruction tuning — the fine-tuning technique that transforms pre-trained language models into instruction-following assistants — covering datasets, training objectives, and why it unlocks generalization.
---

**Instruction tuning** (also called **supervised fine-tuning**, or SFT) is the process of fine-tuning a pre-trained language model on a dataset of (instruction, response) pairs to teach it to follow natural language instructions. It is the bridge between a raw language model — which predicts the next token — and a practical assistant that answers questions, writes code, and completes tasks on demand.

## The Pre-Training / Fine-Tuning Gap

Pre-trained LLMs (trained on next-token prediction over internet text) learn rich linguistic and world knowledge, but their behavior is not aligned with what users want:

- Given "Write a poem about autumn," a raw LLM might continue with another instruction: "Write a poem about spring."
- Given "What is the capital of France?", it might output: "What is the capital of Germany?"

The model has learned the *distribution of text* — not the *intent to help*. Instruction tuning closes this gap.

## The Instruction-Tuning Dataset

An instruction-tuning dataset consists of examples in the form:

```
Instruction: Summarize the following paragraph in two sentences.
Input: [paragraph text]
Output: [two-sentence summary]
```

Or in conversational format:

```
User: What are the main causes of climate change?
Assistant: The main causes of climate change are...
```

### Dataset Construction Methods

| Method | Description | Example Datasets |
|---|---|---|
| **Human-written** | Expert annotators write instruction-response pairs | OpenAssistant, Dolly |
| **Template-based** | Convert existing NLP tasks to instruction format | FLAN, Super-NaturalInstructions |
| **LLM-generated** | Use a powerful model to generate instruction-response pairs | Self-Instruct, Alpaca, Evol-Instruct |
| **Hybrid** | Combine human curation with LLM generation | OpenHermes, UltraChat |

### Self-Instruct

**Self-Instruct** (Wang et al., 2022) demonstrated that models can bootstrap their own instruction-tuning datasets:

1. Start with 175 seed instructions written by humans.
2. Use the model to generate new instructions, inputs, and outputs.
3. Filter low-quality or duplicate examples.
4. Fine-tune the model on the generated dataset.
5. Repeat.

This enabled the creation of **Stanford Alpaca** — a 7B parameter model instruction-tuned on 52,000 GPT-generated examples for under $500.

### Evol-Instruct and WizardLM

**Evol-Instruct** (Xu et al., 2023) augments a seed instruction dataset by systematically evolving instructions to be more complex, diverse, and challenging:

- **In-depth evolving**: Add constraints, deepen requirements, increase difficulty.
- **In-breadth evolving**: Generate new instructions from scratch inspired by existing ones.

The result is a dataset with greater diversity and difficulty distribution, producing more capable instruction-following models.

## Training Objective

Instruction tuning uses standard **supervised cross-entropy loss**, but applied only to the **output tokens** (not the instruction or input tokens):

$$\mathcal{L}(\theta) = -\sum_{t \in \text{output}} \log P_\theta(y_t \mid x_{1:t-1})$$

Where:

- $x$ is the full sequence (instruction + input + output).
- $y_t$ is the output token at position $t$.
- Loss is computed only over output token positions.

This is sometimes called **causal language modeling with loss masking**.

## Why Instruction Tuning Generalizes

A key empirical finding (**FLAN**, Wei et al., 2021): training on a diverse set of tasks in instruction format dramatically improves **zero-shot performance** on unseen tasks.

The hypothesis: instruction tuning does not teach new facts — it teaches the model to **recognize and respond to instructions** as a behavioral pattern. Since the pre-trained model already "knows" the answers, instruction tuning unlocks this knowledge by teaching the model *how to produce answers on demand*.

**Scaling behavior:**

- More instruction diversity → better generalization.
- Larger base models benefit more from instruction tuning.
- Quality of instruction-response pairs matters more than quantity.

## FLAN: Fine-tuned Language Models

**FLAN** (Google, 2021) was one of the first systematic studies of instruction tuning at scale:

- Fine-tuned a 137B parameter model on 62 NLP tasks reformatted as natural language instructions.
- Showed that instruction tuning improves zero-shot performance on held-out tasks significantly.
- Demonstrated that the benefit increases with model scale and number of training tasks.

**FLAN-T5** and **FLAN-PaLM** extended this work, becoming widely used open-source instruction-tuned models.

## Chat Templates

Modern instruction-tuned models use structured **chat templates** to format multi-turn conversations consistently:

```
<|system|>
You are a helpful assistant.
<|user|>
Explain quantum entanglement in simple terms.
<|assistant|>
Quantum entanglement is a phenomenon where...
```

Templates vary by model family:

- **ChatML**: `<|im_start|>user ... <|im_end|>`
- **Llama 3**: `<|start_header_id|>user<|end_header_id|> ...`
- **Mistral**: `[INST] ... [/INST]`

Using the correct template is critical for inference — a mismatch causes degraded performance.

## Instruction Tuning vs. RLHF

Instruction tuning (SFT) is often the **first phase** of alignment training; it is followed by **Reinforcement Learning from Human Feedback (RLHF)**:

| Stage | Method | Goal |
|---|---|---|
| **Pre-training** | Next-token prediction | Learn language and knowledge |
| **SFT** | Instruction tuning | Learn to follow instructions |
| **Reward modeling** | Train reward model on human preferences | Capture human preferences |
| **RLHF / DPO** | Optimize reward | Align with human values |

SFT alone can produce highly capable models, but RLHF adds preference alignment, safety, and helpfulness optimization on top.

## Key Datasets and Models

| Model | Base Model | Dataset | Notes |
|---|---|---|---|
| **Alpaca** | LLaMA 7B | 52K GPT-generated | Early open-source SFT |
| **Vicuna** | LLaMA 13B | ShareGPT conversations | Strong conversational ability |
| **WizardLM** | LLaMA | Evol-Instruct | High-complexity instructions |
| **Mistral-Instruct** | Mistral 7B | Proprietary | Strong open-source baseline |
| **Llama-3-Instruct** | Llama 3 | Meta's SFT + RLHF | State-of-the-art open source |
| **FLAN-T5** | T5 | 1,800+ NLP tasks | Classic encoder-decoder SFT |

## Practical Considerations

**Data quality over quantity**: A dataset of 1,000 high-quality, diverse instruction-response pairs typically outperforms 100,000 noisy ones.

**Format consistency**: The training format must exactly match the inference format. Inconsistencies in chat template usage lead to performance degradation.

**Catastrophic forgetting**: Aggressive fine-tuning can reduce performance on tasks not represented in the training data. Techniques like **LoRA** (low-rank adaptation) mitigate this by limiting parameter updates.

**Domain-specific instruction tuning**: Fine-tuning on domain-specific instructions (medical, legal, code) can significantly boost performance in that domain without sacrificing general capability, especially with LoRA.

## Further Reading

- [Fine-Tuned Language Models Are Zero-Shot Learners (FLAN) — Wei et al., 2021](https://arxiv.org/abs/2109.01652)
- [Self-Instruct — Wang et al., 2022](https://arxiv.org/abs/2212.10560)
- [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [WizardLM: Empowering LLMs with Evol-Instruct — Xu et al., 2023](https://arxiv.org/abs/2304.12244)
