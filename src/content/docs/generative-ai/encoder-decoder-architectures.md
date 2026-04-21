---
title: Encoder-Decoder Architectures in NLP
description: A deep dive into the encoder-decoder (seq2seq) transformer architecture — how it differs from decoder-only and encoder-only models, the role of cross-attention, landmark models like T5, BART, mT5, and FLAN-T5, and when to choose encoder-decoder over decoder-only.
---

**Encoder-decoder architectures** (also called **seq2seq** transformers) process an input sequence through an encoder and generate an output sequence through a decoder — with the decoder attending to the encoder's representations via **cross-attention**. This paradigm is well-suited for tasks that require a meaningful transformation from one sequence to another: translation, summarization, question answering from a document, and code generation from a specification.

Understanding encoder-decoder models alongside encoder-only (BERT-family) and decoder-only (GPT-family) models provides a complete picture of transformer design choices and their trade-offs.

## The Three Transformer Paradigms

Modern NLP models fall into three architectural families:

| Architecture | Examples | Attention Mask | Best For |
| --- | --- | --- | --- |
| Encoder-only | BERT, RoBERTa, DeBERTa | Bidirectional (full) | Classification, NER, embeddings |
| Decoder-only | GPT, LLaMA, Mistral | Causal (left-to-right) | Open-ended generation, chat, reasoning |
| Encoder-decoder | T5, BART, mT5, FLAN-T5 | Bidirectional encoder + causal decoder | Translation, summarization, structured generation |

## The Encoder-Decoder Architecture

### The Encoder

The encoder is a standard bidirectional transformer stack. It processes the full input sequence with **unrestricted self-attention** — every token can attend to every other token in both directions. This allows the encoder to build rich, context-aware representations of each input token.

For a translation task, the encoder processes the full source sentence and produces a sequence of hidden states $H = \{h_1, h_2, \ldots, h_S\}$, one per source token. Each $h_i$ encodes the meaning of token $i$ in the context of the entire sentence.

### The Decoder

The decoder generates the output sequence autoregressively — one token at a time, left to right. Its self-attention uses a **causal mask**: each output token attends only to previously generated output tokens (just like GPT). This ensures that generation is causal and the model doesn't "cheat" by looking at future output tokens.

### Cross-Attention: The Bridge

The key mechanism that distinguishes encoder-decoder from decoder-only is **cross-attention** in each decoder layer. Cross-attention allows the decoder to query the encoder's hidden states:

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Where:

- **Queries** $Q$ come from the decoder's current hidden state.
- **Keys** $K$ and **Values** $V$ come from the encoder's output $H$.

At each decoding step, the decoder learns *which parts of the input* to attend to. In translation, when generating the French word "chat," the decoder's cross-attention focuses on the English word "cat" in the encoder output.

This is fundamentally different from decoder-only models, which can only attend to their own context window (no separate, independently computed encoder representation exists).

## T5: Text-to-Text Transfer Transformer

**T5** (Raffel et al., Google, 2020) is the definitive encoder-decoder model. Its central idea: frame **every NLP task as a text-to-text problem**. Classification becomes text generation; translation becomes text-to-text transformation; question answering produces a text answer. This unified framing lets a single model and training procedure handle all tasks.

### T5 Task Formatting Examples

```text
Task: Translation
Input:  "translate English to French: The cat sat on the mat."
Output: "Le chat était assis sur le tapis."

Task: Summarization
Input:  "summarize: [long article text]"
Output: "[summary]"

Task: Question Answering
Input:  "question: What is the capital of France? context: France is a country in Europe. Its capital is Paris."
Output: "Paris"

Task: Classification (SST-2)
Input:  "sst2 sentence: This movie was great!"
Output: "positive"
```

### T5 Architecture Details

- Encoder: 12 transformer blocks (T5-Base).
- Decoder: 12 transformer blocks with cross-attention.
- Uses **relative position biases** rather than absolute positional encodings.
- Trained with **span corruption**: randomly replace spans of text with sentinel tokens; learn to reconstruct the spans in the output.
- Trained on **C4** (Colossal Clean Crawled Corpus), a 750 GB cleaned web corpus.

### T5 Model Family

- **T5-Small** (60M) → **T5-Base** (220M) → **T5-Large** (770M) → **T5-XL** (3B) → **T5-XXL** (11B).
- **mT5**: Multilingual T5, trained on 101 languages from the multilingual Common Crawl.
- **FLAN-T5**: T5 fine-tuned with **instruction tuning** on over 1000 NLP tasks formatted as natural language instructions. FLAN-T5 dramatically outperforms vanilla T5 on zero- and few-shot tasks.
- **UL2**: Unifies multiple pretraining objectives (causal, prefix, span corruption) into a single model via a "mode token" that indicates which objective to use.

## BART: Denoising Autoencoder for Seq2Seq

**BART** (Lewis et al., Facebook AI, 2020) approaches encoder-decoder pretraining differently: it uses a **denoising autoencoder** objective. The encoder receives a corrupted version of the text; the decoder learns to reconstruct the original.

### BART Corruption Strategies

- **Token masking**: Replace random tokens with `[MASK]`.
- **Token deletion**: Delete random tokens (decoder must infer where tokens were deleted).
- **Text infilling**: Replace random spans with a single `[MASK]` token.
- **Sentence permutation**: Shuffle sentences in a document randomly.
- **Document rotation**: Rotate the document to begin at a random token.

Combining all five noising strategies produced the strongest BART model, particularly for **abstractive summarization** and **question generation**. BART outperformed T5 on summarization (CNN/DailyMail, XSum) due to its denoising objective being naturally aligned with summarization.

### BART for Summarization and Translation

BART's architecture is identical to the original Transformer (encoder + decoder). Its encoder and decoder are initialized with BERT-style bidirectional and GPT-style causal pretraining respectively, then jointly pretrained with denoising.

Fine-tuned BART (and **mBART**, its multilingual variant) set state-of-the-art on:

- **Abstractive summarization** (CNN/DM, XSum, Multi-News).
- **Machine translation** with extremely limited parallel data.
- **Dialogue and story generation**.

## Encoder-Decoder vs. Decoder-Only: The Modern Debate

As decoder-only models (GPT-3, LLaMA) have scaled to enormous sizes and demonstrated strong in-context learning, the question has arisen: do encoder-decoder models still have a place?

### Arguments for Encoder-Decoder

**Efficiency for conditional generation**: When the input is fixed and the output is what varies (translation, summarization, question answering), the encoder processes the input once. In decoder-only models, the full input-output concatenation must be processed at every decoding step — wasting compute re-encoding the input.

**Structured generation quality**: For tasks where the output structure is tightly coupled to the input (table-to-text, structured extraction, schema-guided generation), cross-attention provides a more direct mechanism than hoping the decoder "remembers" the input from its causal context.

**Smaller models for specialized tasks**: A 250M-parameter FLAN-T5-Large can outperform GPT-3 (175B) on many structured NLP benchmarks due to task-specific fine-tuning. For resource-constrained production deployments, encoder-decoder models remain highly practical.

### Arguments for Decoder-Only

**Simplicity and unification**: A single architecture handles both understanding and generation. Scaling laws apply cleanly. In-context learning emerges naturally.

**No cross-attention overhead**: Decoder-only models have simpler attention patterns — one less attention type to implement and optimize.

**Larger scale advantages**: Decoder-only models dominate at the frontier (100B+ parameters). The emergence of reasoning, planning, and chain-of-thought abilities at large scale has been primarily demonstrated in decoder-only models.

**Generalization to arbitrary tasks**: Decoder-only models, with appropriate prompting, can handle any task without architectural modification. Encoder-decoder models require the task to be framed as a seq2seq problem with a clear input-output boundary.

## Prefix Language Models (Hybrid)

**Prefix LMs** (used in UniLM, PaLM-style extensions) are a hybrid: the input tokens attend bidirectionally (like an encoder), and the output tokens attend causally to all previous tokens including the prefix (like a decoder). This eliminates the separate encoder stack while retaining some bidirectional encoding benefit.

T5 also supports a **prefix mode** (via the "fill in the middle" framework) where a configurable prefix is encoded bidirectionally and the rest is generated autoregressively.

## When to Choose Encoder-Decoder

| Scenario | Recommended Architecture |
| --- | --- |
| Production translation at scale | Encoder-decoder (T5, mBART) |
| Abstractive summarization (specialized) | Encoder-decoder (BART, PEGASUS) |
| Large-scale instruction following | Decoder-only (LLaMA, FLAN + decoder) |
| Open-ended generation, chat | Decoder-only |
| Document extraction / structured QA | Encoder-decoder (FLAN-T5) or decoder-only with long context |
| Multilingual NLP with limited resources | mT5, mBART |
| Research on bounded seq2seq tasks | Encoder-decoder (cleaner benchmarking) |

## Key Models at a Glance

| Model | Type | Parameters | Key Strength |
| --- | --- | --- | --- |
| T5-XXL | Enc-Dec | 11B | General seq2seq, text-to-text |
| FLAN-T5-XXL | Enc-Dec | 11B | Zero/few-shot instruction following |
| BART-Large | Enc-Dec | 400M | Abstractive summarization |
| mT5-XXL | Enc-Dec | 13B | Multilingual seq2seq |
| mBART-50 | Enc-Dec | 600M | Multilingual translation |
| PEGASUS | Enc-Dec | 568M | News summarization |
| UL2 | Enc-Dec | 20B | Mixed pretraining objectives |

Encoder-decoder architectures remain the right tool for a substantial portion of NLP production workloads — where conditional generation quality, inference efficiency on bounded input-output tasks, and the ability to fine-tune compact models matter more than the raw frontier capabilities of trillion-token decoder-only giants.
