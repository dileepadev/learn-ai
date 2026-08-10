---
title: "Mixture of Formats: Multi-Format Training for LLMs"
description: Explore the Mixture of Formats (MoF) paradigm for LLM training — how combining diverse data representations like prose, tables, code, JSON, and mathematical notation improves generalization, format transfer, and structured output quality.
---

Large language models are trained on text corpora that span many document types: web pages, scientific papers, code repositories, SQL tables, Markdown documentation, JSON APIs, and mathematical derivations. Yet most LLM training pipelines treat all of this as undifferentiated token sequences — losing structural signals that could significantly improve format-specific capabilities.

**Mixture of Formats (MoF)** is a training paradigm that deliberately curates and balances training data across diverse textual formats, and in some approaches, trains with format-aware objectives that make the model explicitly aware of the structural properties of different data representations.

## What Is a "Format" in LLM Training?

A format is a structured way of representing information with a distinct grammar, syntax, and semantic interpretation:

| Format | Examples | Key Structural Properties |
|---|---|---|
| Prose | Wikipedia, books, web articles | Linear argument flow, paragraph structure |
| Code | Python, JavaScript, SQL, Bash | Executable semantics, strict syntax, indentation |
| Structured data | JSON, YAML, TOML, XML | Hierarchical key-value relationships |
| Tables | Markdown tables, HTML tables, CSV | Row-column relationships, headers |
| Mathematical notation | LaTeX, MathML, plain math | Symbolic reasoning, operator precedence |
| Lists and outlines | Bullet points, numbered lists | Hierarchical enumeration |
| Dialogue | Chat logs, interviews | Turn-taking, speaker attribution |
| Mixed documents | Jupyter notebooks, technical reports | Interleaved formats |

Most training corpora are heavily skewed toward prose (web text) with code and structured data as significant but minority contributions. MoF asks: does format balance matter? And what happens when we train models to explicitly reason about format transitions?

## Why Format Balance Matters

### Format-Specific Capabilities

Models exhibit dramatically different capability profiles across formats even when training on the same underlying information:

- A model trained heavily on code produces better code but worse prose
- A model trained mostly on prose generates fluent text but makes structural errors in JSON
- Models under-trained on tables often hallucinate table structure rather than accurately representing relationships

Format balance in training data directly influences these capability asymmetries.

### Format Transfer Learning

A key finding from MoF research is **format transfer** — learning to work in one format improves capabilities in structurally similar formats:

- Training on code improves mathematical reasoning (both require precise, sequential logic)
- Training on SQL improves structured JSON generation (both involve schema-constrained data representations)
- Training on Markdown improves document structure generation (hierarchical formatting skills)

This suggests that format diversity in training data has superlinear benefits: the model doesn't just learn each format in isolation, it learns abstract structural reasoning skills that generalize across format boundaries.

### Instruction Following and Format Compliance

When users request output in a specific format ("respond in JSON," "write a Python function," "create a Markdown table"), the model must:
1. Understand the format specification
2. Generate content that is syntactically valid in that format
3. Preserve semantic accuracy while conforming to format constraints

Models with richer format training handle format-switching instructions more reliably.

## MoF Training Approaches

### 1. Format-Stratified Data Mixing

The simplest MoF approach is curriculum data mixing with explicit format-based stratification:

```python
def build_mof_dataset(
    prose_data,
    code_data,
    json_data,
    table_data,
    math_data,
    target_tokens: int,
    mixing_weights: dict,
) -> Dataset:
    """
    Construct a training dataset with specified format mixing weights.
    mixing_weights: {"prose": 0.4, "code": 0.25, "json": 0.15, "table": 0.1, "math": 0.1}
    """
    datasets = {
        "prose": prose_data,
        "code": code_data,
        "json": json_data,
        "table": table_data,
        "math": math_data,
    }

    total = sum(mixing_weights.values())
    samples = {}
    for fmt, weight in mixing_weights.items():
        n_tokens = int(target_tokens * weight / total)
        samples[fmt] = datasets[fmt].sample_by_tokens(n_tokens)

    combined = concatenate_datasets(list(samples.values()))
    return combined.shuffle(seed=42)
```

The key insight is that **explicit format stratification prevents natural imbalance** in scraped web corpora, which are typically ~70-80% prose by token count.

### 2. Format-Conditioned Training

A more sophisticated approach adds **format conditioning tokens** — special tokens that prefix each training document and explicitly identify its format:

```
<|format:python_code|>
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

<|format:markdown_table|>
| Name | Age | Department |
|------|-----|------------|
| Alice | 32 | Engineering |
| Bob | 28 | Design |

<|format:json_object|>
{"name": "Alice", "age": 32, "department": "Engineering"}
```

Format conditioning teaches the model to associate format tokens with their structural properties. At inference time, injecting a format token into the prompt biases generation toward that format's conventions — a form of lightweight format-specific fine-tuning without separate models.

### 3. Cross-Format Translation Objectives

The most ambitious MoF approach trains models explicitly on **format translation** — converting the same information between formats as a training objective:

```
Task: Convert the following table to JSON

Input (table):
| Country | GDP (trillion USD) | Population (millions) |
|---------|-------------------|----------------------|
| USA | 27.4 | 335 |
| China | 17.7 | 1412 |
| Germany | 4.1 | 83 |

Target output (JSON):
[
  {"country": "USA", "gdp_trillion_usd": 27.4, "population_millions": 335},
  {"country": "China", "gdp_trillion_usd": 17.7, "population_millions": 1412},
  {"country": "Germany", "gdp_trillion_usd": 4.1, "population_millions": 83}
]
```

Training on thousands of such format translation examples teaches the model that different formats are alternative representations of the same semantic content — building a format-agnostic semantic layer on top of format-specific surface renderings.

### 4. Mixed-Format Document Training

Real-world documents (Jupyter notebooks, technical manuals, scientific papers) naturally interleave formats. Training on these documents specifically — rather than separating formats into homogeneous chunks — teaches models to handle format transitions mid-document:

```
# Sales Analysis (Markdown header)

This quarter we observed strong performance in the enterprise segment.
(Prose)

```python
import pandas as pd
df = pd.read_csv("sales_q3.csv")
summary = df.groupby("segment")["revenue"].sum()
```
(Code block)

| Segment | Revenue ($M) | Growth YoY |
|---------|-------------|------------|
| Enterprise | 42.3 | +18% |
| SMB | 28.1 | +7% |
(Table)

The data confirms our hypothesis: enterprise adoption is accelerating faster than forecast.
(Prose again)
```

Mixed-format document training is particularly valuable for models deployed in knowledge work contexts where users work with mixed documents daily.

## Format-Aware Evaluation

Evaluating MoF models requires format-specific benchmarks:

**Structural validity rate:** What percentage of outputs in a requested format are syntactically valid?
```python
def json_validity_rate(model_outputs: list[str]) -> float:
    import json
    valid = sum(1 for o in model_outputs if try_parse_json(o))
    return valid / len(model_outputs)
```

**Format transfer accuracy:** Given information in format A, how accurately does the model reproduce it in format B?

**Format instruction compliance:** When explicitly requested to use a specific format, what fraction of responses comply?

**Cross-format reasoning:** Can the model reason about relationships between data when it is presented in different formats?

## Connections to Multimodal Training

MoF is conceptually related to multimodal training — both paradigms argue that models benefit from diverse representational modalities. The key difference is that MoF operates entirely within the text modality, using the structural diversity of text formats as the source of representational variety.

Some researchers consider MoF a precursor to multimodal training: a model with strong MoF training handles the "text side" of multimodal inputs more robustly, because it can already navigate rich structural variation within text.

## Practical Findings from MoF Research

Several empirical findings have emerged from MoF experiments:

- **Code data helps math:** Adding code to math-heavy training corpora consistently improves performance on formal reasoning tasks (MATH, GSM8K) even without additional math data
- **Structured data improves instruction following:** Training on JSON and YAML improves the model's ability to follow structured instructions in prose form
- **Format diversity reduces format overfitting:** Models trained on a single format become brittle when asked to generate in a different format; MoF training improves robustness
- **Mixing ratio matters more than total format tokens:** A well-balanced 10B token corpus often outperforms a poorly-balanced 50B token corpus on multi-format benchmarks
- **Format-conditioned generation is learnable with few examples:** As few as 1,000 format-conditioned examples per format is sufficient to teach reliable format token conditioning

## Applying MoF in Practice

For teams fine-tuning or instruction-tuning models on proprietary data:

1. **Audit your training data format distribution** — most enterprise datasets are heavily prose-skewed; deliberately add code examples, JSON schemas, and table data
2. **Include format translation examples** in your instruction tuning dataset
3. **Test format compliance explicitly** in your evaluation suite
4. **Use format conditioning prompts** (even simple ones like "Respond in JSON:") rather than relying on the model to infer format from context
5. **For structured output tasks**, pair MoF training with grammar-constrained decoding for maximum reliability

Format awareness is an underappreciated dimension of LLM capability that directly impacts the reliability of AI systems in real-world document processing, data extraction, and structured generation applications.
