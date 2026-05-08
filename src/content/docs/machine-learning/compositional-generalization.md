---
title: Compositional Generalization in Neural Networks
description: Understand compositional generalization — the ability to understand and produce novel combinations of known concepts — why neural networks systematically fail at it, and the research approaches designed to close the gap with human-like systematic generalization.
---

Compositional generalization is the capacity to understand and produce a potentially **infinite number of novel combinations** from a finite set of known primitives and rules. A child who learns the words "swim" and "twice" can immediately understand "swim twice" — and "jump twice," "swim three times," and so on — even if they have never heard those exact phrases before. This ability is considered a hallmark of human intelligence and language understanding.

Modern neural networks, including large language models, show surprising **failures** at systematic compositional generalization despite achieving high performance on many language benchmarks. Understanding why, and how to fix it, is a central challenge at the intersection of deep learning, cognitive science, and AI reasoning.

## What Compositionality Means Formally

A system is **compositional** if the meaning of a complex expression can be computed from the meanings of its parts and the rules governing their combination. In logic:

$$\text{meaning}(f(a, b)) = F(\text{meaning}(a), \text{meaning}(b))$$

where $F$ is a compositional combination rule. In natural language, the phrase "the large red ball" can be decomposed as:

- Noun phrase structure: $[\text{Adj}_1\ \text{Adj}_2\ \text{Noun}]$
- Meaning: apply $\text{large}$, apply $\text{red}$, to the concept $\text{ball}$

Systematicity is violated if a model that understands "red ball" and "large cube" cannot generalize to "large red ball" — because it learned the phrases as holistic patterns rather than compositional rules.

## The SCAN Benchmark

**SCAN** (Simple Commands and Navigation, Lake & Baroni, 2018) is the canonical compositional generalization benchmark. Agents must translate simple command strings into action sequences:

- **Train:** `jump` → `JUMP`, `walk left` → `LTURN WALK`
- **Test:** `jump left` → `LTURN JUMP`

The compositional structure is clear: "left" means "turn left before the action," and it should combine with "jump" just as it combines with "walk." However, sequence-to-sequence LSTMs trained on SCAN achieve near-perfect accuracy on i.i.d. test splits but **drop to near-zero** on compositional splits where commands require applying learned modifiers to primitives not seen with those modifiers during training.

### SCAN Split Types

| Split | Train Example | Test Challenge |
| --- | --- | --- |
| Simple | `walk left` | `jump left` (new verb + modifier) |
| Add Primitive | All verbs except `jump` | All commands with `jump` |
| Length | Short commands | Long commands requiring length generalization |
| TMCD | Frequent compounds | Rare compounds from frequent primitives |

Even large pretrained models (T5, GPT) show significant accuracy drops on compositional splits unless explicitly trained with compositional objectives.

## COGS: Compositional Generalization in Semantic Parsing

**COGS** (Kim & Linzen, 2020) extends the benchmark to semantic parsing — mapping English sentences to logical forms. COGS tests 21 compositional generalization types, including:

- **Structural recursion:** Nested relative clauses and PP attachments not seen during training.
- **Argument role generalization:** Verbs seen only as intransitive used in transitive constructions.
- **Noun-to-verb generalization:** Nouns in subject position generalized to object position.

Standard seq2seq transformers achieve 35–80% accuracy on COGS generalization sets, compared to 98%+ on in-distribution splits — a large systematic gap.

## Why Neural Networks Fail at Systematic Generalization

Several mechanisms explain this failure:

### Spurious Correlation Learning

Neural networks are strong pattern matchers. When a training corpus contains "red ball" 1,000 times and "large red ball" 0 times, the model learns the co-occurrence statistics of individual patterns rather than the compositional rule. **The network learns a lookup table rather than an algorithm.**

### Entanglement of Representations

For compositional generalization to work, the representations of "red" and "ball" must be **independently manipulable** — activating "red" and independently activating "ball" should compose predictably. Standard dense representations entangle these features, making independent manipulation difficult.

### Lack of Structural Inductive Bias

Standard attention is **permutation-equivariant** and does not enforce tree structure or argument-role binding. Humans implicitly track syntactic structure (who did what to whom); transformers must learn this structure from data alone, which is unreliable under distribution shift.

### Length Generalization Failure

Autoregressive models fail to generalize to sequences longer than those seen in training. This is partially a positional encoding issue: absolute positional encodings degrade for unseen lengths, and relative encodings must be carefully designed to support length extrapolation.

## Approaches to Compositional Generalization

### Structured Architecture Modifications

**Tree-structured networks** encode syntactic structure explicitly:

- **Tree-LSTMs** process constituency or dependency parse trees, naturally composing sub-tree representations.
- **Recursive Neural Networks** apply the same transformation at every node in a parse tree, enforcing weight sharing across structural positions.
- **Hierarchical Transformers** add tree-structural inductive bias through span-based attention patterns.

These approaches improve compositional generalization but require parse trees as inputs or jointly learning parsing and composition.

### Disentangled Representations

**Object-centric representations** (slot-based models, binding networks) represent each concept as an independent slot — a vector in a shared representational space that can be independently combined. If $\mathbf{z}_\text{red}$ and $\mathbf{z}_\text{ball}$ are independent, composition by concatenation or structured sum produces predictable combinations.

Recent work on **binding via synchrony** (neural circuits that tag co-active features with the same oscillatory phase) provides a biologically-inspired mechanism for this kind of dynamic variable binding.

### Data Augmentation for Compositional Splits

**GECA (Good-Enough Compositional Augmentation)** (Andreas, 2020) generates compositionally novel training examples by substituting semantically equivalent fragments:

If "walk left" and "jump" are both in training, GECA generates "jump left" as an augmented example. Training with GECA substantially closes the compositional generalization gap on SCAN without changing the architecture.

### Meta-Learning for Systematic Generalization

**Meta-learning** trains models to learn from few examples — if the model sees "dax" (novel verb) used in one sentence, it should generalize to "dax twice," "dax left," etc. **MAML**-style approaches learn initialization points that enable rapid compositional generalization from a handful of examples.

### Neuro-Symbolic Hybrid Approaches

**Program synthesis** approaches decompose the task into a neural parser (that produces a structured program) and a symbolic executor (that runs the program):

1. Parse "jump twice" into `REPEAT(JUMP, 2)`.
2. Execute the program symbolically: `[JUMP, JUMP]`.

Because the execution engine is **symbolic and rule-based**, it composes perfectly by construction. The neural component only needs to learn parsing — a simpler task than learning both parsing and composition end-to-end.

Systems like **SCAN → NQGM**, **SCFG-based parsers**, and **DeepMind's Neural Modular Networks** adopt this hybrid strategy and achieve near-perfect compositional generalization on SCAN and COGS with training data sizes orders of magnitude smaller than end-to-end neural systems.

### In-Context Compositional Learning in LLMs

Recent work finds that **large language models show emergent compositional generalization** via few-shot prompting. By providing a few examples of the target compositional rule in the context window, LLMs can apply novel compositions correctly. This suggests that scale and in-context learning partially compensate for the lack of structural inductive bias — but the generalization is brittle and degrades significantly under distribution shift or with longer compositional depth.

## Benchmarks Summary

| Benchmark | Domain | Key Challenge |
| --- | --- | --- |
| SCAN | Synthetic command navigation | Primitive–modifier compositionality |
| COGS | Semantic parsing | Structural recursion, argument roles |
| PCFG | Synthetic function composition | Nested function calls |
| GeoQuery | NL to SQL | Spatial relation compositionality |
| CFQ | NL to SPARQL | Compound divergence splits |
| SLOG | NL to logic | Verb argument structure |

## Broader Significance

Compositional generalization is not just a benchmark problem — it underlies several practical failure modes:

- **Zero-shot instruction following:** A model that cannot compose novel instruction combinations will require example-based prompting for every new task variant.
- **Code generation:** Writing a function that calls a library API in an unusual pattern requires composing known function signatures with new control flow.
- **Mathematical reasoning:** Solving a novel equation type requires composing known algebraic operations in a new configuration.
- **Robustness:** Models without compositional structure are brittle to rephrasing, especially in long-form reasoning tasks where intermediate steps contain novel combinations.

## Summary

Compositional generalization exposes a fundamental gap between statistical pattern matching and rule-based reasoning. Standard neural networks excel at interpolation within the training distribution but fail at the systematic extrapolation that compositional rules support. Closing this gap requires a combination of structured inductive biases, disentangled representations, neuro-symbolic hybrid execution, and training methodologies that reward generalization over memorization — each an active and promising line of research connecting deep learning to classical theories of language and cognition.
