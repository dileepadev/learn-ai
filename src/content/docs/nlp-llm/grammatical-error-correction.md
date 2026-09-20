---
title: Grammatical Error Correction - Automatically Fixing Written Text
description: Learn how grammatical error correction systems detect and fix grammar, spelling, and fluency errors, and how they're evaluated.
---

Grammatical error correction (GEC) automatically detects and fixes errors in written text — subject-verb agreement, article usage, spelling, preposition choice, and broader fluency issues — producing a corrected version of the input.

```text
Input:     "She go to school every days and like it very much."
Corrected: "She goes to school every day and likes it very much."
```

## Framing GEC as Sequence-to-Sequence

Modern GEC systems typically frame the task as sequence-to-sequence generation: a model reads the erroneous sentence and generates the corrected version directly, using the same encoder-decoder or decoder-only transformer architectures used for translation, since correcting grammar is conceptually similar to translating from "learner English" to "standard English." Fine-tuning a general-purpose language model on parallel erroneous/corrected sentence pairs is now the standard approach, and produces fluent corrections that handle errors classic rule-based grammar checkers miss.

## Edit-Based Approaches

An alternative framing predicts a sequence of discrete edit operations (insert, delete, replace, keep) applied to the original text token by token, rather than generating the whole corrected sentence from scratch. Edit-based models are typically faster at inference and make it easier to trace exactly which spans were changed and why, which matters for applications that want to highlight specific corrections to a user rather than silently replacing their entire sentence.

## Evaluation Challenges

Multiple valid corrections often exist for the same error — "every days" could become "every day" or be rephrased as "daily" — so evaluation metrics like the ERRANT toolkit compare predicted edits against a set of human-annotated reference edits rather than requiring an exact string match against one single reference correction. GEC systems are typically evaluated with a metric that weights precision more heavily than recall, since incorrectly "fixing" already-correct text is often judged as more harmful to user trust than missing a real error.

## Practical Guidance

For language-learning applications, prefer showing users the specific suggested edits with brief explanations rather than silently rewriting their text, since the pedagogical value comes from understanding the correction, not just receiving corrected output. For general writing-assistant use cases, tune the correction model's aggressiveness deliberately — a model that over-corrects stylistic choices as "errors" frustrates users who deliberately wrote in a particular voice.
