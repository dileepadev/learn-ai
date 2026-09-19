---
title: Connectionist Temporal Classification (CTC) Explained
description: Learn how CTC loss lets sequence models train on speech and handwriting recognition without requiring frame-level alignment labels.
---

Connectionist Temporal Classification (CTC) is a loss function that lets a neural network learn to map an input sequence (like audio frames) to an output sequence (like text) of a different, unknown length, without requiring labeled alignment between individual input frames and output symbols.

## The Alignment Problem CTC Solves

Speech recognition training data typically provides an audio clip and its correct transcript, but not which exact audio frame corresponds to which letter or phoneme — manually annotating frame-level alignment for large speech datasets would be prohibitively expensive. CTC sidesteps this by summing over all possible valid alignments between the input and output during training, rather than requiring one single ground-truth alignment to be specified upfront.

## The Blank Token and Collapsing Rule

CTC introduces a special blank symbol and a "collapsing" rule for interpreting the network's frame-by-frame output: repeated consecutive symbols are merged into one, and blank tokens are removed, which is what allows a fixed-length sequence of per-frame predictions to map onto a shorter, variable-length text output.

```text
raw per-frame output: "h h e _ l l l o _"    (_ = blank)
collapse repeats:      "h e _ l o _"
remove blanks:         "h e l o"
```

This collapsing rule also correctly handles genuinely repeated letters in the true output (like the double "l" in "hello") by requiring a blank token between the two intended repetitions, distinguishing "one long l held across frames" from "two separate l sounds."

## Training and Inference

During training, CTC's forward-backward algorithm efficiently sums the probability over every alignment path that would collapse to the correct target sequence, giving a tractable loss despite the exponentially large number of possible frame-to-symbol alignments. At inference, a simple greedy decoding (take the most likely symbol per frame, then collapse) is fast but not always optimal — beam search decoding, sometimes combined with an external language model, generally produces more accurate transcriptions by considering multiple candidate sequences rather than only the single greedy path.

## CTC vs. Attention-Based Sequence Models

CTC assumes the output sequence's symbols occur in the same order as the corresponding input regions (monotonic alignment), which holds naturally for speech-to-text but not for tasks like translation where word order can reorder substantially between languages. Modern speech recognition systems like Whisper instead use attention-based encoder-decoder architectures that don't require this monotonic assumption, though CTC and hybrid CTC-attention approaches remain common in streaming and low-latency speech recognition systems where CTC's simpler, faster decoding is a meaningful practical advantage.

## Practical Guidance

Reach for CTC-based architectures when building low-latency, streaming speech recognition where decoding speed and the monotonic input-output alignment assumption both hold. For offline, high-accuracy transcription without strict latency constraints, attention-based or hybrid models generally achieve better accuracy at the cost of higher inference latency.
