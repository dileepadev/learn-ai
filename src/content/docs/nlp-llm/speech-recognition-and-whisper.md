---
title: Automatic Speech Recognition and Whisper
description: Master modern Automatic Speech Recognition (ASR), acoustic modeling, Connectionist Temporal Classification (CTC), and the end-to-end Whisper transformer architecture.
---

**Automatic Speech Recognition (ASR)** is the technology that converts spoken acoustic audio signals into readable text. For decades, ASR was dominated by complex statistical pipelines combining Gaussian Mixture Models (GMMs), Hidden Markov Models (HMMs), pronunciation lexicons, and n-gram language models (such as the Kaldi toolkit).

In recent years, deep learning has unified these disparate components into **End-to-End (E2E) Neural Speech Systems**. The release of OpenAI's **Whisper** in 2022 marked a major inflection point: trained weakly supervised on 680,000 hours of diverse multilingual web audio, Whisper demonstrated unprecedented robustness to accents, background noise, and specialized technical jargon without requiring fine-tuning.

---

## Evolution of Speech Recognition Pipelines

```
Traditional Cascaded ASR (Complex, Modular):
Audio -> [ Feature (MFCC) ] -> [ Acoustic Model (HMM) ] -> [ Pronunciation Lexicon ] -> [ N-Gram LM ] -> Text

Connectionist Temporal Classification (CTC / Conformer):
Audio -> [ Log-Mel Spectrogram ] -> [ Deep Neural Network ] -> CTC Loss with Blank Tokens -> Text

Encoder-Decoder Whisper (End-to-End Sequence-to-Sequence):
Audio -> [ 80-Channel Log-Mel ] -> [ Audio Transformer Encoder ] -> [ Multitask Autoregressive Decoder ] -> Text + Timestamps
```

---

## 1. Acoustic Features: Log-Mel Spectrograms

Raw audio waveforms are continuous 1D pressure waves sampled at high frequency (typically 16 kHz, or 16,000 samples per second). Deep neural networks do not process raw waveforms directly; instead, they convert audio into a 2D time-frequency representation via the **Short-Time Fourier Transform (STFT)**.

1. The waveform is chopped into overlapping frames (e.g., 25ms windows every 10ms).
2. STFT computes the discrete Fourier transform for each window to obtain frequency magnitudes.
3. Frequencies are passed through a **Mel-scale filter bank** that mimics the non-linear pitch perception of the human cochlea:

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

4. Taking the logarithm of energy yields an **80-channel Log-Mel Spectrogram** tensor of shape $(80, T)$, where $T$ corresponds to time frames.

---

## 2. Connectionist Temporal Classification (CTC)

In speech recognition, the audio input sequence is typically much longer than the target text sequence (e.g., a 5-second audio clip has 500 frames, but only 10 transcribed words). Furthermore, the exact alignment between audio frames and phonemes is unknown.

**Connectionist Temporal Classification (CTC)** (Graves et al., 2006) resolves this by introducing an explicit **blank token** $\epsilon$:
- At each audio frame $t$, the network outputs a probability distribution over vocabulary characters plus $\epsilon$.
- An alignment collapse function $\mathcal{B}$ removes sequential duplicate tokens and blanks:
  $$\mathcal{B}(\text{"c - a - a - t - -"}) \to \text{"cat"}$$
- The CTC loss maximizes the total marginal probability of all valid alignments that collapse to the ground-truth text:

$$P(Y \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^T P(\pi_t \mid x_t)$$

---

## 3. The Whisper Architecture

Whisper adopts a standard **encoder-decoder transformer architecture** operating on 30-second audio chunks:

```
30-Second Audio Chunk (16 kHz)
             │
             ▼
  [ 80-Channel Log-Mel ]
             │
             ▼
[ Conv1D Layer (Stride 2) ] ──► Compresses time dimension by 2x
             │
             ▼
 [ Transformer Encoder ]    ──► Contextualized Acoustic Embeddings
             │
             ├──────────────────────────┐ Cross-Attention
             ▼                          ▼
 [ Multitask Autoregressive Decoder ] ◄── Token Prompt: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
             │
             ▼
     Generated Tokens
```

### The Multitask Special Token Format
Rather than maintaining separate models for speech recognition, language identification, speech translation, and timestamp alignment, Whisper frames all speech capabilities as an **autoregressive sequence generation task** controlled by special prompt tokens:

```
[<|startoftranscript|>] -> [<|language_id|>] -> [<|task|>] -> [<|timestamps_flag|>] -> [Text Tokens...]
```

1. `<|startoftranscript|>`: Tells the decoder to begin generation.
2. `<|language_id|>`: The model identifies the spoken language (e.g., `<|en|>`, `<|es|>`, `<|zh|>`).
3. `<|task|>`: Specifies whether to perform verbatim `<|transcribe|>` or English `<|translate|>`.
4. `<|notimestamps|>` vs `<|timestamps|>`: Directs the model to interleave precise phrase-level start/end timestamps (quantized in 20ms steps, e.g., `<|0.00|>` ... `<|2.40|>`).

---

## Temperature Fallback and Hallucination Mitigation

During long-form audio transcription, autoregressive decoders can fall into **repetition loops** or fabricate hallucinations during silent intervals.

Whisper implements a **temperature fallback strategy**:
1. It first attempts greedy decoding at temperature $T = 0.0$.
2. It evaluates heuristic failure metrics:
   - **Repetition penalty check:** Is the gzip compression ratio of the text abnormally high?
   - **Average log probability:** Did average token confidence drop below threshold $\tau_{\text{logprob}}$?
   - **No-speech probability:** Does the `<|nospeech|>` token probability exceed $0.6$ while text was emitted?
3. If any failure metric triggers, the system discards the hypothesis and retries decoding with non-zero temperature ($T \in \{0.2, 0.4, 0.6, 0.8\}$), introducing exploration to break out of failure loops.

---

## Python Implementation with `faster-whisper`

In production, **`faster-whisper`** (powered by CTranslate2) provides up to $4\times$ faster execution with 8-bit quantization:

```python
from faster_whisper import WhisperModel

# Initialize Whisper model on GPU with INT8 computation
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

# Transcribe audio with automatic language detection and word timestamps
segments, info = model.transcribe(
    "conference_call.mp3",
    beam_size=5,
    word_timestamps=True
)

print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

---

## Key Takeaways

- Modern ASR relies on Log-Mel Spectrograms to convert raw 1D acoustic pressures into perceptually-scaled 2D time-frequency tensors.
- CTC enables alignment-free training, while encoder-decoder transformers (Whisper) formulate transcription, translation, and timestamp alignment under a unified token format.
- Weakly supervised training on massive, diverse datasets eliminates the brittle domain-mismatch failures of traditional acoustic models.
