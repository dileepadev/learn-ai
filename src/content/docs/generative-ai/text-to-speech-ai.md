---
title: Text-to-Speech AI
description: Explore the evolution of AI-powered text-to-speech systems — from WaveNet and Tacotron to modern end-to-end models like VITS, XTTS, and VoiceBox. Covers acoustic feature generation, neural vocoders, zero-shot voice cloning, multilingual TTS, and practical usage with open-source libraries.
---

**Text-to-speech (TTS)** synthesis has been transformed by deep learning from a pipeline of hand-crafted phonetic rules and signal processing into end-to-end neural systems that produce speech indistinguishable from human recordings. Modern TTS models can clone a voice from a few seconds of audio, speak naturally in dozens of languages, and control prosody (emphasis, pacing, emotional tone) through prompt instructions — capabilities that were science fiction a decade ago.

## The TTS Pipeline

Traditional and neural TTS share a common conceptual pipeline, even as neural systems increasingly collapse it into a single model:

```
Text → [Text Analysis] → Phonemes/Tokens → [Acoustic Model] → Mel Spectrogram → [Vocoder] → Waveform
         (normalization,     (pronunciation,    (duration, pitch,   (80-channel log      (raw audio
          G2P conversion)    stress, prosody)    energy prediction)  mel spectrogram)      samples)
```

### Text Analysis and Phoneme Conversion

Before acoustic synthesis, raw text must be normalized (expanding "Dr." → "Doctor", "12.5" → "twelve point five") and converted to phonemes. Neural TTS increasingly handles this implicitly via character or byte-pair encoding, but explicit grapheme-to-phoneme (G2P) conversion remains important for proper nouns and technical vocabulary.

## WaveNet: Autoregressive Waveform Synthesis

WaveNet (van den Oord et al., DeepMind, 2016) was the first neural model to produce human-quality speech by generating raw audio samples autoregressively. Each sample $x_t$ is conditioned on all previous samples:

$$p(x) = \prod_{t=1}^T p(x_t | x_1, \ldots, x_{t-1})$$

The model uses **dilated causal convolutions** with exponentially increasing dilation rates (1, 2, 4, 8, ..., 512) to efficiently capture long-range temporal dependencies without recurrence:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNetBlock(nn.Module):
    """
    One WaveNet residual block with dilated causal convolution.
    Uses gated activation: tanh(W_f * x) ⊙ sigmoid(W_g * x)
    Conditioning on mel spectrogram via 1×1 convolution.
    """
    def __init__(self, residual_channels: int, dilation: int,
                 kernel_size: int = 2, condition_channels: int = 80):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # Dilated causal conv: padding = (kernel-1)*dilation on left only
        self.conv = nn.Conv1d(
            residual_channels, residual_channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        
        # 1×1 conv for conditioning signal (mel spectrogram, upsampled)
        self.cond_conv = nn.Conv1d(condition_channels, residual_channels * 2, 1)
        
        # Residual and skip connections
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, residual_channels, 1)

    def forward(self, x: torch.Tensor,
                condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, T) — residual features
        condition: (B, 80, T) — upsampled mel spectrogram
        """
        # Causal: remove the future-looking padding
        h = self.conv(x)[..., :x.shape[-1]]
        h = h + self.cond_conv(condition)
        
        # Gated activation unit (GAU): tanh ⊙ sigmoid
        h_tanh, h_sigmoid = h.chunk(2, dim=1)
        h = torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)
        
        # Skip and residual
        skip = self.skip_conv(h)
        residual = self.res_conv(h) + x
        return residual, skip


class WaveNet(nn.Module):
    def __init__(self, residual_channels: int = 64, num_layers: int = 30,
                 dilation_cycle: int = 10, condition_channels: int = 80):
        super().__init__()
        self.input_conv = nn.Conv1d(1, residual_channels, 1)
        
        dilations = [2 ** (i % dilation_cycle) for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_channels, d, condition_channels=condition_channels)
            for d in dilations
        ])
        
        # Output layers: residual_channels → 256 mu-law quantization bins
        self.output_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, residual_channels, 1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, 256, 1)   # 256-class output (mu-law)
        )

    def forward(self, audio: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, 1, T) — quantized waveform input (shifted by 1 for prediction)
        mel: (B, 80, T) — conditioning mel spectrogram (upsampled to audio rate)
        Returns: (B, 256, T) — logits over 256 mu-law quantization levels
        """
        x = self.input_conv(audio)
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x, mel)
            skip_sum = skip_sum + skip
        return self.output_net(skip_sum)
```

WaveNet's autoregressive generation is slow: generating 1 second of 24kHz audio requires 24,000 sequential model calls. This motivated parallel and non-autoregressive vocoders.

## Tacotron 2: Text to Mel Spectrogram

Tacotron 2 (Wang et al., Google, 2018) is a seq2seq model with attention that generates mel spectrograms from character sequences. It is typically paired with WaveNet or HiFi-GAN as the vocoder:

- **Encoder**: character embedding + CBHG (convolution bank + highway + GRU)
- **Decoder**: autoregressive attention-based RNN predicting mel frames
- **Location-sensitive attention**: attends to encoder outputs, tracking position
- **Post-net**: 5-layer CNN refining the raw mel spectrogram prediction

## FastSpeech 2: Non-Autoregressive Parallel TTS

FastSpeech 2 eliminates the autoregressive bottleneck by predicting mel spectrogram frames in parallel using a **duration predictor** to align phonemes to frames:

```python
class DurationPredictor(nn.Module):
    """
    Predicts how many mel frames each phoneme should span.
    Trained with ground-truth durations from an alignment model (e.g., MFA).
    At inference, integer durations are used to upsample phoneme embeddings.
    """
    def __init__(self, d_model: int = 256, conv_channels: int = 256,
                 kernel_size: int = 3, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = d_model if i == 0 else conv_channels
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_ch, conv_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.LayerNorm(conv_channels)
            ))
        self.linear = nn.Linear(conv_channels, 1)   # log duration prediction

    def forward(self, phoneme_embeddings: torch.Tensor) -> torch.Tensor:
        """
        phoneme_embeddings: (B, N_phonemes, d_model)
        Returns: (B, N_phonemes) — predicted log durations (in frames)
        """
        x = phoneme_embeddings.transpose(1, 2)   # (B, d_model, N)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)   # (B, N, conv_channels)
        return self.linear(x).squeeze(-1)   # (B, N)


def length_regulate(phoneme_embeddings: torch.Tensor,
                    durations: torch.Tensor) -> torch.Tensor:
    """
    Upsample phoneme embeddings by repeating each embedding
    according to its predicted duration in frames.
    
    phoneme_embeddings: (B, N, D)
    durations: (B, N) integer frame counts
    Returns: (B, T_mel, D) — frame-level representations
    """
    outputs = []
    for emb, dur in zip(phoneme_embeddings, durations):
        # Repeat each phoneme embedding `dur` times along time axis
        repeated = torch.repeat_interleave(emb, dur.long(), dim=0)
        outputs.append(repeated)
    return torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)
```

FastSpeech 2 also predicts **pitch** (F0 fundamental frequency) and **energy** (frame amplitude) from phoneme embeddings, enabling fine-grained prosody control by modifying predicted values at inference time.

## VITS: End-to-End Variational TTS

VITS (Kim et al., 2021) is a fully end-to-end model that combines a VAE, normalizing flow, and HiFi-GAN vocoder — learning to map text directly to waveforms with no intermediate acoustic features:

- **Posterior encoder**: encodes linear spectrogram to latent $z$
- **Prior encoder**: encodes phoneme sequence to prior distribution $p(z|c_{text})$
- **Decoder**: HiFi-GAN vocoder mapping $z$ to waveform
- **Flow network**: transforms $z$ to align posterio and prior distributions
- **Duration predictor**: stochastic (samples durations from log-normal)
- **Discriminator**: multi-period + multi-scale discriminator for adversarial training

VITS produces more natural prosody than pipeline systems because the entire system is trained end-to-end — prosody, duration, and waveform quality are jointly optimized rather than independently.

## Modern Systems: Zero-Shot Voice Cloning

Contemporary TTS models can clone a voice from a few seconds of audio without fine-tuning:

```python
# XTTS v2 (Coqui): multilingual zero-shot voice cloning
from TTS.api import TTS

# Load XTTS v2 — supports 17 languages
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Clone voice from reference audio (minimum 3 seconds recommended)
tts.tts_to_file(
    text="Hello, this is a demonstration of zero-shot voice cloning.",
    speaker_wav="reference_speaker.wav",   # 5-10 seconds of clean target voice
    language="en",
    file_path="output.wav"
)

# Multilingual: same voice, different language
tts.tts_to_file(
    text="Bonjour, ceci est une démonstration du clonage de voix.",
    speaker_wav="reference_speaker.wav",
    language="fr",
    file_path="output_french.wav"
)
```

**VoiceBox** (Meta, 2023) is a flow-matching-based TTS model trained on 50,000 hours of audio. It frames TTS as in-context learning: given a few seconds of audio and a matching transcript, it predicts the same speaker's voice for new text — similar to how LLMs do few-shot learning.

**F5-TTS** uses a Diffusion Transformer (DiT) operating directly on audio tokens. It achieves state-of-the-art naturalness and speaker similarity while being trainable on consumer hardware.

## Neural Vocoders

The vocoder converts mel spectrograms (or latent representations) to audio waveforms:

| Vocoder | Architecture | Speed | Quality |
| --- | --- | --- | --- |
| WaveNet | Dilated causal conv (autoregressive) | ~0.02× RT | Very high |
| WaveGlow | Normalizing flow (parallel) | ~25× RT | High |
| MelGAN | GAN (parallel) | >100× RT | Good |
| HiFi-GAN | GAN with multi-period discriminator | >100× RT | Very high |
| BigVGAN | GAN with anti-aliased activations | >100× RT | State-of-art |

HiFi-GAN is the de facto standard vocoder: it runs in real-time on CPU and produces near-WaveNet quality. The multi-period discriminator (MPD) captures periodic structures at multiple scales (periods 2, 3, 5, 7, 11), while the multi-scale discriminator (MSD) operates at different audio resolutions.

## Applications

- **Accessibility**: Screen readers, reading aids for visual impairment
- **Content creation**: Audiobooks, podcasts, video narration without voice actors
- **Voice assistants**: Natural-sounding responses in conversational AI
- **Localization**: Dubbing and translation while preserving speaker identity
- **Gaming**: Dynamic NPC dialogue without recording every line
- **Personalized learning**: Adaptive tutoring in the learner's preferred voice

The combination of zero-shot cloning, multilingual capability, and real-time inference has made high-quality TTS production-ready across virtually every domain. The remaining open problems are extreme expressivity (whispers, shouting, emotional speech), robustness to rare words and code-switching, and detecting and preventing misuse for voice fraud and deepfake audio.
