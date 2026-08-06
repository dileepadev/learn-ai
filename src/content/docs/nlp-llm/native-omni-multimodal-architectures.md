---
title: Native Omni-Multimodal Model Architectures
description: Discover native omni models (speech, vision, text) trained end-to-end with unified stream processing for real-time, low-latency audio-to-audio and visual interaction.
---

Early multimodal AI systems relied on **cascaded pipelines**: an Automatic Speech Recognition (ASR) model transcribed audio to text, a text LLM processed the prompt and generated text, and a Text-to-Speech (TTS) model synthesized the final audio output.

While modular, cascaded pipelines incur prohibitive latency (often 1.5 to 3 seconds) and discard critical non-verbal nuances such as emotion, tone, pitch, cadence, background acoustics, and simultaneous visual context.

**Native Omni-Multimodal Architectures** (exemplified by models like GPT-4o and Gemini 1.5 Pro) replace cascaded systems with a **single unified neural network** trained end-to-end across vision, speech, and text modalities simultaneously.

## Cascaded vs. Native Omni Architecture

```
Cascaded Pipeline (High Latency, Loss of Tone):
Audio In -> [ ASR Model ] -> Text -> [ Text LLM ] -> Text -> [ TTS Model ] -> Audio Out

Native Omni-Model (Low Latency <300ms, Full Modality Preservation):
Audio/Vision/Text In  ---> [ Single Transformer Back-Bone ] ---> Audio/Vision/Text Out
```

## Core Technological Pillars

### 1. Neural Audio Tokenization
To process continuous raw audio waveforms ($\text{44.1 kHz}$) natively alongside discrete text tokens, omni models use neural audio codecs (such as **SNAC**, **EnCodec**, or **Descript Audio Codec**).

Audio codecs map continuous waveforms into streams of discrete codebook indices (audio tokens) at low frame rates (e.g., 50–100 tokens per second per channel):

$$x_{\text{audio}}(t) \xrightarrow{\text{Encoder}} z_{\text{quantized}} \xrightarrow{\text{Residual Vector Quantization (RVQ)}} [\text{Token}_1, \text{Token}_2, \dots, \text{Token}_N]$$

### 2. Multi-Stream Token Interleaving & Duplex Processing
Unlike static text generation where the model takes turns responding, native omni models process **full-duplex audio streams**:
- **Input Stream:** Model continuously ingests user audio and visual tokens.
- **Output Stream:** Model emits real-time audio tokens concurrently.
- **Interruption Handling:** If the user speaks while the model is responding, the input stream detects incoming audio tokens and triggers immediate cancellation of the output audio decoder stream.

### 3. Early Fusion Architecture
Text, vision patches (from ViT encoders), and audio codec tokens are projected into a shared embedding space. A single transformer backbone processes all modalities with unified self-attention:

$$\mathbf{E} = [\mathbf{W}_v \cdot \text{VisionTokens} \;||\; \mathbf{W}_t \cdot \text{TextTokens} \;||\; \mathbf{W}_a \cdot \text{AudioTokens}]$$

```
+--------------------------------------------------------------------+
|                      Unified Transformer Layer                     |
+--------------------------------------------------------------------+
       ^                               ^                      ^
  Vision Embeddings             Text Embeddings        Audio Codec Embeddings
```

## Training Methodology for Native Omni Models

Native omni pretraining proceeds in three distinct phases:

1. **Modality Tokenization & Pretraining:** Pretraining the vision ViT encoder, neural audio codec, and language backbone on billions of text, image, and raw audio hours independently.
2. **Cross-Modal Early Fusion Alignment:** Joint training on interleaved datasets (e.g., image-text, video-audio, speech-text conversation transcripts) to align latent representations across all input/output pairs.
3. **Duplex Post-Training & Reinforcement Learning:** Fine-tuning on multi-turn voice conversations using direct preference optimization (DPO) and reinforcement learning to control voice tone, pacing, emphasis, and low latency (<300ms target).

## Latency & Modality Performance Matrix

| Metric | Cascaded Pipeline (ASR + LLM + TTS) | Native Omni-Multimodal Model |
|---|---|---|
| **Audio-to-Audio Latency** | 1,500ms – 3,500ms | 230ms – 320ms |
| **Speech Nuance Retention** | Lost (Flattened to plain text) | Preserved (Tone, emotion, laughter, pitch) |
| **Visual Ingestion Speed** | Frame sampling + OCR pass | Native continuous ViT embedding stream |
| **Interruption Flexibility** | Hard reset required | Dynamic token stream cancellation |
| **Compute Overhead** | 3 Separate Models | Single Shared Transformer Weights |

## Conceptual PyTorch Code: Interleaved Multi-Stream Forward Pass

```python
import torch
import torch.nn as nn

class NativeOmniTransformer(nn.Module):
    def __init__(self, vocab_size, audio_codebook_size, d_model=4096):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.audio_embed = nn.Embedding(audio_codebook_size, d_model)
        self.vision_proj = nn.Linear(768, d_model)  # Project ViT patch tokens
        
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=32, batch_first=True),
            num_layers=32
        )
        
        self.text_head = nn.Linear(d_model, vocab_size)
        self.audio_head = nn.Linear(d_model, audio_codebook_size)

    def forward(self, text_ids=None, audio_ids=None, vision_patches=None):
        embeddings = []
        
        if vision_patches is not None:
            embeddings.append(self.vision_proj(vision_patches))
        if text_ids is not None:
            embeddings.append(self.text_embed(text_ids))
        if audio_ids is not None:
            embeddings.append(self.audio_embed(audio_ids))
            
        # Concatenate modal streams along sequence dimension
        x = torch.cat(embeddings, dim=1)
        hidden_states = self.backbone(x)
        
        text_logits = self.text_head(hidden_states)
        audio_logits = self.audio_head(hidden_states)
        
        return text_logits, audio_logits
```

## Summary

Native omni-multimodal architectures represent a fundamental paradigm shift from text-centric models with modality adapters to truly unified perceptual models. By processing text, vision, and real-time audio streams in a single transformer backbone, omni models deliver human-like conversational responsiveness and cross-modal understanding.

## Further Reading

- OpenAI (2024), *GPT-4o System Card & Architectural Overview*
- Defossez et al. (2022), *High Fidelity Neural Audio Compression (EnCodec)*
- Team Gemini (2024), *Gemini 1.5: Unlocking Multimodal Reasoning Across Millions of Tokens*
