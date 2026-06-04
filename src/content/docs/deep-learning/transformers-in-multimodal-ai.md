---
title: "Transformers in Multimodal AI: Vision, Audio, and Beyond"
description: "Explore how transformer architecture has expanded beyond language to power multimodal AI systems — vision transformers, audio transformers, and cross-modal transformers."
---

The transformer architecture has become the universal backbone for multimodal AI. Originally designed for sequence-to-sequence translation, transformers now process images, audio, video, and combine multiple modalities. This guide covers how transformers work across modalities.

## Vision Transformers (ViT)

Vision transformers apply the transformer architecture to image understanding:

```python
import torch
import torch.nn as nn
from einops import rearrange

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        
        # Patch embedding: Conv projection
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Class token (for classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model)
        )
        
        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.head = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, d_model, h', w')
        x = rearrange(x, 'b d h w -> b (h w) d')  # (batch, n_patches, d_model)
        
        # Add class token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches+1, d_model)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer encoding
        for layer in self.encoder:
            x = layer(x)
        
        # Use CLS token for classification
        x = x[:, 0, :]  # (batch, d_model)
        return self.head(x)
```

### Swin Transformer: Hierarchical Vision Transformer

```python
class SwinTransformer(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        # Patch embedding (overlapping patches)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        
        # Build stages with shifted windows
        self.stages = nn.ModuleList()
        for i, (depth, num_h) in enumerate(zip(depths, num_heads)):
            stage = SwinStage(
                dim=embed_dim * (2 ** i),
                depth=depth,
                num_heads=num_h,
                window_size=7,
                downsample=(i < len(depths) - 1)
            )
            self.stages.append(stage)
    
    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        return x

class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.attention = WindowAttention(dim, num_heads)
        self.shift_size = window_size // 2
    
    def forward(self, x, mask=None):
        # Shift features for shifted window attention
        shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        # Apply attention in windows
        # ... window partitioning and attention
        
        # Shift back
        return torch.roll(attended, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```

## Audio Transformers

### Audio Spectrogram Transformer

```python
class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, num_classes=527):
        super().__init__()
        
        # Convert audio to spectrogram
        self.spec = nn.Sequential(
            MelSpectrogram(n_mels=128),
            nn.Log1p(),  # Log-scale
        )
        
        # Time-frequency patch embedding
        self.patch_embed = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        
        # Standard ViT transformer
        self.transformer = VisionTransformer(
            image_size=None,  # Variable length
            patch_size=None,
            in_channels=768,
            d_model=768,
            n_heads=12,
            n_layers=12,
        )
        
        # Classification
        self.fc = nn.Linear(768, num_classes)
    
    def forward(self, waveform):
        # waveform: (batch, samples) or (batch, 1, samples)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        # Convert to spectrogram
        spectrogram = self.spec(waveform)  # (batch, 1, n_mels, time)
        
        # Apply transformer
        x = self.patch_embed(spectrogram)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add class token and position embeddings
        x = self.transformer.add_tokens(x)
        
        # Transformer encode
        x = self.transformer.encoder(x)
        
        # Classify
        return self.fc(x[:, 0])  # CLS token
```

### Whisper: Speech Recognition Transformer

```python
class Whisper(nn.Module):
    def __init__(self, n_mels=80, n_audio_ctx=1500, n_text_ctx=448, d_model=1024):
        super().__init__()
        
        # Audio encoder
        self.encoder = AudioEncoder(n_mels, d_model)
        
        # Text decoder
        self.decoder = CrossModalDecoder(
            d_model=d_model,
            n_text_ctx=n_text_ctx,
            n_heads=16,
            n_layers=12,
        )
        
        # Token embeddings
        self.token_embed = nn.Embedding(51865, d_model)
    
    def forward(self, audio, input_ids):
        # Encode audio
        audio_features = self.encoder(audio)  # (batch, n_audio_ctx, d_model)
        
        # Encode text tokens
        text_features = self.token_embed(input_ids)  # (batch, n_text_ctx, d_model)
        
        # Cross-attention decode
        output = self.decoder(text_features, audio_features)
        return output
```

## Cross-Modal Transformers

### CLIP: Connecting Vision and Language

CLIP learns a shared embedding space for images and text:

```python
class CLIP(nn.Module):
    def __init__(self, vision_config, text_config, projection_dim=512):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = VisionTransformer(**vision_config)
        
        # Text encoder
        self.text_encoder = TransformerEncoder(**text_config)
        
        # Projection heads
        self.visual_projection = nn.Linear(vision_config.d_model, projection_dim)
        self.text_projection = nn.Linear(text_config.d_model, projection_dim)
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, texts):
        # Encode images
        image_features = self.vision_encoder(images)
        image_features = self.visual_projection(image_features)
        image_features = F.normalize(image_features, dim=-1)
        
        # Encode text
        text_features = self.text_encoder(texts)
        text_features = self.text_projection(text_features[:, 0, :])  # CLS
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        logits = torch.matmul(image_features, text_features.T) * self.temperature.exp()
        
        return logits
    
    def contrastive_loss(self, logits):
        """CLIP contrastive loss (symmetric)."""
        labels = torch.arange(len(logits)).to(logits.device)
        loss_i = F.cross_entropy(logits, labels)  # Image-to-text
        loss_t = F.cross_entropy(logits.T, labels)  # Text-to-image
        return (loss_i + loss_t) / 2
```

### BLIP-2: Bootstrapped Language-Image Pre-training

BLIP-2 connects a frozen vision encoder to a frozen LLM:

```python
class BLIP2(nn.Module):
    def __init__(self, vision_config, llm_config, qformer_config):
        super().__init__()
        
        # Frozen vision encoder
        self.vision_encoder = VisionTransformer(**vision_config)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Q-Former: Learns to extract visual features for LLM
        self.qformer = TransformerEncoder(**qformer_config)
        self.qformer_proj = nn.Linear(qformer_config.d_model, llm_config.hidden_size)
        
        # Frozen LLM
        self.llm = AutoModelForCausalLM.from_config(llm_config)
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # LLM embeddings (trainable)
        self.llm_embedding = self.llm.get_input_embeddings()
    
    def forward(self, images, input_ids):
        # Extract visual features
        image_embeds = self.vision_encoder(images)
        
        # Query transformer to extract relevant visual features
        query_tokens = nn.Parameter(torch.randn(1, 32, qformer_config.d_model))
        query_embeds = self.qformer(
            query_tokens, 
            encoder_hidden_states=image_embeds
        )
        
        # Project to LLM embedding space
        query_embeds = self.qformer_proj(query_embeds)
        
        # Get text embeddings
        text_embeds = self.llm_embedding(input_ids)
        
        # Combine: [text tokens...][visual query tokens]
        combined_embeds = torch.cat([text_embeds, query_embeds], dim=1)
        
        # LLM forward
        outputs = self.llm(inputs_embeds=combined_embeds)
        
        return outputs.logits
```

## Multimodal Fusion Techniques

### Early Fusion

```python
class EarlyFusion(nn.Module):
    """Fuse modalities at input level."""
    def __init__(self, vision_dim, audio_dim, d_model):
        super().__init__()
        # Project both to same dimension, concatenate
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
    
    def forward(self, vision, audio):
        v = self.vision_proj(vision)
        a = self.audio_proj(audio)
        fused = torch.cat([v, a], dim=-1)  # (batch, seq, 2*d_model)
        return fused
```

### Late Fusion

```python
class LateFusion(nn.Module):
    """Process separately, fuse at decision level."""
    def __init__(self, vision_model, audio_model, fusion_dim):
        super().__init__()
        self.vision_model = vision_model
        self.audio_model = audio_model
        self.fusion_classifier = nn.Sequential(
            nn.Linear(vision_model.output_dim + audio_model.output_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, num_classes)
        )
    
    def forward(self, vision, audio):
        v = self.vision_model(vision)
        a = self.audio_model(audio)
        combined = torch.cat([v, a], dim=-1)
        return self.fusion_classifier(combined)
```

### Cross-Attention Fusion

```python
class CrossAttentionFusion(nn.Module):
    """Fuse modalities using cross-attention."""
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(dim, n_heads)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, vision, audio):
        # Vision as query, audio as key/value
        fused, _ = self.cross_attention(vision, audio, audio)
        fused = self.norm(vision + fused)
        return fused
```

## Video Transformers

### Video Swin Transformer

```python
class VideoSwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 3D patch embedding: (T, H, W) -> tokens
        self.patch_embed = nn.Conv3d(3, 96, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        
        # 3D positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 156, 96))
        
        # 3D Swin transformer blocks
        self.stages = nn.ModuleList([
            VideoSwinBlock(dim=96, depth=2, num_heads=3),
            VideoSwinBlock(dim=192, depth=2, num_heads=6),
            VideoSwinBlock(dim=384, depth=6, num_heads=12),
            VideoSwinBlock(dim=768, depth=2, num_heads=24),
        ])
    
    def forward(self, video):
        # video: (batch, frames, channels, height, width)
        x = video.transpose(1, 2)  # (batch, channels, frames, height, width)
        x = self.patch_embed(x)  # (batch, dim, time', h', w')
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        x = x + self.pos_embed
        
        for stage in self.stages:
            x = stage(x)
        
        return x
```

### TimeSformer: Space-Time Attention

```python
class TimeSformer(nn.Module):
    def __init__(self, n_frames=8):
        super().__init__()
        # Separate attention patterns:
        # 1. Divided space-time attention
        # 2. Joint space-time attention
        # 3. Sparse space-time attention
        pass
    
    def forward(self, video_tokens):
        # Each token attends to:
        # - Other tokens at same spatial position (time attention)
        # - Other spatial tokens at same time (space attention)
        pass
```

## Practical Applications

### Image Captioning

```python
class ImageCaptioningModel(nn.Module):
    def __init__(self, vision_model, text_decoder):
        super().__init__()
        self.vision_encoder = vision_model
        self.text_decoder = text_decoder
    
    def forward(self, images, input_ids):
        # Encode image
        image_features = self.vision_encoder(images)
        
        # Decode text
        outputs = self.text_decoder(
            input_ids=input_ids,
            encoder_hidden_states=image_features,
        )
        return outputs.logits
    
    def generate(self, image, max_length=50):
        # Autoregressive generation
        image_features = self.vision_encoder(image)
        
        input_ids = torch.tensor([[bos_token_id]])
        
        for _ in range(max_length):
            outputs = self.text_decoder(
                input_ids=input_ids,
                encoder_hidden_states=image_features,
            )
            next_token = outputs.logits[:, -1, :].argmax()
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token == eos_token_id:
                break
        
        return input_ids
```

### Visual Question Answering

```python
class VQAModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, classifier):
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.classifier = classifier
    
    def forward(self, image, question):
        # Encode image
        image_features = self.vision_encoder(image)
        
        # Encode question
        text_features = self.text_encoder(question)
        
        # Fuse and classify
        fused = cross_attention_fusion(image_features, text_features)
        answer = self.classifier(fused[:, 0])  # CLS token
        return answer
```

The transformer has truly become the "general purpose processor" for AI. From vision to audio to video to cross-modal understanding, the same architectural principles — attention, normalization, feed-forward networks — transfer across modalities, enabling unified approaches to understanding our multimodal world.