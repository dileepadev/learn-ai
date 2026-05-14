---
title: "Multimodal Embedding Models: Bridging Vision and Language"
description: "Learn about CLIP, SigLIP, and other multimodal embedding models that enable cross-modal retrieval, zero-shot image classification, and vision-language understanding."
---

Multimodal embedding models map images and text into a shared vector space, enabling powerful capabilities like cross-modal search, zero-shot classification, and image-text understanding. This guide covers the architecture, training, and applications of these models.

## Why Multimodal Embeddings?

Traditional embeddings work on a single modality. Multimodal embeddings let you:

- **Search images with text**: "Find pictures of sunset over mountains."
- **Classify images without training**: Use text prompts for zero-shot classification.
- **Find similar images**: Compare images directly using the embedding space.
- **Cross-modal retrieval**: Retrieve documents based on images and vice versa.

## CLIP: Learning Transferable Visual Models

CLIP (Contrastive Language-Image Pre-training) revolutionized multimodal learning by training on 400 million image-text pairs from the internet.

### CLIP Architecture

```python
class CLIPModel(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.vision_encoder = ViTModel(vision_config)
        self.text_encoder = TransformerModel(text_config)
        self.projection = nn.Linear(vision_config.hidden_size, config.projection_dim)
        self.text_projection = nn.Linear(text_config.hidden_size, config.projection_dim)
    
    def forward(self, image, text):
        # Encode image
        image_embeds = self.vision_encoder(image).last_hidden_state[:, 0]  # CLS token
        image_embeds = self.projection(image_embeds)  # (batch, projection_dim)
        image_embeds = F.normalize(image_embeds, dim=-1)
        
        # Encode text
        text_embeds = self.text_encoder(text).last_hidden_state[:, 0]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Compute similarity
        logits = image_embeds @ text_embeds.T  # (batch, batch)
        return logits
```

### CLIP Training: Contrastive Loss

```python
def clip_loss(image_logits, text_logits, temperature=0.07):
    """CLIP uses symmetric cross-entropy loss."""
    # Image-to-text loss
    labels = torch.arange(image_logits.shape[0]).to(image_logits.device)
    i2t_loss = F.cross_entropy(image_logits / temperature, labels)
    
    # Text-to-image loss
    t2i_loss = F.cross_entropy(text_logits / temperature, labels)
    
    return (i2t_loss + t2i_loss) / 2
```

The key insight: for each batch, images should match their corresponding text descriptions and not match unrelated text.

### Zero-Shot Classification with CLIP

```python
def zero_shot_classify(model, image, class_names):
    """Classify image using text descriptions of classes."""
    # Create text embeddings for each class
    text_descriptions = [f"a photo of a {name}" for name in class_names]
    text_tokens = tokenizer(text_descriptions, padding=True, return_tensors="pt")
    
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
    
    # Get image embedding
    image_tokens = preprocess_image(image).unsqueeze(0)
    image_embed = model.encode_image(image_tokens)
    
    # Compute similarity
    similarities = image_embed @ text_embeds.T
    predicted_idx = similarities.argmax().item()
    
    return class_names[predicted_idx]
```

## SigLIP: Improved CLIP Training

SigLIP replaces the softmax cross-entropy with a sigmoid loss, improving training stability and performance:

```python
def siglip_loss(image_embeds, text_embeds, temperature=0.07):
    """Sigmoid loss instead of softmax cross-entropy."""
    logits = (image_embeds @ text_embeds.T) / temperature
    
    # Sigmoid loss: each image-text pair is independent
    # No need for batch-level normalization
    labels = torch.ones(logits.shape).to(logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    
    return loss
```

**SigLIP advantages**:
- No need for in-batch negatives (contrastive pairs are implicit).
- Better scaling with batch size.
- More stable training.

## BLIP and BLIP-2: Bootstrapped Language-Image Pre-training

BLIP introduces a multimodal encoder-decoder architecture for better captioning:

```python
class BLIP(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.vision_encoder = VisionTransformer(vision_config)
        self.text_encoder = Transformer(text_config)
        self.vision_projection = nn.Linear(vision_config.hidden_size, text_config.hidden_size)
    
    def forward_image_text(self, image, text):
        """Image-text matching."""
        vision_embeds = self.vision_encoder(image)
        vision_embeds = self.vision_projection(vision_embeds)
        
        # Cross-attention between image and text
        text_embeds = self.text_encoder(text, encoder_hidden_states=vision_embeds)
        return text_embeds
    
    def forward_image_captioning(self, image, decoder_input_ids):
        """Generate captions."""
        vision_embeds = self.vision_encoder(image)
        vision_embeds = self.vision_projection(vision_embeds)
        
        outputs = self.text_decoder(
            decoder_input_ids,
            encoder_hidden_states=vision_embeds
        )
        return outputs.logits
```

## Vision-Language Models with LLMs

BLIP-2 and later models connect frozen vision encoders to frozen LLMs:

```python
class BLIP2QFormer(nn.Module):
    """Q-Former: Transformer that maps image to LLM-compatible tokens."""
    def __init__(self, num_query_tokens=32, hidden_size=768):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        self.transformer = Transformer(num_layers=6)
        self.projection = nn.Linear(hidden_size, LLM.hidden_size)
    
    def forward(self, image_embeds):
        # Query tokens interact with image via cross-attention
        batch_size = image_embeds.shape[0]
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        # Cross-attention
        for layer in self.transformer.layers:
            queries = layer(queries, encoder_hidden_states=image_embeds)
        
        # Project to LLM embedding space
        query_output = self.projection(queries)
        return query_output  # (batch, num_queries, LLM_dim)
```

## Building with Multimodal Embeddings

### Image Search with CLIP

```python
import torch
from PIL import Image

def build_image_index(image_paths, model, batch_size=32):
    """Build an index of image embeddings."""
    embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        images = [load_image(p) for p in batch]
        batch_tensor = preprocess_images(images)
        
        with torch.no_grad():
            embeds = model.encode_image(batch_tensor)
            embeds = F.normalize(embeds, dim=-1)
        embeddings.append(embeds)
    
    return torch.cat(embeddings, dim=0)

def search_images(query, index, image_paths, model, top_k=10):
    """Search for images matching text query."""
    # Encode query
    text_tokens = tokenizer([query]).to(model.device)
    with torch.no_grad():
        query_embed = model.encode_text(text_tokens)
        query_embed = F.normalize(query_embed, dim=-1)
    
    # Search
    similarities = query_embed @ index.T
    top_indices = similarities.topk(top_k).indices[0]
    
    return [image_paths[i] for i in top_indices]
```

### Image Ranking and Filtering

```python
def rank_images_by_description(model, image_paths, description):
    """Rank images by relevance to a description."""
    # Get text embedding
    text_embed = model.encode_text(tokenizer([description]))
    text_embed = F.normalize(text_embed, dim=-1)
    
    scores = []
    for path in image_paths:
        image = preprocess_image(load_image(path)).unsqueeze(0)
        image_embed = model.encode_image(image)
        image_embed = F.normalize(image_embed, dim=-1)
        
        score = (text_embed @ image_embed.T).item()
        scores.append((path, score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

## Evaluating Multimodal Embeddings

### Retrieval Metrics

```python
def evaluate_retrieval(image_embeds, text_embeds, ground_truth):
    """Evaluate image-text retrieval."""
    # Compute all pairwise similarities
    similarities = image_embeds @ text_embeds.T
    
    # Image-to-text retrieval
    i2t_metrics = {
        "recall@1": recall_at_k(similarities, ground_truth, k=1),
        "recall@5": recall_at_k(similarities, ground_truth, k=5),
        "recall@10": recall_at_k(similarities, ground_truth, k=10),
        "r_precision": r_precision(similarities, ground_truth),
    }
    
    return i2t_metrics
```

### Zero-Shot Classification Accuracy

```python
def evaluate_zero_shot(model, dataset, class_names):
    """Evaluate zero-shot classification on a dataset."""
    correct = 0
    total = 0
    
    for image, label in dataset:
        predicted = zero_shot_classify(model, image, class_names)
        if predicted == class_names[label]:
            correct += 1
        total += 1
    
    return correct / total
```

## Popular Multimodal Models

| Model | Embedding Dim | Context | Best For |
|-------|--------------|---------|----------|
| CLIP ViT-L/14 | 768 | 77 tokens | General purpose |
| SigLIP ViT-SO400M | 1152 | 64 tokens | Better accuracy |
| BLIP-2 | 768 | 32 queries | Captioning + VQA |
| LLaVA | 4096 | 336×336px | Conversation |
| GPT-4V (API) | API | Variable | Best quality |

## Practical Applications

### Product Search

```python
def product_search(query, product_images, product_data, model):
    """Search product catalog by image or text."""
    # Get query embedding
    if is_image(query):  # Image-to-image search
        query_embed = get_image_embedding(query)
    else:  # Text-to-image search
        query_embed = get_text_embedding(query)
    
    # Search product embeddings
    scores = query_embed @ product_embeddings.T
    top_products = get_top_k(scores, k=20)
    
    return [product_data[i] for i in top_products]
```

### Visual Question Answering

```python
def visual_question_answering(image, question, model):
    """Answer questions about an image."""
    # Prepare input
    prompt = f"Question: {question} Answer:"
    inputs = model.prepare_inputs(image, prompt)
    
    # Generate answer
    outputs = model.generate(**inputs, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer
```

Multimodal embeddings are transforming how we search and understand visual content. Models like CLIP, SigLIP, and BLIP provide the foundation for applications ranging from product search to medical imaging analysis.