---
title: "Feature Extraction with Pre-trained Models"
description: "Using pre-trained CNNs and transformers as feature extractors for transfer learning."
date: "2026-06-06"
tags: ["deep-learning", "transfer-learning", "computer-vision"]
---

Pre-trained models learned rich feature representations on large datasets. Using them as feature extractors is often the fastest way to get good performance on new tasks.

## Using Pre-trained CNNs as Feature Extractors

```python
import torchvision.models as models

# Load pre-trained ResNet
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Remove the final classification layer
features = nn.Sequential(*list(resnet.children())[:-1])

# Extract features
def extract_features(model, dataloader):
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            batch_features = model(inputs)
            features.append(batch_features)
            labels.append(targets)
    
    return torch.cat(features), torch.cat(labels)

# Features shape: (batch, 2048, 1, 1) for ResNet
# Reshape to (batch, 2048)
```

## Feature Extraction with Frozen Backbone

```python
class FeatureExtractor(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # New classifier
        self.classifier = nn.Sequential(
            nn.Linear(backbone.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


# Only classifier gradients are computed
model = FeatureExtractor(resnet, num_classes=10)
```

## Grad-CAM for Visualization

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, target_class=None):
        # Compute weighted combination of activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU to show only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()
```

## Extracting Features from Transformers

```python
from transformers import AutoModel, AutoTokenizer

# Load pre-trained transformer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = AutoModel.from_pretrained(model_name)

# Extract features
def get_bert_features(texts, max_length=128):
    inputs = tokenizer(
        texts, padding=True, truncation=True, 
        max_length=max_length, return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = bert(**inputs)
        # Use [CLS] token representation
        cls_features = outputs.last_hidden_state[:, 0, :]
        # Or mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        mean_features = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return cls_features, mean_features
```

## Feature Extraction from CLIP

```python
import clip

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Encode images
def encode_images(images):
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features

# Encode text
def encode_text(texts):
    with torch.no_grad():
        text_tokens = clip.tokenize(texts)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features
```

## Practical Tips

- Remove final classification layer for feature extraction
- Freeze early layers, fine-tune later layers for domain adaptation
- Use mean pooling for sentence-level representations from transformers
- Consider CLS token for classification-relevant features