---
title: "Data Augmentation in Deep Learning"
description: "Comprehensive guide to data augmentation strategies — basic techniques, AutoAugment, and domain-specific augmentations."
date: "2026-06-06"
tags: ["deep-learning", "data-augmentation", "training"]
---

Data augmentation artificially expands the training dataset by applying transformations. This improves generalization, acts as a regularizer, and is especially valuable when data is limited.

## Image Augmentations

### Basic Geometric Transformations

```python
import torchvision.transforms as T

basic_augmentations = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.RandomResize((224, 224)),
])

# Random crop with resize
crop_transform = T.Compose([
    T.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
])
```

### Color Jittering

```python
color_jitter = T.Compose([
    T.RandomHorizontalFlip(),
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
])
```

### Cutout and Random Erasing

```python
class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        c, h, w = img.shape
        area = h * w
        
        for _ in range(10):
            erase_area = random.uniform(*self.scale) * area
            aspect = random.uniform(*self.ratio)
            
            eh = int(math.sqrt(erase_area * aspect))
            ew = int(math.sqrt(erase_area / aspect))
            
            if eh < h and ew < w:
                i = random.randint(0, h - eh)
                j = random.randint(0, w - ew)
                img[:, i:i+eh, j:j+ew] = 0
                return img
        
        return img


# PyTorch built-in
from torchvision.transforms import RandomErasing
erasing = RandomErasing(p=0.5)
```

### Mixup

Mix two images and their labels:

$$x_{mixed} = \lambda x_i + (1-\lambda)x_j$$
$$y_{mixed} = \lambda y_i + (1-\lambda)y_j$$

```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

### CutMix

Cut a region from one image and paste it onto another:

```python
def cutmix_data(x, y, beta=1.0):
    lam = np.random.beta(beta, beta)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    # Generate bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # Mix images
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda for loss
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y, y[index], lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2
```

## AutoAugment

Learn augmentation policies from data:

```python
# AutoAugment policies are learned search spaces
# PyTorch provides RandAugment and TrivialAugment

from torchvision.transforms.autoaugment import AutoAugment, RandAugment

# AutoAugment (learned policy)
auto_augment = AutoAugment()

# RandAugment (simpler, faster)
rand_augment = RandAugment(num_ops=9, magnitude=9)
```

## Text Augmentations

```python
# Back-translation
def back_translate(text, src_model, tgt_model):
    # Translate to another language
    translated = translate(text, src_model, tgt_model)
    # Translate back
    back_translated = translate(translated, tgt_model, src_model)
    return back_translated

# Synonym replacement
def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        word = random.choice(words)
        synonyms = get_synonyms(word)
        if synonyms:
            new_words = [random.choice(synonyms) if w == word else w 
                        for w in new_words]
    return ' '.join(new_words)
```

## Audio Augmentations

```python
import librosa

def augment_audio(waveform, sample_rate):
    # Time stretching
    rate = random.uniform(0.9, 1.1)
    waveform = librosa.effects.time_stretch(waveform, rate=rate)
    
    # Pitch shifting
    n_steps = random.uniform(-2, 2)
    waveform = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=n_steps)
    
    # Add noise
    noise = np.random.randn(len(waveform)) * 0.005
    waveform = waveform + noise
    
    return waveform
```

## Practical Recommendations

| Task | Augmentations |
| --- | --- |
| Image classification | Flip, crop, color jitter, cutout, mixup |
| Object detection | Flips, scale, aspect ratio changes |
| Semantic segmentation | Same geometric transforms for image and mask |
| NLP | Back-translation, synonym replacement |
| Audio | Time stretch, pitch shift, noise injection |

Start with basic augmentations, then add more sophisticated ones if underfitting.