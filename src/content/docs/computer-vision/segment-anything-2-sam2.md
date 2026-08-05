---
title: "Segment Anything 2 (SAM 2): Promptable Visual Segmentation for Video & Images"
description: Deep dive into Meta's SAM 2 architecture, streaming memory attention mechanism, promptable video segmentation, and real-time inference techniques.
---

Meta’s **Segment Anything Model 2 (SAM 2)** expands the foundational image segmentation capabilities of the original SAM into the continuous spatial-temporal domain of video. Designed as a unified model for promptable segmentation in both images and videos, SAM 2 treats images simply as single-frame videos.

## Background: From SAM to SAM 2

The original SAM revolutionized computer vision by enabling zero-shot image segmentation driven by user prompts (points, bounding boxes, or rough masks). However, extending image segmentation to video presents severe challenges:
1. **Temporal Continuity:** Objects deform, change scale, undergo occlusion, and re-appear across frames.
2. **Computational Overhead:** Processing video frame-by-frame independently loses temporal context and is computationally prohibitive.
3. **Memory Accumulation:** Tracking objects across thousands of frames requires managing memory without GPU memory collapse.

SAM 2 addresses these challenges with a streaming memory architecture that achieves real-time interactive performance.

## SAM 2 Architectural Breakdown

SAM 2 introduces a frame-by-frame streaming pipeline consisting of five core modules:

```
+------------------+     +-------------------+     +------------------+
| Frame Image      | --> | Image Encoder     | --> | Memory Attention | --> Mask Decoder --> Output Mask
+------------------+     +-------------------+     +------------------+
                                                            ^
                                                            | (Past Frame Features & Prompts)
                                                   +------------------+
                                                   | Memory Bank      |
                                                   +------------------+
```

### 1. Image Encoder
A hierarchical vision transformer (Hiera) extracts multi-scale feature embeddings for each video frame. Hiera provides higher throughput and better spatial feature maps than standard ViT architectures.

### 2. Memory Attention Module
The memory attention module conditions current frame features on past frame memories and user prompts. It uses cross-attention layers to query:
- **Spatial Memory:** Features of recent object masks in prior frames.
- **Prompt Memory:** Features from frames where explicit user interactions (clicks/boxes) occurred.

### 3. Mask Decoder & Occlusion Head
The mask decoder outputs predicted segmentation masks for the target frame. Crucially, SAM 2 incorporates an **Occlusion Head** that outputs an "occlusion score" indicating whether the target object is currently visible or hidden behind another object.

### 4. Memory Encoder & Memory Bank
- **Memory Encoder:** Downsamples frame predictions and spatial features into compact memory vectors.
- **Memory Bank:** Maintains a FIFO queue of recent frame memories alongside a fixed pool of keyframe memories (frames with explicit user prompts).

## Interactive Video Prompting Workflow

SAM 2 supports continuous interactive refinement throughout video playback:

1. **Initial Prompt:** User clicks an object on frame 0. SAM 2 generates a mask for frame 0 and propagates tracking forward across subsequent frames.
2. **Correction:** If the model drifts on frame 45 due to severe rotation, the user clicks the object on frame 45 to correct it.
3. **Bidirectional Propagation:** SAM 2 instantly updates memory vectors and updates segmentation both **forward** (frame 46+) and **backward** (frames 1–44).

## Python Code Example: Using SAM 2

Below is an implementation snippet using the official `sam2` library to segment objects in a video stream:

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

# Load SAM 2 checkpoint
sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# Initialize inference state on video directory
inference_state = predictor.init_state(video_path="video_frames/")

# Add prompt on frame 0 (point prompt: [x, y], label: 1 for foreground)
ann_frame_idx = 0
ann_obj_id = 1
points = torch.tensor([[500, 300]], dtype=torch.float32)
labels = torch.tensor([1], dtype=torch.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Propagate segmentation throughout video
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
```

## Key Applications

- **Video Editing & VFX:** Rotoscoping objects, background replacement, and style transfer in real-time.
- **Autonomous Systems:** Segmenting moving pedestrians, vehicles, and obstacles across continuous video feeds.
- **Medical Imaging:** Tracking organ boundaries or tumor movement across dynamic ultrasound and MRI sequences.
- **Robotics & Manipulation:** Visual tracking for robotic arms grasping moving objects.

## SAM vs. SAM 2 Feature Comparison

| Feature | SAM (2023) | SAM 2 (2024) |
|---|---|---|
| **Domain** | Static Images | Images & Continuous Video |
| **Backbone** | ViT-H / ViT-L | Hiera (Hierarchical ViT) |
| **Interactive Latency** | ~50ms / image | ~44 FPS video tracking |
| **Occlusion Handling** | None | Dedicated Occlusion Head |
| **Memory Tracking** | N/A | Streaming Memory Bank & Attention |

## Summary

SAM 2 marks a major step toward real-time spatial-temporal visual intelligence. By pairing hierarchical vision backbones with a streaming memory bank, it achieves state-of-the-art zero-shot video object segmentation while maintaining interactive multi-point feedback capabilities.

## Further Reading

- Ravi et al. (2024), *SAM 2: Segment Anything in Images and Videos* (Meta AI)
- Kirillov et al. (2023), *Segment Anything* (Meta AI)
- Meta AI Open Source SAM 2 Repository: `github.com/facebookresearch/segment-anything-2`
