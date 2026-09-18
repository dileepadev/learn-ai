---
title: Video Super-Resolution - Upscaling Video While Preserving Temporal Consistency
description: Learn how video super-resolution differs from single-image upscaling by exploiting information across frames while avoiding visible flicker.
---

Video super-resolution upscales low-resolution video into higher-resolution output, building on single-image super-resolution techniques but with an additional requirement that single-frame methods don't need to satisfy: temporal consistency across the resulting video.

## Why Video Isn't Just Many Independent Images

Applying an image super-resolution model independently to each frame of a video, frame by frame, can produce visible flickering — subtle differences in how the model hallucinates fine detail from frame to frame, even for a nearly static scene, become jarring temporal artifacts once played back as video, even though each individual frame might look perfectly sharp and plausible on its own.

## Exploiting Information Across Frames

Video super-resolution models address this by using information from multiple neighboring frames when reconstructing each output frame, both to improve detail recovery (a detail blurred or occluded in one frame might be visible in an adjacent frame due to slight camera or object motion) and to enforce consistency between consecutive output frames.

```text
frame_t-1, frame_t, frame_t+1 -> motion-compensated alignment -> fuse aligned frames -> upscaled frame_t
```

Motion estimation (optical flow) aligns neighboring frames to the target frame's viewpoint before fusing information across them, since simply averaging misaligned frames from a moving scene would blur rather than sharpen detail.

## Recurrent and Bidirectional Architectures

Rather than only looking at a small fixed window of neighboring frames, recurrent video super-resolution architectures propagate information across an entire video sequence using a hidden state, similarly to how a recurrent neural network processes a sequence — bidirectional variants propagate information both forward and backward through the sequence, letting a frame in the middle of a clip benefit from context on both sides rather than only preceding frames.

## Practical Guidance

Evaluate video super-resolution specifically on temporal consistency metrics (flicker measures across consecutive frames) in addition to per-frame sharpness metrics, since a model can score well on single-frame quality metrics while still producing visually distracting flicker when played as video. For content with fast motion or scene cuts, verify the model's motion-compensation approach degrades gracefully rather than producing severe artifacts when its motion estimation assumptions (smooth, continuous motion) are violated.
