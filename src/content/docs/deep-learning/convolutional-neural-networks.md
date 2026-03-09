---
title: Convolutional Neural Networks (CNNs)
description: The power behind modern computer vision and image recognition.
---

Convolutional Neural Networks (CNNs or ConvNets) are a specialized type of deep neural network designed to process data with a grid-like topology, most notably images. They are the backbone of most computer vision applications today.

## Why CNNs for Images?

Regular neural networks don't scale well to images. A small 100x100 RGB image has 30,000 pixels. If each pixel is an input to a fully connected layer, the number of parameters would explode. More importantly, regular networks don't capture the spatial structure of images (the fact that nearby pixels are related).

CNNs solve this by using **convolutions** to extract local features.

## Core Components of a CNN

### 1. Convolutional Layer

This is the building block of a CNN. It uses "filters" (or kernels) that slide over the input image to detect features like edges, corners, or textures. The output is a **feature map**.

### 2. Pooling Layer

Pooling layers reduce the spatial size of the feature maps, which decreases the number of parameters and computation in the network. The most common type is **Max Pooling**, which takes the maximum value in a small window.

### 3. Activation Function (ReLU)

The Rectified Linear Unit (ReLU) is usually applied after each convolution operation. it introduces non-linearity, allowing the network to learn complex patterns.

### 4. Fully Connected (FC) Layer

After several convolutional and pooling layers, the high-level features are flattened and passed through one or more fully connected layers to produce the final output (e.g., classification into "cat" or "dog").

## Applications of CNNs

- **Image Classification:** Identifying the primary object in an image.
- **Object Detection:** Locating and identifying multiple objects within an image.
- **Image Segmentation:** Dividing an image into multiple segments (e.g., for medical imaging or self-driving cars).
- **Facial Recognition:** Identifying individuals based on facial features.

CNNs have fundamentally changed how machines "see" and interpret the visual world.
