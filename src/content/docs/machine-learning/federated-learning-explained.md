---
title: Introduction to Federated Learning
description: A privacy-preserving machine learning technique that trains models on decentralized data.
---

Federated Learning (FL) is a distributed machine learning approach that allows a central model to be trained without removing raw data from local devices.

## How Federated Learning Works

Instead of centralizing data in one location, FL trains the model locally on each device and only shares the weight updates (gradients) with a central server.

1. **Local Training:** Each device (e.g., smartphone or hospital server) trains a copy of the model on its own data.
2. **Aggregation:** The model updates are securely sent to a central server and averaged (e.g., using **FedAvg**).
3. **Global Update:** The central server sends the updated "global" model back to all devices for the next round.

## Benefits of FL

- **Privacy:** Data never leaves its source, making it ideal for healthcare and finance.
- **Latency:** Reduces the need for massive data transfers across networks.
- **Efficiency:** Leverages the compute power of millions of edge devices.

## Common Frameworks

- **Flower:** A friendly and extensible framework for federated learning.
- **TensorFlow Federated (TFF):** Google's open-source framework for FL.
- **PySyft:** A library for secure and private deep learning.
