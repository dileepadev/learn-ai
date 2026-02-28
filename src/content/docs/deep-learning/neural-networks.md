---
title: Introduction to Neural Networks
description: Understanding the fundamental building blocks of Deep Learning.
---

Artificial Neural Networks (ANNs), often simply called Neural Networks, are a subset of machine learning and are at the heart of deep learning algorithms. their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.

## What is a Neural Network?

A neural network is a computational model consisting of layers of interconnected nodes, or "neurons," which process information. They are designed to recognize patterns and solve complex problems in areas such as image recognition, natural language processing, and speech recognition.

Unlike traditional algorithms that are programmed with specific rules, neural networks learn to perform tasks by analyzing examples.

## Structure of a Neural Network

A simple neural network includes three types of layers:

1. **Input Layer:** The first layer that receives the raw data (e.g., pixels of an image, words in a sentence). It passes this information to the next layer.
2. **Hidden Layers:** Layers between the input and output layers. Ideally, there can be one or many (deep) hidden layers. This is where the computation happens and features are extracted. The "deep" in deep learning refers to having multiple hidden layers.
3. **Output Layer:** The final layer that produces the result or prediction (e.g., probability of an image being a cat).

Each connection between neurons has a **weight**, which determines the strength of the signal. Each neuron also has a **bias**.

## How Neural Networks Work

The process of training a neural network involves two main passes:

### 1. Forward Propagation

Data flows through the network from the input layer to the output layer. Is calculates a weighted sum of its inputs, adds a bias, and passes the result through an **activation function**.

$$ Output = Activation(\sum (Weight \times Input) + Bias) $$

The activation function helps the network learn complex patterns by introducing non-linearity. Common activation functions include:

- **Sigmoid**: Maps output to a range between 0 and 1.
- **ReLU (Rectified Linear Unit)**: Outputs input directly if positive, otherwise, it will output zero.

### 2. Backpropagation

Once the output is generated, the network compares it to the actual target value using a **Loss Function** (or Cost Function). The loss function measures the error or difference between the predicted and actual values.

The network then propagates this error backward through the layers (backpropagation) to update the weights and biases using an optimization algorithm like **Gradient Descent**. This step minimizes the error for future predictions.

## Types of Neural Networks

- **Feedforward Neural Networks (FNN):** The simplest type where connections between nodes do not form a cycle.
- **Convolutional Neural Networks (CNN):** Primarily used for image processing and computer vision tasks.
- **Recurrent Neural Networks (RNN):** Designed for sequential data like time series or natural language.

Neural networks have revolutionized the field of AI, enabling breakthroughs in autonomous driving, machine translation, and generative AI.
