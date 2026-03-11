---
title: Getting Started with Scikit-Learn
description: A beginner's guide to the most popular library for traditional machine learning.
---

Scikit-learn is the go-to Python library for traditional Machine Learning algorithms. It is built on top of NumPy, SciPy, and Matplotlib.

## Why Scikit-Learn?

- **Simple and efficient:** Consistent API across different algorithms.
- **Comprehensive:** Tools for data preprocessing, modeling, and evaluation.
- **Open-source:** Large community and extensive documentation.

## The Scikit-Learn Workflow

Most tasks in Scikit-learn follow a similar pattern:

1. **Load Data:** Import your dataset (often using Pandas).
2. **Preprocessing:** Clean and prepare data (e.g., scaling, encoding categorical variables).
3. **Split Data:** Divide data into training and testing sets.

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

4. **Choose a Model:** Select an algorithm (e.g., Random Forest, SVM).
5. **Train (Fit):** Teach the model using the training data.

   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   ```

6. **Predict:** Use the trained model on new data.

   ```python
   predictions = model.predict(X_test)
   ```

7. **Evaluate:** Measure performance using metrics.

Scikit-learn is ideal for data exploration and building baseline models before moving to more complex deep learning frameworks like PyTorch or TensorFlow.
