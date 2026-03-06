# CS760 – Machine Learning (UW–Madison)

This repository contains materials, experiments, and implementations developed during **CS 760: Machine Learning** at the University of Wisconsin–Madison.

The course focuses on **computational approaches to learning**, exploring how machines can automatically extract patterns from data and make predictions or decisions. Topics span **supervised learning, unsupervised learning, reinforcement learning, statistical learning theory, and neural networks**, as well as theoretical foundations of machine learning.

Throughout the course, I implemented and experimented with algorithms, evaluated model performance, and studied theoretical guarantees behind modern machine learning systems.

---

# Core Learning Objectives

The course covers the fundamental question:

> *What does it mean for a machine to learn from data?*

Key topics include:

- learning from labeled and unlabeled data
- designing predictive models
- evaluating model generalization
- understanding tradeoffs between bias and variance
- optimization techniques used in training models
- probabilistic and statistical foundations of learning

---

# Supervised Learning

Supervised learning focuses on learning a function that maps inputs `x` to outputs `y` using labeled training data.

Algorithms studied and implemented include:

## Linear Models

- Linear Regression
- Logistic Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Softmax regression for multi-class classification

These models are widely used for tasks such as:

- predicting housing or rental prices
- medical risk prediction
- classification tasks

Example application:

Predicting **real estate rent prices** using property features such as location, square footage, occupancy rate, and nearby market trends.

---

## Instance-Based Learning

- k-Nearest Neighbors (kNN)

This method predicts outcomes based on the closest training examples in feature space.

Example use cases:

- predicting housing prices based on nearby properties
- classification of handwritten digits
- similarity-based recommendations

---

## Tree-Based Models

- Decision Trees
- Random Forests

Random Forests combine multiple decision trees to improve prediction accuracy and reduce overfitting.

Example applications:

- predicting rent or property value
- medical diagnosis prediction
- fraud detection

---

## Margin-Based Methods

- Support Vector Machines (SVM)
- Kernel Methods

Kernel functions allow SVMs to model nonlinear relationships in data.

Example applications:

- image classification
- document classification
- medical classification problems

---

## Neural Networks

Topics studied include:

- Perceptrons
- Multilayer Neural Networks
- Backpropagation
- Gradient Descent optimization

Neural networks are capable of learning complex nonlinear relationships.

Applications studied include:

- pattern recognition
- regression tasks
- classification problems

---

# Deep Learning Topics

The course introduced key architectures used in modern deep learning.

## Convolutional Neural Networks (CNNs)

CNNs are commonly used for image processing tasks.

Applications:

- object recognition
- medical imaging analysis
- autonomous driving perception

---

## Recurrent Neural Networks (RNNs)

RNNs process sequential data such as time series or language.

Topics covered:

- encoder–decoder architectures
- Long Short-Term Memory networks (LSTM)

Applications:

- language translation
- text generation
- speech recognition

---

# Natural Language Processing

Language modeling techniques were explored for understanding text data.

Topics include:

- language models
- sequence prediction
- neural network approaches to NLP

Example applications:

- text classification
- sentiment analysis
- machine translation

---

# Unsupervised Learning

Unsupervised learning focuses on discovering patterns in unlabeled data.

Algorithms studied include:

## Clustering

- K-Means Clustering
- Hierarchical Clustering
- Gaussian Mixture Models (GMM)

Clustering algorithms group similar observations together.

Example applications:

- customer segmentation
- grouping medical patient profiles
- identifying housing market clusters

---

## Dimensionality Reduction

- Principal Component Analysis (PCA)

PCA reduces the number of features while preserving the most important variance in the data.

Applications include:

- visualization of high-dimensional datasets
- noise reduction
- improving model performance

---

# Reinforcement Learning

Reinforcement learning studies how agents learn to make decisions through interaction with an environment.

Topics covered include:

- Markov Decision Processes
- Bellman Equation
- Q-Learning
- policy optimization
- credit assignment problem
- exploration vs exploitation tradeoff

Example applications:

- robotics navigation
- autonomous systems
- game playing agents

---

# Semi-Supervised Learning

Semi-supervised learning combines labeled and unlabeled data to improve model performance when labeled data is limited.

Techniques studied include:

- fine-tuning pre-trained models
- leveraging unlabeled datasets
- improving generalization

Applications include:

- NLP models trained on large text corpora
- image classification with limited labels

---

# Theoretical Foundations of Machine Learning

A key focus of the course is understanding why machine learning algorithms work.

## Bias–Variance Tradeoff

Understanding how model complexity affects overfitting and generalization.

---

## PAC Learning (Probably Approximately Correct)

The PAC framework provides theoretical guarantees on how well a model learned from training data will perform on unseen data.

---

## VC Dimension

The Vapnik–Chervonenkis dimension measures the expressive capacity of a model class.

---

## Empirical Risk Minimization

ERM describes the principle of minimizing training error over the dataset.

---

## Maximum Likelihood Estimation

MLE is a statistical framework used to estimate model parameters.

---

# Optimization Methods

Training machine learning models requires solving optimization problems.

Methods studied include:

- gradient descent
- stochastic gradient descent
- loss functions
- backpropagation in neural networks

These techniques allow models to iteratively improve their predictions.

---

# Evaluation of Machine Learning Models

The course also emphasizes evaluating model performance.

Common techniques include:

- training vs test set validation
- cross validation
- accuracy, precision, recall
- model comparison and benchmarking

Understanding these metrics is essential for selecting the best learning algorithm.

---

# Skills Developed

Through this course I gained experience in:

- implementing machine learning algorithms
- evaluating model performance
- designing predictive systems
- understanding theoretical foundations of learning
- applying machine learning to real-world problems

---

# Technologies Used

- Python
- NumPy
- SciPy
- scikit-learn
- TensorFlow / PyTorch (for neural networks)
- Jupyter notebooks

---

# Summary

CS760 provides a rigorous introduction to modern machine learning, combining:

- algorithm design
- statistical theory
- real-world applications

The course builds a strong foundation for advanced work in **artificial intelligence, data science, and applied machine learning systems**.
