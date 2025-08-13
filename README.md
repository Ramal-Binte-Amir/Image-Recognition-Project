# Image-Recognition-Project
🐾 Cats vs. Dogs Classifier — Built with TensorFlow/Keras using both a custom CNN and MobileNetV2, achieving high accuracy through data augmentation, model tuning, and performance evaluation.
# Project Overview
-The goal was to train and compare two models:
-Custom CNN — Built from scratch with Conv2D, Pooling, and Dense layers.
-MobileNetV2 — A pre-trained model fine-tuned for our dataset.
-We evaluated performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix, along with visualizations for training progress.

# Dataset
Source: Kaggle — Cats and Dogs Dataset
Size: ~2,000 training images (balanced classes)
Preprocessing:
Resized to 224x224 pixels
Normalized pixel values to [0, 1]
Applied data augmentation (rotation, flipping, zoom)

# Model Details
# 🔹 Custom CNN
-Layers: Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output
-Optimizer: Adam (lr=0.001)
Epochs: 20

# 🔹 MobileNetV2
Pre-trained on ImageNet
Fine-tuned final layers for binary classification
Optimizer: Adam (lr=0.0001)
Epochs: 10

# Results
Model	Accuracy	Precision	Recall	F1-score
Custom CNN	85%	0.85	0.84	0.84
MobileNetV2	90%	0.91	0.90	0.90

# Key Learnings
Difference between building CNNs from scratch vs. using transfer learning
Importance of preprocessing & augmentation
Model evaluation beyond accuracy
