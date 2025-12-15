# Comparative_Steel_Inspection_using_Densnet_and_k_cross_validation
This repository contains a PyTorch implementation for classifying steel surface defects using the DenseNet121 convolutional neural network. The project demonstrates a comparative approach by integrating k-fold cross-validation to evaluate model performance more robustly across the dataset.
Comparative Steel Surface Defect Inspection using DenseNet and K-Fold Cross-Validation

This repository contains a PyTorch implementation for classifying steel surface defects using the DenseNet121 convolutional neural network. The project demonstrates a comparative approach by integrating k-fold cross-validation to evaluate model performance more robustly across the dataset.

Features

DenseNet121 fine-tuned on a steel defect dataset with 6 defect classes: crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.

Stratified K-Fold Cross-Validation to ensure fair evaluation and reduce bias from dataset splits.

Comprehensive evaluation metrics, including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

Visualization tools for:

Training and validation loss curves

Confusion matrices

Well-classified and misclassified images

Efficiency metrics such as model parameters, size, and inference time.

Dataset

The dataset consists of steel surface images (provided by the user) divided into train and test sets. All images are resized to 224×224 and normalized before training.

Purpose

This repository is intended to:

Serve as a reference for steel surface defect inspection using deep learning.

Provide a template for implementing k-fold cross-validation with PyTorch.

Facilitate comparison with other models (e.g., MobileNet, NASNet) for research or industrial applications.

Usage

Clone the repository.

Prepare your dataset in the specified train/test folder structure.

Install dependencies: torch, torchvision, numpy, scikit-learn, matplotlib, seaborn.

Run the notebook to train DenseNet121 with or without k-fold cross-validation.
