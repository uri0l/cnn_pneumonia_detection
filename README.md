# Explainable CNN for Pneumonia Detection from Chest X-Rays
> A machine learning project developed as part of the Machine Learning course at the Universitat Politècnica de Barcelona (UPC), Bachelor of Science (BSc) in Bioinformatics, May 2026.

**Authors:** 
* [Òscar Contreras](https://github.com/ocontrerasparejo)
* [Eira Fontanals](https://github.com/eirafontanals)
* [Oriol Leal](https://github.com/uri0l)
* Marc J. Torres

[![Bioinformatics](https://img.shields.io/badge/Field-Bioinformatics-green.svg)](https://en.wikipedia.org/wiki/Bioinformatics)
[![Machine Learning](https://img.shields.io/badge/Methods-PLS--DA%20%7C%20PCA-orange.svg)](https://en.wikipedia.org/wiki/Partial_least_squares_regression)
[![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![ResNet50](https://img.shields.io/badge/Model-ResNet50-009688?style=flat&logoColor=white)](https://arxiv.org/abs/1512.03385)

---

## Overview

This project develops a Convolutional Neural Network (CNN) for detecting pneumonia from chest X-ray images, while also providing visual explanations of its predictions through Grad-CAM heatmaps. Although deep learning models have achieved strong performance in medical imaging tasks, they are often treated as black boxes, which limits their adoption in clinical settings where interpretability and trust are essential.

The proposed approach is based on the methodology presented in **[*Pneumonia Detection Using CNN-based Feature Extraction*](https://ieeexplore.ieee.org/document/8869364)**, which demonstrates the effectiveness of convolutional architectures for automated pneumonia classification. This work further extends the focus to model interpretability, aiming to improve transparency and support clinical decision-making.

Using the publicly available **[Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**, we implemented a full deep learning pipeline in Python that includes image preprocessing, exploratory data analysis, CNN training and evaluation, and explainability via Gradient-weighted Class Activation Mapping (Grad-CAM). The resulting heatmaps allow clinicians to visually inspect which lung regions most influenced the model’s decision, helping bridge the gap between artificial intelligence and clinical interpretability.

## Goals
### General Goal

Develop an interpretable deep learning model capable of detecting pneumonia from chest X-ray images while providing transparent visual explanations of its predictions.

### Specific Goals
- Implement a robust data augmentation and preprocessing pipeline to standardize input dimensions and neutralize intensity/contrast confounders.
- Fine-tune a pre-trained ResNet50 convolutional neural network architecture using a controlled ...
- ...
