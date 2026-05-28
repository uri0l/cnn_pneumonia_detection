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

<img width="1302" height="433" alt="gradcam_correct_predictions" src="https://github.com/user-attachments/assets/43199455-5ee7-4e06-aaf4-244224235405" />

---

## Overview

This project develops a Convolutional Neural Network (CNN) for detecting pneumonia from chest X-ray images, while also providing visual explanations of its predictions through Grad-CAM heatmaps. Although deep learning models have achieved strong performance in medical imaging tasks, they are often treated as black boxes, which limits their adoption in clinical settings where interpretability and trust are essential.

The proposed approach is based on the methodology presented in **[*Pneumonia Detection Using CNN-based Feature Extraction*](https://ieeexplore.ieee.org/document/8869364)**, which demonstrates the effectiveness of convolutional architectures for automated pneumonia classification. This work further extends the focus to model interpretability, aiming to improve transparency and support clinical decision-making.

Using the publicly available **[Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**, we implemented a full deep learning pipeline in Python that includes image preprocessing, exploratory data analysis, CNN training and evaluation, and explainability via Gradient-weighted Class Activation Mapping (Grad-CAM). The resulting heatmaps allow clinicians to visually inspect which lung regions most influenced the model’s decision, helping bridge the gap between artificial intelligence and clinical interpretability.




## Goals
### General Goal
Develop and evaluate an explainable deep learning classifier for binary pneumonia detection from chest X-rays, using transfer learning and Grad-CAM to produce clinically interpretable predictions.
 
### Specific Goals
- Reproduce and adapt a ResNet50 transfer learning pipeline on the Kaggle chest X-ray pneumonia dataset.
- Design a data augmentation strategy that removes non-clinical artifacts (corner stamps, R/L markers) while preserving lung structure.
- Identify the image regions driving each prediction using Grad-CAM heatmaps attached to the last convolutional block.
- Compare model performance (AUC, balanced accuracy) against a baseline CNN trained from scratch.
- Analyze operating point trade-offs (sensitivity vs. specificity) using Youden's J statistic.
---

## Dataset
 
- **Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), originally from Kermany et al., *Cell* (2018).
- **Samples:** ~5,856 paediatric chest X-rays from Guangzhou Women and Children's Medical Centre.
- **Labels:** `NORMAL = 0`, `PNEUMONIA = 1`.
- **Class imbalance:** ~74% pneumonia in training - realistic biomedical scenario.
- **Re-split strategy:** the original Kaggle train/val split is merged and re-split 80/20 stratified, because the original validation set has only 16 images.
```
chest_xray/
├── train/
│   ├── NORMAL/          (~1,341 images)
│   └── PNEUMONIA/       (~3,875 images)
├── val/
│   ├── NORMAL/          (8 images - merged into train pool)
│   └── PNEUMONIA/       (8 images - merged into train pool)
└── test/
    ├── NORMAL/          (~234 images)
    └── PNEUMONIA/       (~390 images)
```
 
> The dataset is not included in this repository. Download instructions are provided in the notebook.
 
---
 
## Methods & Pipeline
 
### 1. Data Import & Preprocessing
- Images loaded as grayscale, resized to 224×224, stored as `float32` in [0, 1].
- Original Kaggle train/val pool merged and re-split 80/20 stratified by label.
- Test set kept completely held-out throughout all development.
### 2. Data Augmentation (training only)
- **Peripheral mask** (7% border zeroed) - removes corner stamps and hospital text before any transform.
- **Random resized crop** (scale 0.85–1.0, ratio 0.9–1.1) - zoom augmentation pushing artifacts out of frame.
- **Random horizontal flip + rotation** (±10°).
- **Grayscale → 3-channel replication** + ImageNet normalization (μ/σ per channel).
### 3. Model Architecture - ResNet50 + Transfer Learning
 
| Phase | Layers trained | LR | Epochs | Purpose |
|---|---|---|---|---|
| Phase 1 | Head only | 1e-3 | 10 | Warm up classifier; backbone frozen |
| Phase 2 | Entire network | 1e-4 | 15 | End-to-end fine-tuning |
 
Custom head:
 
```
2048 (GAP output) → Dense(256, ReLU) → Dropout(0.4) → Dense(1, logit)
```
 
Training uses weighted BCE loss (to address class imbalance), Adam optimizer, CosineAnnealingLR, and early stopping on validation AUC (patience = 5).
 
### 4. Evaluation
- Confusion matrix, accuracy, balanced accuracy, sensitivity, specificity on the held-out test set.
- AUROC curve with operating point analysis (Youden's J statistic for optimal threshold).
- Precision-recall curve.
### 5. Explainability - Grad-CAM
- Hooks registered on `model.layer4[-1]` (last residual block).
- Gradient-weighted class activation map upsampled to 224×224 and overlaid on the original X-ray.
- Qualitatively validates that the model attends to lung parenchyma rather than image artifacts.
### 6. Baseline Comparison
- Lightweight 3-block CNN (~200K params) trained from scratch on 64×64 resized images.
- Compared to ResNet50 on test AUC.
---
 
## Key Results
 
| Model | Test Accuracy | Balanced Accuracy | AUC-ROC |
|---|---|---|---|
| Baseline CNN (scratch) | - | - | 0.9293 |
| ResNet50 (transfer learning) | 0.9375 | 0.9397 | 0.9808 |
 
> *Results shown are from one representative training run. Due to random initialization, data shuffling, and augmentation, repeated runs may produce slightly different metrics, though performance should remain in a similar range.*
 
---
 
## Repository Structure
 
> All figures generated by the pipeline are saved to the `plots/` folder. If you encounter issues running locally, you can browse pre-generated outputs there.
 
```
.
├── MODEL3__FINAL___new_split_new_transform_with_EDA.ipynb   # Full annotated pipeline
├── app.py                                                    # Streamlit inference app
├── checkpoints/
│   └── pneumonia_resnet50.pth                               # Saved model weights (phase 2)
├── plots/
│   ├── baseline_vs_resnet50_roc.png
│   ├── confusion_matrix_and_roc.png
│   ├── eda_brightness_scatter.png
│   ├── eda_class_distribution.png
│   ├── eda_intensity_distributions.png
│   ├── eda_mean_images.png
│   ├── eda_sample_images.png
│   ├── eda_variance_maps.png
│   ├── gradcam_grad-cam__correct_normal_detections_tn.png
│   ├── gradcam_grad-cam__correct_pneumonia_detections_tp.png
│   ├── gradcam_grad-cam__correct_predictions_top_x-ray,_bottom_heatmap.png
│   ├── gradcam_grad-cam__false_negatives_pneumonia_missed__all_cases.png
│   ├── gradcam_grad-cam__false_positives_normal_predicted_as_pneumonia.png
│   ├── gradcam_grad-cam__incorrect_predictions.png
│   ├── operating_point_analysis.png
│   ├── training_phase_1__head_training_backbone_frozen.png
│   ├── training_phase_2__full_fine-tuning.png
│   ├── tsne_projection.png
│   └── umap_projection.png
└── README.md
```
 
---
 
## Dependencies
 
This project was developed in Python 3.10 on Google Colab with a Tesla T4 GPU.
 
```bash
pip install torch torchvision
pip install numpy matplotlib pillow scikit-learn scipy
pip install streamlit          # for the inference app
```
 
Training is ~10× faster on GPU. Enable via **Runtime → Change runtime type → T4 GPU** in Colab.
 
---
 
## How to Run
 
### Notebook (training + evaluation)
 
1. Open `MODEL3__FINAL___new_split_new_transform_with_EDA.ipynb` in Google Colab.
2. Enable GPU: **Runtime → Change runtime type → T4 GPU → Save**.
3. Download the dataset following Option A or Option B in Section 2.1 of the notebook.
4. Run all cells in order. The model checkpoint is saved to `checkpoints/pneumonia_resnet50.pth`.


### Streamlit App
 
```bash
streamlit run app.py
```
 
Upload any chest X-ray image to get a classification prediction, confidence score, and Grad-CAM heatmap overlay.

---
 
## Reference
 
Kermany, D.S. et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell **172**, 1122–1131 (2018). https://doi.org/10.1016/j.cell.2018.02.010
 



