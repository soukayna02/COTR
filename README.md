# Pulmonary Disease Classification  
## Machine Learning Project

This project focuses on the automatic classification of lung X-ray images using classical Machine Learning techniques.  
It aims to support medical diagnosis by identifying:

- Normal
- COVID-19
- Viral Pneumonia

---

## Project Overview

This project uses hand-crafted features extracted from medical images and applies dimensionality reduction and classification models to achieve accurate predictions.

It was developed as an academic project at INSA Rennes.

---

## Dataset

The dataset is divided into three parts:

| Set        | Images |
|------------|---------|
| Training   | 125     |
| Testing    | 66      |
| Validation | 30      |

### Class Distribution

- Normal: 25 images  
- COVID-19: 50 images  
- Pneumonia: 50 images  

The dataset is balanced.

---

## Data Preprocessing

All images are processed using the following steps:

- Resize to 224 × 224  
- Convert to grayscale  
- Normalization  
- Missing value removal  
- Feature scaling  

---

## Feature Extraction (68 Features)

Each image is represented using 68 hand-crafted features:

| Feature Type        | Dimensions |
|---------------------|------------|
| Luminance Histogram | 32         |
| Texture (LBP)       | 26         |
| Hu Moments          | 7          |
| Density Features    | 3          |

---

## Dimensionality Reduction

### PCA (Principal Component Analysis)
- Preserves 95% of variance  
- Reduces dimensionality  

### LDA (Linear Discriminant Analysis)
- Improves class separation  
- Used before k-NN  

---

## Classification Models

### k-Nearest Neighbors (k-NN)
- k = 9  
- Manhattan distance  

### Support Vector Machine (SVM)
- Linear and RBF kernels  
- Class balancing  

### Ensemble Learning
- Combines SVM and k-NN  
- Improves final predictions  

---

## Results

| Metric            | Value   |
|-------------------|---------|
| Train Accuracy    | 87.92%  |
| Test Accuracy     | 84.85%  |
| SVM AUC           | 0.97    |
| COVID Sensitivity | 92.3%   |

The models show good generalization and low overfitting.

---

## Data Path Configuration (Notebook)

Before running the notebook, open `Projet_MLIP.ipynb` and edit the first cell:

```python
path_train = "/your/path/to/train"
path_test = "/your/path/to/test"
path_val = "/your/path/to/validation"

##Installation
Requirements
Python 3.8+


Install dependencies:
pip install numpy opencv-python scikit-image scikit-learn matplotlib seaborn scipy
