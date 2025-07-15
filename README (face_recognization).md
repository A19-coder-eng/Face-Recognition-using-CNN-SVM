
# Face Recognition using CNN & SVM
This project compares two machine learning approaches — Support Vector Machine (SVM) with PCA and a Convolutional Neural Network (CNN) — for classifying grayscale face images. It demonstrates preprocessing, dimensionality reduction, classification, and evaluation of both models.


## Table of Contents

* Getting Started

* Dataset Overview

* Installation

* Project

* Modeling

* Evaluation Metrics

* Results

* Contact
##  Getting Started
**This project uses:**

* NumPy, pandas for data handling

* OpenCV for image processing

* scikit-learn for PCA & SVM

* TensorFlow/Keras for CNN modeling


## Dataset Overview
* Images stored in .npz format (62×47 grayscale)

* Labels provided in .csv format

* No missing or duplicate values

## Installation
**Install required libraries:**

`pip install numpy pandas matplotlib opencv-python scikit-learn tensorflow`
## Project
**1. Loading the data**

* .npz for image arrays

* .csv for labels and sample submission

**2. Preprocessing**

* Images are normalized (/255.0)

* Reshaped for flat input (SVM) or 4D input (CNN)

**3. Splitting**

* Training/Validation split (80/20)

**4. Dimensionality Reduction (SVM only)**

* PCA with 100 components
## Modeling
**SVM Models with PCA**

Trained using different kernels:

* **Linear:** F1 Score = 0.815

* **RBF:**F1 Score = 0.670

* **Polynomial:** F1 Score = 0.475

**Final SVM (Linear) Performance:**

* Train Accuracy: 0.99375

* Validation Accuracy: 0.815

* Train F1 Score: 0.99375

* Validation F1 Score: 0.815


## CNN Model
```cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(62, 47, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```
* Compiled with `adam` optimizer, `categorical_crossentropy` loss

* Trained for 20 epochs with batch size = 32


## Evaluation Metrics
**Used:**

* **Accuracy***

* **F1 Score (Micro)**

* **Validation Accuracy**


## Results
| Model                   | Validation Accuracy |
| ----------------------- | ------------------- |
| **SVM (Linear Kernel)** | 0.8150              |
| **CNN**                 | 0.8800              |
