# Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms

This repository is implement code for "Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms"

# Implementation
The training and validation were performed by sequentially implementing the notebook "Implement_notebook.ipynb". The processes consist of four step 1) Load and split dataset, 2) Training of the Feature extraction model based on temporal convolutional networks, 3) Feature extraction, and 4) record- and patient-level validation. This repository does not include dataset, therfore downloading the dataset first before running the code.
## Dataset
Our study was performed on two publicly accessible database, "Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016" and "The CirCor DigiScope Phonocardiogram Dataset"

# Requirements
This repository requires the library below to implement our code.
```
numpy==1.22.0
scipy==1.7.3
pandas==1.4.4
matplotlib==3.5.3
seaborn==0.13.2
sklearn==1.0.2
joblib==1.1.1
torch==2.1.2+cu121
torchsummary
```
