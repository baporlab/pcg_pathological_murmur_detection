# Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms

This repository is implement code for "Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms"

# Dataset
Our study was performed on two publicly accessible database, "Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016" and "The CirCor DigiScope Phonocardiogram Dataset"

# Implementation
```python
from load_data import *
path = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/' # this path is dataset path of PhysioNet 2022
wave_info, demo_info, outcomes, grade, paitnet_id = get_dataset(path)
```
