# Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms

This repository is implement code for "Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms"

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

# Dataset
Our study was performed on two publicly accessible database, "Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016" and "The CirCor DigiScope Phonocardiogram Dataset"

# Implementation
The following code help you to load dataset.
```python
from load_data import *
path = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/' # this path is dataset path of PhysioNet 2022
wave_info, demo_info, outcomes, grade, paitnet_id = get_dataset(path)
```
The dataset was split to noraml heart sound and pathological heart sound.
The following code separates training and test data based on patient and murmur/outcome. 
```python
from sklearn.model_selection import train_test_split
from split_data import *
train_x, test_x, train_outcome, test_outcome, train_demo, test_demo, train_grade, test_grade, fold, mur_list, train_sub_id, test_sub_id = get_split_for_murmurnet(
                                            wave_info, demo_info, outcomes, grade, test_size = 0.2, id_set = paitnet_id) # Data split by patient id
print(len(train_x), len(test_x))
# mur_list = neg_idx, pos_idx, strong_idx, weak_idx, inno_idx

ensemble1, ensemble2 = train_test_split(mur_list[1], test_size = len(mur_list[2]), random_state = 42)
ensemble1, ensemble2 = train_test_split(mur_list[1], test_size = len(mur_list[2]), random_state = 42)

murmurnet_domain1 = np.concatenate(  [mur_list[0], mur_list[1]] ) # total domain (normal vs pathological murmur)
murmurnet_domain2 = np.concatenate(  [mur_list[0], mur_list[2]] ) # strong murmur domain
murmurnet_domain3 = np.concatenate(  [mur_list[0], mur_list[3]] ) # weak murmur domain
murmurnet_domain4 = np.concatenate(  [mur_list[0], ensemble1] ) # random ensemble1 domain (sample size was same to weak domain)
murmurnet_domain5 = np.concatenate(  [mur_list[0], ensemble2] ) # random ensemble2 domain (sample size was same to strong domain)

# Training domain
murmurnet_domain = murmurnet_domain1.copy() # Data domain selection

for i in range(0, 5):
    temp_fold = fold[i]
    del_idx = []
    for j in range(0, len(temp_fold)):
        if temp_fold[j] not in murmurnet_domain:
            del_idx.append(j)
    fold[i] = np.delete(temp_fold, del_idx)
```

