# Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms

This repository is implement code for "Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms"

# Implementation
The training and validation were performed by sequentially implementing the notebook "Implement_notebook.ipynb". The procedures consist of four steps: 1) loading and splitting the dataset, 2) training the feature extraction model based on temporal convolutional networks, 3) feature extraction, and 4) validation at the record- and patient-level. Since this repository does not contain a dataset, the dataset need to first be downloaded before running the code.
## Dataset
Our study was conducted using two publicly available databases, "Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016" (PNC2016) and "The CirCor DigiScope Phonocardiogram Dataset" (PNC2022).

## Requirements
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
## Implementation
The following code sets the domain of the training dataset based on the intensity of the murmur.
```python
ensemble1, ensemble2 = train_test_split(mur_list[1], test_size = len(mur_list[2]), random_state = 42)

murmurnet_domain1 = np.concatenate(  [mur_list[0], mur_list[1]] ) # total domain (normal vs pathological murmur)
murmurnet_domain2 = np.concatenate(  [mur_list[0], mur_list[2]] ) # strong murmur domain
murmurnet_domain3 = np.concatenate(  [mur_list[0], mur_list[3]] ) # weak murmur domain
murmurnet_domain4 = np.concatenate(  [mur_list[0], ensemble1] ) # random ensemble1 domain (sample size was same to weak domain)
murmurnet_domain5 = np.concatenate(  [mur_list[0], ensemble2] ) # random ensemble2 domain (sample size was same to strong domain)

# Training domain
murmurnet_domain = murmurnet_domain1.copy() # Data domain selection
```
If you want to train other convolutional networks, please replace the "model_name" variable of the "training_model" function
```python
train_loss, train_metric, valid_loss, valid_metric = training_models(train_collection, valid_collection, save_path = save_path + model,
                                                                     gpu_num = 1, random_seed = 42, num_epochs = 10, batch_size = 128, learning_rate = 0.001, patience = 3, n_ch = 1,
                                                                     model_name = 'tcn' # training various model. please select model option of [vgg16, resnet, inception, tcn, ctan, convnext]
                                                                    ) # if train weak model, set batch_size = 128 because of error "Only one class present in y_true"
```
