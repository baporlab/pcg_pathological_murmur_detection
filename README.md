# Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms

This repository is implement code for "Temporal Convolutional Neural Network-Based Feature Extraction and Asynchronous Channel Information Fusion Method for Heart Abnormality Detection in Phonocardiograms"

# Explanation


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
The following is the training code for the feature extractiom model. 
We performed 5-fold cross validation (three folds for training, one fold for optimization, and one fold for validation)
You can change the model from the folder "ModelClass"
```python
from preprocessing import *
from training_models import *
from load_models import *

train = [[0,1,2], [1,2,3], [2,3,4], [3,4,0], [4,0,1]]
valid1 = [3,4,0,1,2]
valid2 = [4,0,1,2,3]
sampling_rate = 2000

for n_fold in range(0, 5):
    train_fold = train[n_fold] 
    train_fold1 = train_fold[0]
    train_fold2 = train_fold[1]
    train_fold3 = train_fold[2]
    
    valid1_fold = valid1[n_fold]
    valid2_fold = valid2[n_fold]
    
    train_idx = np.concatenate((fold[train_fold1], fold[train_fold2], fold[train_fold3]))
    valid1_idx = fold[valid1_fold]
    valid2_idx = fold[valid2_fold]
    
    train_wav, train_wav_label = generate_pair(train_idx, train_x)
    valid1_wav, valid1_wav_label = generate_pair(valid1_idx, train_x)
    valid2_wav, valid2_wav_label = generate_pair(valid2_idx, train_x)
    print(len(train_wav), len(valid1_wav), len(valid2_wav))
    
    train_features, train_target = segmentation(train_wav, train_wav_label, sampling_rate = sampling_rate, seconds = 1)
    valid1_features, valid1_target = segmentation(valid1_wav, valid1_wav_label, sampling_rate = sampling_rate, seconds = 1)
    valid2_features, valid2_target = segmentation(valid2_wav, valid2_wav_label, sampling_rate = sampling_rate, seconds = 1)
    
    # reshape for Pytorch
    train_features = train_features.reshape((train_features.shape[0], train_features.shape[2], train_features.shape[1]))
    valid1_features = valid1_features.reshape((valid1_features.shape[0], valid1_features.shape[2], valid1_features.shape[1]))
    print(train_features.shape, valid1_features.shape, valid2_features.shape)
    print(train_target.shape, valid1_target.shape, valid2_target.shape)
    
    arr = np.arange(len(train_features))
    np.random.seed(42)
    np.random.shuffle(arr)
    train_features = train_features[arr]
    train_target = train_target[arr]
    plt.plot(train_features[0].flatten())
    
    train_collection = []
    valid_collection = []
    for i in range(0, len(train_features)):
        train_collection.append([train_features[i], np.array([train_target[i]])])
    for i in range(0, len(valid1_features)):
        valid_collection.append([valid1_features[i], np.array([valid1_target[i]])])
    
    # Training
    model_name = 'TCN_fold{}_' # Saved model name
    model = model_name.format(n_fold+1)
    save_path = 'save_model/' # model saved folder
    train_loss, train_metric, valid_loss, valid_metric = training_models(train_collection, valid_collection, save_path = save_path + model,
                                                                         gpu_num = 1, random_seed = 42, num_epochs = 10, batch_size = 128, learning_rate = 0.001, patience = 3, n_ch = 1,
                                                                         model_name = 'tcn' # training various model. please select model option of [vgg16, resnet, inception, tcn, ctan, convnext]
                                                                        ) # if train weak model, set batch_size = 128 because of error "Only one class present in y_true"
    df = pd.DataFrame({'train_loss':train_loss, 'train_metric':train_metric, 'valid_loss':valid_loss, 'valid_metric':valid_metric})
    df.to_csv(save_path + model + 'model_loss.csv')

```
The feature extraction from a trained model is performed following code.
```python
model_path = 'save_model/' # write your path of saved model

train_murmur_loc = []
for i in range(0, len(train_x)):
    current_patient = train_x[i]
    current_murmur = ''
    for j in range(0, len(current_patient)):
        current_loc = current_patient[j]
        if current_loc[-1] == 'Present':
            current_murmur = current_murmur + current_loc[-2] + '+'
    train_murmur_loc.append(current_murmur)

test_murmur_loc = []
for i in range(0, len(test_x)):
    current_patient = test_x[i]
    current_murmur = ''
    for j in range(0, len(current_patient)):
        current_loc = current_patient[j]
        if current_loc[-1] == 'Present':
            current_murmur = current_murmur + current_loc[-2] + '+'
    test_murmur_loc.append(current_murmur)

train = [[0,1,2], [1,2,3], [2,3,4], [3,4,0], [4,0,1]]
valid1 = [3,4,0,1,2]
valid2 = [4,0,1,2,3]

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df_cv = []
valid_df_cv = []
test_df_cv = []
for n_fold in range(0, 5):
    train_fold = train[n_fold]
    train_fold1 = train_fold[0]
    train_fold2 = train_fold[1]
    train_fold3 = train_fold[2]
    
    valid1_fold = valid1[n_fold]
    valid2_fold = valid2[n_fold]
    
    train_idx = np.concatenate((fold[train_fold1], fold[train_fold2], fold[train_fold3], fold[valid1_fold]))
    valid_idx = fold[valid2_fold]
    
    """Load model"""
    path = 'save_model/TCN_fold{}_model.pt'.format(n_fold + 1)
    param_saved = torch.load(path, map_location = device)
    test_model = load_model(model_name = 'tcn', n_ch = 1).to(device)
    test_model.load_state_dict(param_saved)
    test_model.eval()
    """"""
    
    train_df = pd.DataFrame(np.zeros((len(train_idx), 24)), columns = ['AV_min', 'AV_max', 'AV_median', 'AV_mean',
                                                                       'PV_min', 'PV_max', 'PV_median', 'PV_mean',
                                                                       'TV_min', 'TV_max', 'TV_median', 'TV_mean',
                                                                       'MV_min', 'MV_max', 'MV_median', 'MV_mean',
                                                                       'grade', 'outcome', 'age', 'height', 'weight', 'bmi', 'sex_features', 'is_pregnant']) - 1
    
    valid_df = pd.DataFrame(np.zeros((len(valid_idx), 24)), columns = ['AV_min', 'AV_max', 'AV_median', 'AV_mean',
                                                                       'PV_min', 'PV_max', 'PV_median', 'PV_mean',
                                                                       'TV_min', 'TV_max', 'TV_median', 'TV_mean',
                                                                       'MV_min', 'MV_max', 'MV_median', 'MV_mean',
                                                                       'grade', 'outcome', 'age', 'height', 'weight', 'bmi', 'sex_features', 'is_pregnant']) - 1
    
    test_df = pd.DataFrame(np.zeros((len(test_x), 24)), columns = ['AV_min', 'AV_max', 'AV_median', 'AV_mean',
                                                                   'PV_min', 'PV_max', 'PV_median', 'PV_mean',
                                                                   'TV_min', 'TV_max', 'TV_median', 'TV_mean',
                                                                   'MV_min', 'MV_max', 'MV_median', 'MV_mean',
                                                                   'grade', 'outcome', 'age', 'height', 'weight', 'bmi', 'sex_features', 'is_pregnant']) - 1
    
    train_df['grade'] = train_grade[train_idx]
    valid_df['grade'] = train_grade[valid_idx]
    test_df['grade'] = test_grade
    train_df['outcome'] = train_outcome[train_idx]
    valid_df['outcome'] = train_outcome[valid_idx]
    test_df['outcome'] = test_outcome
    
    train_df['mur_loc'] = [train_murmur_loc[k] for k in train_idx]
    valid_df['mur_loc'] = [train_murmur_loc[k] for k in valid_idx]
    test_df['mur_loc'] = test_murmur_loc
    
    train_df[['age', 'height', 'weight', 'bmi', 'sex_features', 'is_pregnant']] = train_demo[train_idx]
    valid_df[['age', 'height', 'weight', 'bmi', 'sex_features', 'is_pregnant']] = train_demo[valid_idx]
    test_df[['age', 'height', 'weight', 'bmi', 'sex_features', 'is_pregnant']] = test_demo
    
    train_df['patient_id'] = train_id[train_idx]
    valid_df['patient_id'] = train_id[valid_idx]
    test_df['patient_id'] = test_id
    
    for i, idx in enumerate(train_idx):
        current_patient = train_x[idx]
        strong_features = []
        weak_features = []
        for j in range(0, len(current_patient)):
            features = []
            loc_name = current_patient[j][2]
            if loc_name == 'Ph':
                continue
            loc_wave = current_patient[j][0]
            for n in range(0, len(loc_wave) - 4000, 2000):
                features.append(get_wave_features(loc_wave[n: n + 4000], featuresFs = 2000))
            features = np.array(features)
            if len(features) != 0:
                features = features.reshape((len(features), 1, 2000))
                features = torch.Tensor(features)
                features = features.to(device).float()
                pred_strong = test_model(features).cpu().detach().flatten().numpy()
                
                # length.append(len(pred))
                mean_feature_strong = np.mean(pred_strong)
                median_feature_strong = np.median(pred_strong)
                max_feature_strong = np.quantile(pred_strong,  q = 0.95)
                min_feature_strong = np.quantile(pred_strong, q = 0.05)
                
                train_df[loc_name + '_min'][i] = min_feature_strong
                train_df[loc_name + '_max'][i] = max_feature_strong
                train_df[loc_name + '_median'][i] = median_feature_strong
                train_df[loc_name + '_mean'][i] = mean_feature_strong

    for i, idx in enumerate(valid_idx):
        current_patient = train_x[idx]
        strong_features = []
        weak_features = []
        for j in range(0, len(current_patient)):
            features = []
            loc_name = current_patient[j][2]
            if loc_name == 'Ph':
                continue
            loc_wave = current_patient[j][0]
            for n in range(0, len(loc_wave) - 4000, 2000):
                features.append(get_wave_features(loc_wave[n: n + 4000], featuresFs = 2000))
            features = np.array(features)
            if len(features) != 0:
                features = features.reshape((len(features), 1, 2000))
                features = torch.Tensor(features).to(device).float()
                pred_strong = test_model(features).cpu().detach().flatten().numpy()
                
                mean_feature_strong = np.mean(pred_strong)
                median_feature_strong = np.median(pred_strong)
                max_feature_strong = np.quantile(pred_strong,  q = 0.95)
                min_feature_strong = np.quantile(pred_strong, q = 0.05)
                
                valid_df[loc_name + '_min'][i] = min_feature_strong
                valid_df[loc_name + '_max'][i] = max_feature_strong
                valid_df[loc_name + '_median'][i] = median_feature_strong
                valid_df[loc_name + '_mean'][i] = mean_feature_strong
    for i in range(0, len(test_x)):
        current_patient = test_x[i]
        strong_features = []
        weak_features = []
        for j in range(0, len(current_patient)):
            features = []
            loc_name = current_patient[j][2]
            if loc_name == 'Ph':
                continue
            loc_wave = current_patient[j][0]
            for n in range(0, len(loc_wave) - 4000, 2000):
                features.append(get_wave_features(loc_wave[n: n + 4000], featuresFs = 2000))
            features = np.array(features)
            if len(features) != 0:
                features = features.reshape((len(features), 1, 2000))
                features = torch.Tensor(features)
                features = features.to(device).float()
                pred_strong = test_model(features).cpu().detach().flatten().numpy()
                
                # length.append(len(pred))
                mean_feature_strong = np.mean(pred_strong)
                median_feature_strong = np.median(pred_strong)
                max_feature_strong = np.quantile(pred_strong,  q = 0.95)
                min_feature_strong = np.quantile(pred_strong, q = 0.05)
                
                test_df[loc_name + '_min'][i] = min_feature_strong
                test_df[loc_name + '_max'][i] = max_feature_strong
                test_df[loc_name + '_median'][i] = median_feature_strong
                test_df[loc_name + '_mean'][i] = mean_feature_strong
    train_df_cv.append(train_df)
    valid_df_cv.append(valid_df)
    test_df_cv.append(test_df)

```
