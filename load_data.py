import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy
import warnings
warnings.filterwarnings('ignore')

from helper_code import *
###########################################################################################
# Custom Helper Function

"""
Code example

path = '/srv/project_data/Physionet2022_PCG/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/'
wave_info, demo_info, outcomes, grade = get_dataset(path)

wave_info: list, collection of patient's wave, segmentation, locations
demo_info: numpy array. (age, height, weight, bmi, sex_features, is_pregnant)

outcomes: Heart disease.
          Normal: 0
          Abnormal: 1

grade: Murmur Grade
       0: not murmur
       1: murmur grade 1
       2: murmur grade 2
       3: murmur grade 3
"""
# Load recordings(custom ver).
def custom_load_recordings(data_folder, data):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]
    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        ##############################
        loc_murmur = data.split('\n')[1+num_locations+5][9:]
        recording_file = entries[2]
        tsv_file = entries[3]
        recording_loc = entries[0][:2] # location 추가
        if loc_murmur == 'Present':
            if recording_loc in data.split('\n')[1+num_locations+6]:
                loc_murmur = 'Present'
            else:
                loc_murmur = 'Absent'
        ##############################
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        segment = pd.read_csv(os.path.join(data_folder, tsv_file), sep = '\t', names = ['start', 'end', 'seg'])
        recordings.append([recording, segment, recording_loc, loc_murmur])
        frequencies.append(frequency)
    return recordings

def custom_get_features(data):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    else:
        age = -1 # 6 * 12 # mode value

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)
    sex_features = -1
    if compare_strings(sex, 'Female'):
        sex_features = 1
    if compare_strings(sex, 'Male'):
        sex_features = 0

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)
    if np.isnan(height):
        height = -1 #110 # mean value
    if np.isnan(weight):
        weight = -1 #23 # mean value
    bmi = np.round(weight / ((height/100)**2) ,2)
    
    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)
    
    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    
    features = np.hstack(([age], [height], [weight], [bmi], [sex_features], [is_pregnant]))

    return np.asarray(features, dtype=np.float32)

def get_dataset(data_folder):
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Normal', 'Abnormal']
    num_outcome_classes = len(outcome_classes)
    ########################################################################### My Custom Processing
    wave_info = list()
    wave_info2 = list()
    demo_info = list()
    murmurs = list()
    outcomes = list()
    grade = list()
    patient_id = list()
    # Get Data
    for i in range(num_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = custom_load_recordings(data_folder, current_patient_data)
        current_recordings2 = custom_load_recordings(data_folder, current_patient_data)
        current_features = custom_get_features(current_patient_data)
        
        # grade 추가
        current_sys_grade = current_patient_data.split('\n')[-12][26:]
        current_dia_grade = current_patient_data.split('\n')[-7][27:]
        if current_dia_grade != 'nan': # Except diastolic murmur
            continue
        current_grade = current_sys_grade

        # Extract features.
        wave_info.append(current_recordings)
        wave_info2.append(current_recordings2)
        demo_info.append(current_features)
        grade.append(current_grade)
        patient_id.append(patient_files[i])

        # Extract labels and use one-hot encoding.
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)
    
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)
    demo_info = np.vstack(demo_info)
    patient_id = np.vstack(patient_id)

    # Get Segmentation data
    for i in range(0, len(wave_info)): # Use only segmentation part
        n_loc = len(wave_info[i])
        for j in range(0, n_loc):
            tsv = wave_info[i][j][1]
            tsv = tsv[tsv.seg != 0].reset_index(drop = True)
            start = int(tsv.start[0]*4000)
            end = int(tsv.end[len(tsv)-1]*4000)
            wave_info[i][j][0] = wave_info[i][j][0][start:end]

    # Grade encoding
    for i in range(0, len(grade)):
        if grade[i] == 'III/VI':
            grade[i] = 3
        if grade[i] == 'II/VI':
            grade[i] = 2
        if grade[i] == 'I/VI':
            grade[i] = 1
        if grade[i] == 'nan':
            grade[i] = 0
    grade = np.array(grade)

    # Except Unknown
    temp_inpo = []
    temp_label = []
    temp_outcomes = []
    temp_demo = []
    temp_id = []

    for i in range(0, len(murmurs)):
        if np.argmax(murmurs, axis = 1)[i] != 1:
            temp_inpo.append(wave_info[i])
            temp_label.append(grade[i])
            temp_outcomes.append(np.argmax(outcomes[i]))
            temp_demo.append(demo_info[i])
            temp_id.append(patient_id[i])

    wave_info = temp_inpo
    grade = np.array(temp_label)
    outcomes = np.array(temp_outcomes)
    demo_info = np.array(temp_demo) # ['age', 'height', 'weight', 'bmi', 'sex_features', 'is_pregnant']
    patient_id = np.array(temp_id)
    return wave_info, demo_info, outcomes, grade, patient_id # , wave_info2, murmurs