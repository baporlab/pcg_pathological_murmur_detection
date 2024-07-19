import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def get_split_for_murmurnet(wave_set, demo_set, outcomes_set, grade_set, id_set, test_size = 0.2):
    
    """
    x: wave_info
    y: grade
    test_size: 0 ~ 1
    classifier: 'strong', 'weak', 'base'
    """
    
    grade_set[grade_set == 3] = 2 # combine grade 2 and 3
    
    # except non-murmur abnormal cases
    except_nonmurmur_abnormal = []
    for i in range(0, len(wave_set)):
        if (grade_set[i] == 0)&(outcomes_set[i] == 1):
            except_nonmurmur_abnormal.append(i)
    
    wave_set_new = []
    demo_set_new = []
    grade_set_new = []
    outcomes_set_new = []
    id_set_new = []
    
    for i in range(0, len(wave_set)):
        if i not in except_nonmurmur_abnormal:
            wave_set_new.append(wave_set[i])
            demo_set_new.append(demo_set[i])
            grade_set_new.append(grade_set[i])
            outcomes_set_new.append(outcomes_set[i])
            id_set_new.append(id_set[i])
    
    demo_set_new = np.array(demo_set_new)
    grade_set_new = np.array(grade_set_new)
    outcomes_set_new = np.array(outcomes_set_new)
    id_set_new = np.array(id_set_new)
    
    train_wave, test_wave, train_grade, test_grade = train_test_split(wave_set_new, grade_set_new, stratify = grade_set_new, test_size = test_size, random_state = 42)
    train_outcome, test_outcome, train_grade1, test_grade1 = train_test_split(outcomes_set_new, grade_set_new, stratify = grade_set_new, test_size = test_size, random_state = 42)
    train_demo, test_demo, train_grade2, test_grade2 = train_test_split(demo_set_new, grade_set_new, stratify = grade_set_new, test_size = test_size, random_state = 42)
    train_sub_id, test_sub_id, train_grade3, test_grade3 = train_test_split(id_set_new, grade_set_new, stratify = grade_set_new, test_size = test_size, random_state = 42)
    
    # Split 5 Fold
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    fold = []
    for train_index, valid_index in skf.split(train_wave, train_grade):
        fold.append(valid_index)
    
    neg_idx = np.where( (train_grade == 0)&(train_outcome == 0) )[0]
    pos_idx = np.where( (train_grade >= 1)&(train_outcome == 1) )[0]
    strong_idx = np.where( (train_grade >= 2)&(train_outcome == 1) )[0]
    weak_idx = np.where( (train_grade == 1)&(train_outcome == 1) )[0]
    inno_idx = np.where( (train_grade >= 1)&(train_outcome == 0) )[0]
    idx_list = [neg_idx, pos_idx, strong_idx, weak_idx, inno_idx]

    return train_wave, test_wave, train_outcome, test_outcome, train_demo, test_demo, train_grade, test_grade, fold, idx_list, train_sub_id, test_sub_id

def get_split_for_non_murmur(wave_set, demo_set, outcomes_set, grade_set, id_set, test_size = 0.2):
    
    """
    x: wave_info
    y: grade
    test_size: 0 ~ 1
    classifier: 'strong', 'weak', 'base'
    """
    
    grade_set[grade_set == 3] = 2 # combine grade 2 and 3
    
    # non-murmur abnormal cases
    except_nonmurmur_abnormal = []
    for i in range(0, len(wave_set)):
        if (grade_set[i] == 0)&(outcomes_set[i] == 1):
            except_nonmurmur_abnormal.append(i)
        if (grade_set[i] == 0)&(outcomes_set[i] == 0):
            except_nonmurmur_abnormal.append(i)
    
    wave_set_new = []
    demo_set_new = []
    grade_set_new = []
    outcomes_set_new = []
    id_set_new = []
    
    for i in range(0, len(wave_set)):
        if i in except_nonmurmur_abnormal:
            wave_set_new.append(wave_set[i])
            demo_set_new.append(demo_set[i])
            grade_set_new.append(grade_set[i])
            outcomes_set_new.append(outcomes_set[i])
            id_set_new.append(id_set[i])
    
    demo_set_new = np.array(demo_set_new)
    grade_set_new = np.array(grade_set_new)
    outcomes_set_new = np.array(outcomes_set_new)
    id_set_new = np.array(id_set_new)
    
    train_wave, test_wave, train_outcome, test_outcome = train_test_split(wave_set_new, outcomes_set_new, stratify = outcomes_set_new, test_size = test_size, random_state = 42)
    train_grade, test_grade, train_grade1, test_grade1 = train_test_split(grade_set_new, outcomes_set_new, stratify = outcomes_set_new, test_size = test_size, random_state = 42)
    train_demo, test_demo, train_grade2, test_grade2 = train_test_split(demo_set_new, outcomes_set_new, stratify = outcomes_set_new, test_size = test_size, random_state = 42)
    train_sub_id, test_sub_id, train_grade3, test_grade3 = train_test_split(id_set_new, outcomes_set_new, stratify = outcomes_set_new, test_size = test_size, random_state = 42)
    
    # Split 5 Fold
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    fold = []
    for train_index, valid_index in skf.split(train_wave, train_outcome):
        fold.append(valid_index)
    
    neg_idx = np.where( (train_grade == 0)&(train_outcome == 0) )[0]
    pos_idx = np.where( (train_grade >= 1)&(train_outcome == 1) )[0]
    strong_idx = np.where( (train_grade >= 2)&(train_outcome == 1) )[0]
    weak_idx = np.where( (train_grade == 1)&(train_outcome == 1) )[0]
    inno_idx = np.where( (train_grade >= 1)&(train_outcome == 0) )[0]
    idx_list = [neg_idx, pos_idx, strong_idx, weak_idx, inno_idx]

    return train_wave, test_wave, train_outcome, test_outcome, train_demo, test_demo, train_grade, test_grade, fold, idx_list, train_sub_id, test_sub_id