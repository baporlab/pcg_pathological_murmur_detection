import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def generate_pair(idx, wave_information): # 1개의 recording과 1개의 murmur label을 매칭
    wave = []
    label = []
    for i in idx:
        try:
            patient = wave_information[i]
            n_loc = len(patient)
            for j in range(0, n_loc):
                wave.append(patient[j][0])
                if patient[j][-1] == 'Absent':
                    label.append(0)
                elif patient[j][-1] == 'Present':
                    label.append(1)
        except:
            pass
    label = np.array(label)
    return wave, label

def normalization(wave):
    normalized_wave = wave / np.max(np.abs(wave))
    return normalized_wave

def schmidt_spike_removal(original_signal, fs = 4000):
    windowsize = int(np.round(fs/4))
    trailingsamples = len(original_signal) % windowsize
    sampleframes = np.reshape(original_signal[0 : len(original_signal)-trailingsamples], (-1, windowsize) )
    MAAs = np.max(np.abs(sampleframes), axis = 1)
    while len(np.where(MAAs > np.median(MAAs)*3 )[0]) != 0:
        window_num = np.argmax(MAAs)
        spike_position = np.argmax(np.abs(sampleframes[window_num,:]))
        zero_crossing = np.abs(np.diff(np.sign(sampleframes[window_num, :])))
        if len(zero_crossing) == 0:
            zero_crossing = [0]
        zero_crossing = np.append(zero_crossing, 0)
        if len(np.nonzero(zero_crossing[:spike_position+1])[0]) > 0:
            spike_start = np.nonzero(zero_crossing[:spike_position+1])[0][-1]
        else:
            spike_start = 0
        zero_crossing[0:spike_position+1] = 0
        spike_end = np.nonzero(zero_crossing)[0][0]
        sampleframes[window_num, spike_start : spike_end] = 0.0001;
        MAAs = np.max(np.abs(sampleframes), axis = 1)
    despiked_signal = sampleframes.flatten()
    despiked_signal = np.concatenate([despiked_signal, original_signal[len(despiked_signal) + 1:]])
    return despiked_signal

def get_wave_features(wave, Fs = 4000, featuresFs = 2000, low = 25, high = 400):
    filtered = wave.copy()
    time = len(wave) / Fs
    n_sample = int(time * featuresFs)
    # Spike removal
    try:
        filtered = schmidt_spike_removal(filtered, fs = 4000)
    except:
        pass
    
    # filtered = butterworth_low_pass_filter(wave, 2, 800, Fs)
    # filtered = butterworth_high_pass_filter(filtered, 2, 20, Fs)
    
    filtered = signal.resample(filtered, n_sample)
    
    filtered = normalization(filtered)
    features = np.zeros((len(filtered), 1))
    features[:, 0] = filtered
    return features

# 길이 조절 (2초, 3초, 4초 실험)
def segmentation(x, y, overlap = 2000, sampling_rate = 2000, seconds = 1):
    features = []
    target = []
    
    for i in range(0, len(x)):
        for j in range(0, len(x[i]) - (4000*seconds), overlap):
            features.append(get_wave_features(x[i][j: j+(4000*seconds)], featuresFs = sampling_rate))
            target.append(y[i])
    
    features = np.array(features)
    target = np.array(target)
    return features, target

def butterworth_high_pass_filter(original_signal, order, cutoff, sampling_frequency): # high pass filter
    B_high, A_high = signal.butter(N = order, Wn = 2*cutoff/sampling_frequency, btype = 'high')
    high_pass_filtered_signal = signal.filtfilt(b = B_high, a = A_high, x = original_signal)
    return high_pass_filtered_signal

def butterworth_low_pass_filter(original_signal, order, cutoff, sampling_frequency): # low pass filter
    B_low, A_low = signal.butter(N = order, Wn = 2*cutoff/sampling_frequency, btype = 'low')
    low_pass_filtered_signal = signal.filtfilt(b = B_low, a = A_low, x = original_signal)
    return low_pass_filtered_signal


##################################
# preprocessing for external validation
def generate_pair_external_ver(idx, patinet_wave, outcome): # 1개의 recording과 1개의 murmur label을 매칭
    wave_list = []
    label_list = []
    
    for i in idx:
        patient = patinet_wave[i]
        label_list.append(outcome[i])
        wave_list.append(patient)
        
    label_list = np.array(label_list)
    return wave_list, label_list

def segmentation_external_ver(x, y, overlap = 1000):
    features = []
    target = []
    for i in range(0, len(x)):
        for j in range(0, len(x[i]) - 2000, overlap):
            features.append(get_wave_features(x[i][j: j+2000], Fs = 2000, featuresFs = 2000))
            target.append(y[i])
    features = np.array(features)
    target = np.array(target)
    return features, target