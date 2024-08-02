import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def generate_pair(idx, wave_information): # matching label recording-murmur label
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
    """
    Reference: Schmidt, Samuel E., et al. "Segmentation of heart sound recordings by a duration-dependent hidden Markov model." Physiological measurement 31.4 (2010): 513.
    This code comes from the following github address.
    https://github.com/davidspringer/Springer-Segmentation-Code/tree/master
    
    We translated the code written in Matlab code to Python.
    """
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
    
    filtered = signal.resample(filtered, n_sample)
    
    filtered = normalization(filtered)
    features = np.zeros((len(filtered), 1))
    features[:, 0] = filtered
    return features

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