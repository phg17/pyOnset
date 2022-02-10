#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for Onset Extraction and general purpose Audio processing
Created on Wed Feb  2 14:34:31 2022

@author: phg17
"""


import scipy.io.wavfile as wav
import scipy.signal as signal
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from time import time, sleep
from sklearn.preprocessing import scale
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

Default_arg = dict()
Default_arg['MFCC'] = {'win_len':0.01,'winstep':0.005,'numcep':13,'delta_f':False, 'ndelta':2,
               'difference':False, 'n_difference' :0}
Default_arg['FBank'] = {'win_len':0.01,'winstep':0.005,'nfilt':26,'nfft':512,
               'difference':False, 'n_difference' :0}
Default_arg['STFT'] = {'window':'hann','t_seg':0.005,
               'Bark_scale': [0,50,150,250,350,450,570,700,840,1000,1170,
                              1370,1600,1850,2150,2500,2900,3400,4000,4800,
                              5800,7000,8500,10500,13500],
               'difference':False, 'n_difference' :0}


def RMS(signal):
    """RMS handler"""
    return np.sqrt(np.mean(np.power(signal, 2)))

def Add_Noise(signal, SNR = 0):
  RMS_s = RMS(signal)
  noise = np.random.normal(0,1,len(signal))
  RMS_n = RMS_s*(10**(-SNR/20.))
  noise *= RMS_n / RMS(noise)
  
  return signal + noise

def resample_label(label, l2 ,samplerate = 16000):
    #resample a list of onsets to a new length, the new position is set by the
    #int rounding
    onsets = []
    l1 = len(label)
    for i in range(l1):
      if label[i]>0:
        onsets.append(i)
    onsets = (np.asarray(onsets)/l1*l2).astype(int)
    label_output = np.zeros(l2)
    for i in onsets:
      label_output[i] = 1
    return label_output

def Extract_Arguments(arg_dict):
    #Return the updated list of arguments based on the type of features
    #Type can be MFCC, STFT or FBank
    if arg_dict['type'] not in ['MFCC','FBank','STFT']:
        raise AssertionError('Type of Feature not supported, try MFCC, FBank or STFT')
    else:
        default_arg = Default_arg[arg_dict['type']]
        for arg_key in default_arg:
            if arg_key not in arg_dict:
                arg_dict[arg_key] = default_arg[arg_key]
    return arg_dict

def Default_Args(feat_type):
    return Default_arg[feat_type]

def p2a(peaks, length = 1000):
    #peaks to array, samplewise
    output = np.zeros(length)
    for i in peaks:
        output[i] = 1
    return output

def a2p(array):
    #array to peaks, samplewise
    output = []
    for i in range(len(array)):
        if array[i] > 0:
            output.append(i)
    return output

def prob2array(proba_function_batch,height = 0.02, threshold = 0.001, distance = 45, prominence = 0.001):
    #convert a batch of proba_function to a batch of onsets array
    peak_list = []
    for proba_function in proba_function_batch:
        peak_list.append(signal.find_peaks(proba_function, height =height, threshold = threshold, distance = distance, prominence = prominence)[0])
    print('Implement Real_time peak generation by symmetry')
    return peak_list
    
    



    
