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