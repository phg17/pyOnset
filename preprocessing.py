#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:22:47 2022

@author: phg17
"""

import scipy.signal as signal
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


def STFT(audio,fs = 16000, window = "hann", t_seg = 0.005, Bark_scale = [0,50,150,250,350,450,570,700,840,1000,1170,1370,1600,1850,2150,2500,2900,3400,4000,4800,5800,7000,8500,10500,13500]):
    #Return the STFT of the signal as well as the features and time
    #Bark_scale can be left empty to have all Components of STFT
    f,t,stft = signal.stft(audio,fs=fs,window="hann",nperseg=int(t_seg*fs))
    if Bark_scale:
        feat = np.zeros([len(Bark_scale)-1,stft.shape[1]])
        for bark in range(len(Bark_scale)-1):
              inf = Bark_scale[bark]
              sup = Bark_scale[bark+1]
              freqs = []
              for i in range(len(f)):
                if f[i]>inf and f[i]<sup:
                  freqs.append(i)
              if len(freqs)>0:
                feat[bark,:] = np.mean(np.abs(stft[freqs,:]),axis=0)
        return np.arange(feat.shape[0]),t,feat
    else:
        return f,t,np.abs(stft)

def MFCC(audio,fs = 16000, win_len = 0.01,winstep = 0.005, numcep = 13, delta_f = False, ndelta = 2):
    #Return the MFCC of the signal as well as the features and time
    #Up tp 26 MFCC
    mfcc_feat = mfcc(audio,fs,winlen = 0.01,winstep = winstep,numcep = 26)
    if delta_f:
        mfcc_feat = delta(mfcc_feat, 2)
    mfcc_feat = mfcc_feat.T
    f = np.arange(mfcc_feat.shape[0])
    t = np.arange(mfcc_feat.shape[1])*winstep
    return f,t,mfcc_feat

def FBank(audio,fs = 16000, win_len = 0.01,winstep = 0.005, nfilt = 26, nfft = 512):
    #Return the FilterBank of the signal as well as the features and time
    #Up tp 26 MFCC
    fbank_feat = logfbank(audio,fs,winlen = win_len,winstep = winstep, nfilt = nfilt, nfft = nfft)
    fbank_feat = fbank_feat.T
    f = np.arange(fbank_feat.shape[0])
    t = np.arange(fbank_feat.shape[1])*winstep
    return f,t,fbank_feat


