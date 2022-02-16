#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:22:47 2022

@author: phg17
"""

import scipy.signal as signal
import numpy as np
from .utils import Extract_Arguments, resample_label, Add_Noise
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import torch


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

def Time_diff(feature, n_difference = 1):
    #Return the Time_Derivative, but can be adjusted to different delays
    #Made sure that we could not use future data!
    #feature is a (n_feat,times) matrix
    feat_diff = np.zeros(feature.shape)
    feat_diff[:,n_difference:] = feature[:,n_difference:] - feature[:,:-n_difference]
    return feat_diff

def Resample_features(list_feat):
    #Resampe the features according to the highest sampling rate
    #Features need to have (n_components,times) structure
    n_resample = 0
    list_resampled_feat = []
    for feat in list_feat:
        if feat.shape[1] > n_resample:
            n_resample = feat.shape[1]
    for i in range(len(list_feat)):
        list_resampled_feat.append(signal.resample(list_feat[i], num = n_resample, axis = 1))
    return list_resampled_feat

def Extract_Feature(audio,fs,arg_dict):
    #Extract the features from audio and the completed arg_dict
    if arg_dict['type'] == 'MFCC':
        feat = MFCC(audio,fs,win_len = arg_dict['win_len'], 
                    winstep = arg_dict['winstep'], numcep = arg_dict['numcep'], 
                    delta_f = arg_dict['delta_f'], ndelta = arg_dict['ndelta'])
    elif arg_dict['type'] == 'FBank':
        feat = FBank(audio,fs,win_len = arg_dict['win_len'], 
                    winstep = arg_dict['winstep'], nfilt = arg_dict['nfilt'], 
                    nfft = arg_dict['nfft'])
    elif arg_dict['type'] == 'STFT':
        feat = STFT(audio,fs,window = arg_dict['window'], 
                    t_seg = arg_dict['t_seg'], 
                    Bark_scale = arg_dict['Bark_scale'])
    return feat


def Generate_Features(audio, onsets, fs, features_dict, SNR = 9):
    #Create the feature vectors from the audio sampled at fs
    #The features_dict contains the arguments for the different features 
    #the dictionary contains sub-dictionaries
    #Each sub-dictionary should contain a type category and the different 
    #arguments for that type
    #type can be 'MFCC', 'FBank' or 'STFT'
    audio = Add_Noise(audio, SNR = SNR)
    features_list = []
    for feature_key in features_dict:
        arguments_dict = Extract_Arguments(features_dict[feature_key])
        f,t,feature = Extract_Feature(audio,fs,arguments_dict)
        if arguments_dict['difference']:
            feature = Time_diff(feature,n_difference = arguments_dict['n_difference'])
        features_list.append(feature)
    resample_list = Resample_features(features_list)
    feature_vector = np.vstack(resample_list)
    onsets_resamp = resample_label(onsets, feature_vector.shape[1],fs)
    
    return feature_vector, onsets_resamp


def Generate_Batch(n_batch, data, label, length = 1500, shift = 0, computation = 'cuda'):
  batch_input = np.zeros([n_batch,data[0].shape[0],length])
  batch_output = np.zeros([n_batch,length])
  index_list = np.random.choice(np.arange(len(data)),n_batch)
  for i in range(n_batch):
    index = index_list[i]
    sentence = data[index]
    onset = label[index]
    if sentence.shape[1] > length:
      batch_input[i,:,:] = sentence[:,:length]
      batch_output[i,:] = np.roll(onset[:length],shift)
    else:
      batch_input[i,:,:sentence.shape[1]] = sentence[:,:] 
      batch_output[i,:sentence.shape[1]] = np.roll(onset[:],shift)
      batch_input[i,:,sentence.shape[1]:] = (np.random.random([sentence.shape[0],length - sentence.shape[1]]) + np.mean(sentence)) * np.std(sentence)
  #batch_input = torch.Tensor(batch_input.swapaxes(1,2)).to(computation)
  batch_output = torch.Tensor(batch_output).to(computation)
  #print('Add Noise onto the training data for augmentation')
  return batch_input, batch_output
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    


