#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library to Extract info from the Timit dataset
Created on Wed Feb  2 16:58:33 2022

@author: phg17
"""

import scipy.io.wavfile as wav
import numpy as np
from os.path import join
from os import listdir
from sklearn.preprocessing import scale


def extract_onset(file):
  #Extract the onsets timing from a PHN file return a list of onsets timings    
  list_onset = []
  f = open(file, "r")
  phn = f.readlines()
  f.close()

  prev = 0
  for line in phn:
    info = line.split(' ')
    if info[2][0] in ['a','e','i','o','u'] and info[2] != 'epi\n' and prev == 0:
      start = int(info[0])
      list_onset.append(start)
      prev += 1
    elif info[2][0] in ['a','e','i','o','u'] and info[2] != 'epi\n' and prev == 1:
      prev += 1
    else:
      prev = 0
  return list_onset


def extract_audio(file):
  #Return audio, fs from wavfile
  samplerate, audio = wav.read(file)
  return audio


def extract_TIMIT(directory):
    train_data = dict()
    train_label = dict()
    test_data = dict()
    test_label = dict()
    train_idx = 0
    test_idx = 0
    
    for repo in listdir(directory):
      current_repo = join(directory,repo)
      for dialect in listdir(current_repo):
        current_dialect = join(current_repo,dialect)
        for speaker in listdir(current_dialect):
          current_speaker = join(current_dialect,speaker)
          #print(sorted(listdir(current_speaker)))
          for file in sorted(listdir(current_speaker)):
            current_file = join(current_speaker,file)
            if current_file[-3:] == 'PHN':
              #print(file)
              list_onset = extract_onset(current_file)
    
              audio_file = current_file[:-3] + 'WAV.wav'
              audio = scale(extract_audio(audio_file))
              onset = np.zeros(len(audio))
              for i in list_onset:
                onset[i] = 1
              if repo == 'TRAIN':
                train_data[train_idx] = audio
                train_label[train_idx] = onset
                train_idx += 1
              else:
                test_data[test_idx] = audio
                test_label[test_idx] = onset
                test_idx += 1
    return train_data, train_label, test_data, test_label


