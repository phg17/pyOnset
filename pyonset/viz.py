#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:09:27 2022

@author: phg17
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_features(feat_vect,onset_vect = None, samplerate = 16000,vmin=0,vmax=1,shading = 'gouraud',size = [10,5],alpha = 1):
    #Plot feat_vect (n_compo X n_times)
    fig,ax = plt.subplots()
    t = np.arange(feat_vect.shape[1])
    f = np.arange(feat_vect.shape[0])
    ax.pcolormesh(t, f, feat_vect, vmin=0, vmax=1, shading='gouraud')
    ax.set_title('Feature Matrix')
    ax.set_ylabel('Components')
    ax.set_xlabel('Time')
    if (onset_vect).any():
        ax.plot(np.arange(len(onset_vect))/len(onset_vect)*t[-1] ,onset_vect*(feat_vect.shape[0]-1),'--r',alpha=1)
    fig.set_size_inches(size)
    print('Try to add argdicts to this so that the content is detailed')