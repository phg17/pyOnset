#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:09:27 2022

@author: phg17
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_features(feat_vect,onset_vect = None,vmin=0,vmax=1,shading = 'gouraud'):
    #Plot feat_vect (n_compo X n_times)
    fig,ax = plt.subplots()
    t = np.arange(feat_vect.shape[1])
    f = np.arange(feat_vect.shape[0])
    ax.pcolormesh(t, f, feat_vect, vmin=0, vmax=1, shading='gouraud')
    ax.set_title('Feature Matrix')
    ax.set_ylabel('Components')
    ax.set_xlabel('Time')
    if onset_vect:
        ax.plot(t ,onset_vect*(feat_vect.shape[0]-1),'--r',alpha=0.25)
    fig.set_size_inches([20,15])
    print('Try to add argdicts to this so that the content is detailed')