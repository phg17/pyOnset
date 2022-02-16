#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:09:27 2022

@author: phg17
"""

import matplotlib.pyplot as plt
import numpy as np
from .utils import a2p,p2a

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
    return fig,ax
    
    
def plot_onsets(onsets_1, onsets_2, proba_function = [], samplerate = 1,size = [10,5], onsets = False):
    #Plot two onsets, the onsets bool indicate if they are in the form of 
    #timing list or arrays
    if onsets:
        onsets1 = p2a(onsets_1)
        onsets2 = p2a(onsets_2)
        peaks1 = onsets_1
        peaks2 = onsets_2
    else:
        onsets1 = onsets_1
        onsets2 = onsets_2
        peaks1 = a2p(onsets_1)
        peaks2 = a2p(onsets_2)
    fig,ax = plt.subplots()
    t1 = np.arange(onsets1.shape[0]) / samplerate
    t2 = np.arange(onsets2.shape[0]) / samplerate

    ax.plot(t1,np.zeros(len(t1)), marker = "|", color = 'k',linewidth = 2)
    ax.plot([t1[peaks1],t1[peaks1]],[0,1], lw=2, alpha=0.8,color = 'k')
    ax.plot([t2[peaks2],t2[peaks2]],[0,-1], lw=2, alpha=0.8, color = 'r')
    ax.plot(t1[peaks1],onsets1[peaks1],'k',linewidth = 4,marker="o", ls="", ms=5)
    ax.plot(t2[peaks2],onsets2[peaks2] * -1,'r',linewidth = 4,marker="o", ls="", ms=5)
    if len(proba_function) > 0:
        t3 = np.arange(len(proba_function)) / samplerate
        ax.plot(t3, proba_function, 'g')
    fig.set_size_inches(size)
    print('Try to add argdicts to this so that the content is detailed')
    return fig,ax
    

def plot_differences(differences, bins = 20):
    #plot the differences with differences in the form of a dictionary with
    #keys 'FP', 'FN' and 'Match'
    fig,ax = plt.subplots(2)
    ax[0].bar(['FP','FN',],[differences['FP'] / (differences['FP'] + len(differences['Match']))* 100,
                         differences['FN'] / (differences['FN'] + len(differences['Match']))*100],color = 'k')
    ax[0].set_ylabel('Probability (%)')
    ax[0].set_title('Missed Onsets')
    if len(differences['Match']) > 0:
        ax[1].hist(np.array(differences['Match']), color = 'k', bins = bins)
        ax[1].set_ylabel('Probability of Delay Occuring')
        ax[1].set_title('Histogram of Delays')
    print('Try to add argdicts to this so that the content is detailed')
    return fig,ax
    
    
    
    