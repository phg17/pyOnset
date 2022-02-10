import numpy as np
from .utils import a2p, p2a


def spkd(s1, s2, cost):
    '''
    Import from MKegler: SETCO model.
    Input:
        s1,2: pair of vectors of spike times
        cost: cost parameter for computing Victor-Purpura spike distance.
        (Note: the above need to have the same units!)
    Output:
        d: VP spike distance.
    '''
    nspi=len(s1);
    nspj=len(s2);

    scr=np.zeros((nspi+1, nspj+1));

    scr[:,0]=np.arange(nspi+1)
    scr[0,:]=np.arange(nspj+1)

    for i in np.arange(1,nspi+1):
        for j in np.arange(1,nspj+1):
            scr[i,j]=min([scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1,j-1]+cost*np.abs(s1[i-1]-s2[j-1])]);

    d=scr[nspi,nspj];

    return d


def distance_onsets(predicted_onsets, true_onsets, onsets = True, cost = 1):
    #wrapper to do spkd over a whole batch
    #if arrays are fed, onsets are computed instead
    distance_list = []
    for i_batch in range(predicted_onsets.shape[0]):
        if onsets:
            current_prediction = predicted_onsets[i_batch,:]
            current_truth = true_onsets[i_batch,:]
        else:
            current_prediction = a2p(predicted_onsets[i_batch,:])
            current_truth = a2p(true_onsets[i_batch,:])
        norm_factor = len(current_prediction) + len(current_truth)
        distance_list.append(spkd(current_prediction, current_truth, cost = cost) / norm_factor)
    return distance_list

def differences_timing(s1,s2, cost = 1/12):
    #For two list of peaks and a cost, gives the number of FP, FN and 
    #the distribution of delay when matches are found
    differences = dict()
    differences['Match'] = []
    differences['FP'] = 0
    differences['FN'] = 0
    nspi=len(s1);
    nspj=len(s2);
    
    scr=np.zeros((nspi+2, nspj+2));
    
    scr[:,0]=np.arange(nspi+2)
    scr[0,:]=np.arange(nspj+2)
    scr[:,nspj+1] = np.ones(nspi+2) * 1000
    scr[nspi+1,:] = np.ones(nspj+2) * 1000
    
    for i in np.arange(1,nspi+1):
        for j in np.arange(1,nspj+1):
            scr[i,j]=min([scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1,j-1]+cost*np.abs(s1[i-1]-s2[j-1])])
    i = 0
    j = 0
    while (i < nspi or j < nspj):
      move = min([scr[i+1,j], scr[i,j+1], scr[i+1,j+1]])
      if move == scr[i+1,j]:
        i += 1
        differences['FP'] += 1
      elif move == scr[i,j+1]:
        j += 1
        differences['FN'] += 1
      else:
        differences['Match'].append(s1[i]-s2[j])
        i += 1
        j += 1
    return differences

