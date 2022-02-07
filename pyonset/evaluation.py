import numpy as np


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