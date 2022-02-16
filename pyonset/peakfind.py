import numpy as np
import scipy.signal as sig
from .utils import p2a

def online_peaks(y, lag, threshold, influence, delay_min = 0, min_proba = 0):
    countdown = 0
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y) - 1):
        countdown -= 1
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1] and countdown <= 0 and y[i]>min_proba:
                signals[i] = 1
                countdown = delay_min
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def online_batch_peak(proba_function_batch, lag, threshold, influence, delay_min = 0, min_proba = 0):
    batch_peak = np.zeros(proba_function_batch.shape)
    for i in range(batch_peak.shape[0]):
        batch_peak[i] = online_peaks(proba_function_batch[i], lag, threshold, influence, delay_min = delay_min, min_proba = min_proba)['signals']
    return batch_peak


def offline_peaks(proba_function,height = 0.1, threshold = None, distance = None, 
                  prominence = None, rel_height = None):
    detected = sig.find_peaks(proba_function,height = height, threshold = threshold, distance = distance, prominence = prominence, rel_height = rel_height)[0]
    detected = p2a(detected, length=len(proba_function))
    return detected

def offline_batch_peak(proba_function_batch,height = 0.1, threshold = None, distance = None, 
                  prominence = None, rel_height = None):
    batch_peak = np.zeros(proba_function_batch.shape)
    for i in range(batch_peak.shape[0]):
        batch_peak[i] = offline_peaks(proba_function_batch[i],height = 0.1, threshold = threshold, distance = distance, 
                          prominence = prominence, rel_height = rel_height)
    return batch_peak