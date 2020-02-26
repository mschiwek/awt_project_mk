import tensorflow as tf
import numpy as np
import pandas as pd

from glob import glob

def s_naive_error(x, freq= 24):
    n = x.shape[0]
    x1 = x[:n-freq]
    x2 = x[freq:]
    
    return 1/(n - freq) * np.abs(x1 - x2).sum()

def m4_mase(in_sample, yTrue, yPred):
    naive_err = s_naive_error(in_sample)

    err = np.abs(yTrue - yPred) / naive_err
    
    return np.mean(err)


def coverage(yTrue, yLower, yUpper):
    covered_from_lower = yTrue >= yLower
    covered_from_upper = yTrue <= yUpper
    covered = covered_from_lower & covered_from_upper

    return covered.sum()

def acd(intervals_coverage, data_points_number):

    return abs( intervals_coverage.sum() / data_points_number - 0.95)
    
def msis(insample, yTrue, yLower, yUpper):
    
    ts_naive_err = s_naive_error(insample)
    
    coff = 40 # 2/0.05
    
    total_penalty = []
    
    interval_width = yUpper - yLower
        
    missed_lower_indx = yTrue < yLower
    missed_lower_penalty = coff*(yLower[missed_lower_indx] - yTrue[missed_lower_indx])
        
    missed_upper_indx = yTrue > yUpper
    missed_upper_penalty = coff*(yTrue[missed_upper_indx] - yUpper[missed_upper_indx])
        
    penalty = interval_width.sum() + missed_lower_penalty.sum() + missed_upper_penalty.sum()

    penalty = penalty / (yTrue.shape[0] * ts_naive_err)
    
    return penalty.mean()
    
def sMAPE(predY, trueY):
    predY = predY.flatten()
    trueY = trueY.flatten()
    sumf = np.sum(np.abs(predY - trueY) / (np.abs(predY) + np.abs(trueY)))
    return (200 *sumf / len(trueY))
