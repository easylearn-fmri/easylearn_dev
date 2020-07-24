# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:07:53 2018

@author: lenovo
"""
import numpy as np


def splitX_accord_sorted_y(y, k):
    if y.shape[0] == 1:
        y = y.reshape(y.shape[1], 0)
    ind_y_sorted = np.argsort(y, axis=0)  # ascending
    ind_one_fold = []
    ind_orig = []
    for i in range(k):
        ind_one_fold = (np.arange(i, y.size, k))
        ind_orig.append(ind_y_sorted[ind_one_fold])
    return ind_orig
