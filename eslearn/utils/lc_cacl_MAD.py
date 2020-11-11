# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:18:08 2018
MAD,median absolute deviation for dimension reduction
MAD=median(|Xi−median(X)|)
refer to {Linked dimensions of psychopathology
and connectivity in functional brain networks}
@author: Li Chao
"""
import numpy as np


def select_features_using_MAD(M, perc=0.1):
    # perc:  how many percentages of feature
    # that have top MAD to be selected
    MAD = cacl_MAD(M)
    Ind_descendOrd = np.argsort(MAD)[::-1]  # decend order
    Ind_select = Ind_descendOrd[0:int(len(Ind_descendOrd) * perc)]
    feature_selected = M[:, Ind_select]
    return feature_selected


def cacl_MAD(M):
    # caculate MAD
    # row is sample, col is feature
    my_median = np.median(M, 0)
    my_abs = np.abs(M - my_median)
    MAD = np.median(my_abs, 0)
    return MAD
