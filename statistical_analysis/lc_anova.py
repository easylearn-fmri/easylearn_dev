# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:40:45 2018
方差分析
@author: lenovo
"""
import scipy.stats as stats

def oneway_anova(data, *args, **kwargs):
    """
    Data is a list, 
    in which each element is a np.array matrix
    (such as N*M, N=sample size, M=number of variable)
    """
    f, p = stats.f_oneway(data, *args, **kwargs)
    return f, p

if __name__ == '__main__':
    import numpy as np
    data = []