# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:48:21 2020

@author: Chao Li
Email: lichao19870617@gmail.com
"""

import numpy.sqrt as npsqrt

def CohenEffectSize(group1=group1, group2=group2):
    """ Calculate Cohen' d 
    Parameters:
    -----------
        group1: NumPy array
            dimension is n_samples * n_features
        group2: NumPy array
            dimension is n_samples * n_features

    Return: float
        Cohen' d 
    """
    
    diff = group1.mean(axis=0) - group2.mean(axis=0)

    n1, n2 = len(group1), len(group2)
    var1 = group1.var(axis=0)
    var2 = group2.var(axis=0)

    pooled_var = ((n1 -1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    d = diff / npsqrt(pooled_var)
    return d