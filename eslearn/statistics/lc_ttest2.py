# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:55:25 2018
t test
method='independent'
OR
method='related'

@author: lenovo
"""

from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import numpy as np

#
def ttest2(a,b,method='independent'):
    if method=='independent':
        t,p=ttest_ind(a,b,axis=0,nan_policy='omit')
    elif method=='related':
        t,p=ttest_rel(a,b,axis=0,nan_policy='omit')
    else:
        print('Nether independent nor related\n')
    return (t,p)
