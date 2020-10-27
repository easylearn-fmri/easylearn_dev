# -*- coding: utf-8 -*-
"""
Using bootstrap to get Confidence Interval for kappa
Created on Sat Oct 24 10:21:10 2020

@author: Li Chao
"""

import numpy as np
from kappa import linear_weighted_kappa, quadratic_weighted_kappa


def func(data):
    d1 = data[:,0]
    d2 = data[:,1]
    kappa = quadratic_weighted_kappa(d1, d2)
    return kappa


def bootstrap(data, N, CI, func):
    """
    Get CI using bootstrap

    Parameters:
    ----------
    data: ndarray with dimension n_samples*2
    N: Sampling times, usually, N>=1000
    CI: Confidence interval
    func: function used for generating statiscics
    
    Returns:
    -------
    lower_limit: float
    higher_limit: float
    """
    array_data = np.array(data)
    n = len(array_data)
    sampled_statisic_arr = []
    for i in range(N):
        index = np.random.choice(range(0,n), size=n)
        data_sample = array_data[index,:]
        sampled_statisic = func(data_sample)
        sampled_statisic_arr.append(sampled_statisic)

    a = 1 - CI 
    k1 = int(N * a / 2)
    k2 = int(N * (1 - a / 2))
    auc_sample_arr_sorted = np.sort(sampled_statisic_arr)
    lower_limit = auc_sample_arr_sorted[k1]
    higher_limit = auc_sample_arr_sorted[k2]

    return lower_limit, higher_limit


if __name__ == '__main__':
    from sklearn.metrics import cohen_kappa_score
    import pandas as pd
    file = r'D:\workstation_b\limengsi\加权Kappa.xlsx'
    data = pd.read_excel(file, sheet_name="3D")
    dd=np.vstack([np.ones(10,), np.ones(10,)]).T
    dd[1,1] = 2
    func(dd)
    kappa = func(data.values)
    result = bootstrap(data.values, 1000, 0.95, func)
    print(kappa, result)
    