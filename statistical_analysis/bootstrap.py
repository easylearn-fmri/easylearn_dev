# -*- coding: utf-8 -*-
"""

Created on Sat Oct 24 10:21:10 2020

@author: lenovo
"""

import numpy as np


def average(data):
    return sum(data) / len(data)


def bootstrap(data, N, CI, func):
    """
    Get CI using bootstrap

    Parameters:
    ----------
    data: ndarray 
    N: Sampling times, usually, N>=1000
    CI: Confidence interval
    func: function used for generating statiscics
    
    Returns:
    -------
    lower_limit: float
    higher_limit: float
    """
    
    array = np.array(data)
    n = len(array)
    sample_result_arr = []
    for i in range(N):
        index_arr = np.random.randint(0, n, size=n)
        data_sample = array[index_arr]
        sample_result = func(data_sample)
        sample_result_arr.append(sample_result)

    a = 1 - CI
    k1 = int(N * a / 2)
    k2 = int(N * (1 - a / 2))
    auc_sample_arr_sorted = sorted(sample_result_arr)
    lower_limit = auc_sample_arr_sorted[k1]
    higher_limit = auc_sample_arr_sorted[k2]

    return lower_limit, higher_limit


if __name__ == '__main__':
    from kappa import quadratic_weighted_kappa
    import pandas as pd
    file = r'D:\workstation_b\limengsi\加权Kappa.xlsx'
    data = pd.read_excel(file, sheet_name="2D")
    result = bootstrap(data, 1000, 0.95, quadratic_weighted_kappa)
    print(result)