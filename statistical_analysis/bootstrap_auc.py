# -*- coding: utf-8 -*-
"""

Created on Sat Oct 24 10:21:10 2020

@author: lenovo
"""

import numpy as np
import pickle
from sklearn.metrics import roc_auc_score


def get_outputs(output1, output2):
    # Load
    model = pickle.load(open(output1, "rb"))
    stat = pickle.load(open(output2, "rb"))

    # Get outputs
    test_targets = model["test_targets"]
    test_probability = model["test_probability"]
    return test_targets, test_probability

def get_auc(data):
    test_targets, test_probability = data[:,0], data[:, 1]
    return roc_auc_score(test_targets, test_probability)


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
    output1 = r"C:\Users\dongm\Desktop\影像组学的demo数据/outputs.pickle"
    output2 = r"C:\Users\dongm\Desktop\影像组学的demo数据/stat.pickle"
    test_targets, test_probability = get_outputs(output1, output2)

    data = np.vstack([test_targets, test_probability]).T
    result = bootstrap(data, 1000, 0.95, get_auc)
    print(result)