# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:38:20 2018
dimension reduction with VarianceThreshold using sklearn.
Feature selector that removes all low-variance features.
@author: lenovo
"""
from sklearn.feature_selection import VarianceThreshold
import numpy as np
#
np.random.seed(1)
X = np.random.randn(100, 10)
X = np.hstack([X, np.zeros([100, 5])])
#


def featureSelection_variance(X, thrd):
    sel = VarianceThreshold(threshold=thrd)
    X_selected = sel.fit_transform(X)
    mask = sel.get_support()
    return X_selected, mask


X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
selector = VarianceThreshold()
selector.fit_transform(X)
selector.variances_
