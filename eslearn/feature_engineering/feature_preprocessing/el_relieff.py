# -*- coding: utf-8 -*-
"""
Created on 2020/03/14
Feature selection: Relief-based feature selection algorithm.
feature_train, label_train, feature_test, n_features_to_select
------
@author: LI Chao
"""
import numpy as np
from skrebate import ReliefF

def relief(feature_train, label_train, feature_test, n_features_to_select=None):
    """
    This functio is used to select the features using relief-based feature selection algorithms.
    Parameters
    ----------
        feature_train: numpy.array
            features of the training set. Dimension is 


        label_train: numpy.array
            labels of the training set

        feature_test: numpy.array
            features of the test set

        n_features_to_select: numpy.array
            Path to save results

    Returns
    -------
        Save all classification results and figures to local disk.
    """

    [n_sub, n_features] = np.shape(feature_train)
    if n_features_to_select is None: 
        n_features_to_select = np.int(np.round(n_features / 10))
        
    if isinstance(n_features_to_select, np.float): 
        n_features_to_select = np.int(np.round(n_features * n_features_to_select))
    
    fs = ReliefF(n_features_to_select=n_features_to_select, 
                 n_neighbors=100, discrete_threshold=10, verbose=True, n_jobs=-1)
    fs.fit(feature_train, label_train)
    feature_train = fs.transform(feature_train)
    feature_test = fs.transform(feature_test)
    mask_selected = fs.top_features_[:n_features_to_select]
    return feature_train, feature_test, mask_selected, n_features
