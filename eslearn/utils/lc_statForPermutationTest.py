# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:29:20 2018
statistical analysis for permutation test
@author: lenovo
"""
from lc_selectFile_permSVC import selectFile
from lc_read_write_Mat import read_mat
import pandas as pd
#


def stat():
    # statistic
    pass


def resultFusion(rootPath=r'D:\myCodes\LC_MVPA\Python\MVPA_Python\perm',
                 datasetName=['predict', 'dec', 'y_sorted', 'weight']):
    # Fusion of all block results of permutation test
    fileName = selectFile(rootPath)
    dataset = []
    for dsname in datasetName:
        Ds = []
        for flname in fileName:
            _, ds = read_mat(flname, dsname)
            Ds.append(ds)
        dataset.append(Ds)
    all_metrics = pd.DataFrame(dataset)
    all_metrics = all_metrics.rename(
        index={
            0: 'predict',
            1: 'dec',
            2: 'y_sorted',
            3: 'weight'})
    y_true = all_metrics.loc['y_sorted'][0]
    y_pred = all_metrics.loc['predict'][0]
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    accuracy_score(y_true.T, y_pred.T)
    confusion_matrix(y_true.T, y_pred.T)
    pr[1]
    return dataset
