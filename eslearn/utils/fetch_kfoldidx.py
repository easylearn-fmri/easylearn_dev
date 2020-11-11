# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:48:39 2019

@author: lenovo
"""
import numpy as np
from sklearn.model_selection import KFold

def fetch_kFold_Index_for_allLabel(x, y, outer_k, seed):
    """分别从每个label对应的数据中，进行kFole选择，
    然后把某个fold的数据组合成一个大的fold数据
    """
    uni_y = np.unique(y)
    loc_uni_y = [np.argwhere(y == uni) for uni in uni_y]

    train_index, test_index = [], []
    for y_ in loc_uni_y:
        tr_index, te_index = fetch_kfold_idx_for_onelabel(y_, outer_k, seed)
        train_index.append(tr_index)
        test_index.append(te_index)

    indexTr_fold = []
    indexTe_fold = []
    for k_ in range(outer_k):
        indTr_fold = np.array([])
        indTe_fold = np.array([])
        for y_ in range(len(uni_y)):
            indTr_fold = np.append(indTr_fold, train_index[y_][k_])
            indTe_fold = np.append(indTe_fold, test_index[y_][k_])
        indexTr_fold.append(indTr_fold)
        indexTe_fold.append(indTe_fold)
    index_train, index_test = [], []
    for I in indexTr_fold:
        index_train.append([int(i) for i in I])
    for I in indexTe_fold:
        index_test.append([int(i) for i in I])

    return index_train, index_test


def fetch_kfold_idx_for_onelabel(originLable, outer_k, seed):
    """获得对某一个类的数据的kfold index"""
    np.random.seed(seed)
    kf = KFold(n_splits=outer_k)
    train_index, test_index = [], []
    for tr_index, te_index in kf.split(originLable):
        train_index.append(originLable[tr_index]), \
            test_index.append(originLable[te_index])
    return train_index, test_index

def fetch_kfold_idx_for_alllabel_LOOCV(y):
    """generate index for leave one out cross validation"""
    index_test = list(np.arange(0, len(y), 1))
    index_train = [list(set(np.arange(0, len(y), 1)) - set([i]))
                   for i in np.arange(0, len(y), 1)]
    return index_train, index_test