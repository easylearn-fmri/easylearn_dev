# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:07:49 2018

@author: lenovo
"""
from lc_svc_oneVsRest import oneVsRest
import numpy as np
import pandas as pd
from lc_read_write_Mat import read_mat
import sys
sys.path.append(r'D:\myCodes\LC_MVPA\Python\MVPA_Python\utils')
sys.path.append(r'D:\myCodes\LC_MVPA\Python\MVPA_Python\classfication')
# X
fileName = r'J:\分类测试_20180828\Ne-L_VS_Ne-R_n=709'
dataset_name = 'coef'
dataset_struct, dataset = read_mat(fileName, dataset_name)
X = dataset
X = pd.DataFrame(X)
# y
s = pd.read_excel(r'J:\分类测试_20180828\机器学习-ID.xlsx')
dgns = s['诊断'].values
# comb
xandy = pd.concat([pd.DataFrame(dgns), X], axis=1)
# NaN
xandy = xandy.dropna()
#
X = xandy.iloc[:, 1:].values
y = xandy.iloc[:, 0].values
X = np.reshape(X, [len(X), X.shape[1]])
y = [int(d) for d in y]
# predict and test
comp = [prd - y][0]
Acc = np.sum(comp == 0) / len(comp)
