# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:03:49 2018
删除scale中敏感信息
@author: lenovo
"""
import pandas as pd
import numpy as np

d = pd.read_excel(
    r'D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Scales\修改表.xlsx')
col = list(d.columns)


del_ind = set(np.array([1, 2, 3, 25, 26, 36, 37, 50, 87, 88]))
all_ind = set(np.arange(0, len(col), 1))

rest_ind = all_ind - del_ind


dd = d.iloc[:, list(rest_ind)]

col_rest = list(dd.columns)

dd.to_excel('no_sensitive_scale.xlsx', index=False)
