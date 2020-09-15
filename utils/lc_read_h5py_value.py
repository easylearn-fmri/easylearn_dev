# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:03:45 2018

@author: lenovo
"""
# import
import h5py
import numpy as np
# def


def read_h5py_value(file_name='enet', dataset_name='Predict'):
    f = h5py.File(file_name + ".hdf5", "a")
    value = np.array([])
    for g in f.keys():
        d = f[g]
        value = np.append(value, d[dataset_name])
#        value=np.hstack((value,d[dataset_name]))
    try:
        col_num = d[dataset_name].shape[0]
#        print(col_num)+
    except BaseException:
        col_num = value.size
#        print(col_num)
    value = value.reshape([value.size // col_num, col_num])
    return value
    f.close()
