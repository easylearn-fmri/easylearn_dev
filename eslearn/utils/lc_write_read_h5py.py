# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:29:22 2018
write and read h5py file
@author: lenovo
"""
# import
import h5py
import numpy as np

# def
# r  只读，文件必须已存在
# r+ 读写，文件必须已存在
# w  新建文件，若存在覆盖
# w- 或x，新建文件，若存在报错
# a  如存在则读写，不存在则创建(默认)


def write_h5py(fileName, group_name, dataset_name, dataset):
    f = h5py.File(fileName + ".hdf5", "a")
    g = f.create_group(group_name)
    for i in range(len(dataset)):
        g.create_dataset(dataset_name[i], data=dataset[i])
    f.close()
#


def read_h5py(fileName='aa'):
    f = h5py.File(fileName + ".hdf5", "a")
    print('group are:\n')
    for g in f.keys():
        print(g)
    d = f[g]
    print('group structure is:\n{}'.format([key for key in d.keys()]))
    one_value = np.array([])
    one_value = np.append(one_value, [value for value in d.values()])
    return one_value
    f.close()
