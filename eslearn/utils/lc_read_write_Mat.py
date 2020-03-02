# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:42:13 2018
read and write .mat(MATLAB) data
@author: Li Chao
"""
# import module
from scipy import io
import numpy as np

# def


def read_mat(fileName, dataset_name=None):
    dataset_struct = io.loadmat(fileName)
    if dataset_name:
        dataset = dataset_struct[dataset_name]
    else:
        dataset = dataset_struct[list(dataset_struct.keys())[3]]
    return dataset

#


def write_mat(fileName='lc_test.mat', dataset_name=['data1', 'data2'],
              dataset=[np.ones([10, 7]), np.ones([10, 8])]):

    cmdStr = str()

    if len(dataset_name) == 1 or isinstance(dataset_name, str):
        cmdStr = cmdStr + 'dataset_name' + ':' + 'dataset' + ','
    else:
        for i in range(len(dataset_name)):
            cmdStr = cmdStr + \
                'dataset_name[' + str(i) + ']' + ':' + 'dataset' + '[' + str(i) + ']' + ','

    cmdStr = cmdStr[:-1]
    cmdStr = 'io.savemat(' + 'fileName' + ',' + '{' + cmdStr + '}' + ')'
    eval(cmdStr)


if __name__ == '__main__':
    fileName_R = r'J:\Research_2017go\GCA+Degree\GCA\Frontiers2018\NewIdea_201708\投稿\Frontier in Neurology\时域BOLD信号\Signals_R62\_signal.mat'
    fileName_L = r'J:\Research_2017go\GCA+Degree\GCA\Frontiers2018\NewIdea_201708\投稿\Frontier in Neurology\时域BOLD信号\Signals_R63\signalAllSubj.mat'

    dataset_struct1, dataset1 = read_mat(fileName_R, 'Signal')
    dataset_struct2, dataset2 = read_mat(fileName_L, 'Signal')
