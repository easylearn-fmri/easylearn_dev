# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:14:01 2019

@author: lenovo
"""

import sys
import os
cpwd = __file__
root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from Utils.lc_read_nii import read_multiNii_LC
from Utils.lc_read_nii import read_sigleNii_LC
from Machine_learning.classfication.lc_svc_rfe_cv_V2 import SVCRefCv


class SvcForGivenTrAndTe(SVCRefCv):
    """
    Training model on given training data_tr.
    Then apply this mode to another testing data_te.
    Last, evaluate the performance
    If you encounter any problem, please contact lichao19870617@gmail.com
    """

    def __init__(self,
                 # =====================================================================
                 # all inputs are follows
                 tr_path=r'D:\folder\file.xlsx',  # tranining  dataset path
                 te_path=r'D:\folder\file.xlsx',  # test dataset path
                 col_name_of_label="label",  # column name of the label
                 col_num_of_data=np.arange(1, 7),  # column number of features
                 inner_k=3  # k: the k-fold cross validation of the inner CV
                 # =====================================================================
                 ):

        super().__init__()
        self.tr_path = tr_path
        self.te_path = te_path
        self.col_name_of_label = col_name_of_label
        self.col_num_of_data = col_num_of_data
        self.inner_k = inner_k

        # Default parameters
        self.pca_n_component = 1  # PCA
        self.verbose = 1  # if print results
        self.show_roc = 1  # if show roc
        self.seed = 100
        self.step = 1
        print("SvcForGivenTrAndTe initiated")

    def _load_data_inexcel(self):
        """load training data_tr/label_tr and validation data_tr"""
        print("loading...")
        data_tr, label_tr = self._load_data_inexcel_forone(
            self.tr_path, self.col_name_of_label, self.col_num_of_data)
        data_te, label_te = self._load_data_inexcel_forone(
            self.te_path, self.col_name_of_label, self.col_num_of_data)
        print("loaded!")
        return data_tr, data_te, label_tr, label_te

    def _load_data_inexcel_forone(self, path, col_name_of_label, col_num_of_data):
        """
        Load training data_tr/label_tr and validation data_tr
        """
        
        data = pd.read_excel(path)
        data = data.dropna()
        label = data[col_name_of_label].values
        # hot encoder
        le = LabelEncoder()
        le.fit(label)
        label = le.transform(label)
        data = data.iloc[:, col_num_of_data].values
        return data, label

    def tr_te_ev(self, data_tr, label_tr, data_te):
        """
        训练，测试，评估
        """
        # scale
        data_tr, data_te = self.scaler(
            data_tr, data_te, self.scale_method)

        # reduce dim
        if 0 < self.pca_n_component < 1:
            data_tr, data_te, trained_pca = self.dimReduction(
                data_tr, data_te, self.pca_n_component)
        else:
            pass

        # training
        print("training...\nYou need to wait for a while")
        model, weight = self.training(data_tr, label_tr,
                                      step=self.step, cv=self.inner_k, n_jobs=self.num_jobs,
                                      permutation=self.permutation)

        # fetch orignal weight
        if 0 < self.pca_n_component < 1:
            weight = trained_pca.inverse_transform(weight)
        self.weight_all = weight
        decision, predict = self.testing(model, data_te)
        return predict, decision

    def eval(self, label_te, predict, decision):
        """
        eval performances
        """
        print('Testing...')
        self.eval_prformance(label_te, predict, decision)
        print('Testing done!')
        return self

    def main(self):
        data_tr, data_te, label_tr, label_te = self._load_data_inexcel()
        self.decision, self.predict = self.tr_te_ev(data_tr, label_tr, data_te)
        self.eval(label_te, self.predict, self.decision)
        return self


if __name__ == "__main__":
    self = SvcForGivenTrAndTe(                 
                 tr_path=r'D:\workstation_b\宝宝\allResampleResult.csv',  # 训练组病人
                 te_path=r'D:\workstation_b\宝宝\allResampleResult.csv',  # 验证集数据
                 col_name_of_label="label",  # label所在列的项目名字
                 col_num_of_data=np.arange(1, 7),  # 特征所在列的序号（第哪几列）
                 inner_k=3)
    
    results = self.main()
    results = results.__dict__
    print("Done!\n")
