# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:15:59 2019

@author: lenovo
"""
from lc_featureSelection_rfe import rfeCV
from lc_read_nii import read_sigleNii_LC
from lc_read_nii import main
import numpy as np
from sklearn import datasets

import sys
sys.path.append(r'F:\黎超\dynamicFC\Code\lc_rsfmri_tools_python-master\Utils')
sys.path.append(
    r'F:\黎超\dynamicFC\Code\lc_rsfmri_tools_python-master\Machine_learning\classfication')


class SVCDeterministicTrAndTe():
    def init(self):
        # ==============================================================================
        # input begin
        patients_path = r'K:\XiaoweiJiang\KETI\3_新建文件夹（20190308）\DATA\REST\ALFF\ALFF_BD'
        hc_path = r'K:\XiaoweiJiang\KETI\3_新建文件夹（20190308）\DATA\REST\ALFF\ALFF_HC1'
        validation_path = r'K:\XiaoweiJiang\KETI\3_新建文件夹（20190308）\DATA\REST\ALFF\ALFF_HC1'
        mask = r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'
        kfold = 5
        # ==============================================================================

    # mask
    mask = read_sigleNii_LC(mask) >= 0.2
    mask = np.array(mask).reshape(-1,)

    # is_train
    if_training = 1

    # input end

    def load_nii_and_gen_label(patients_path, hc_path, mask):
        # train data
        data1 = main(patients_path)
        data1 = np.squeeze(
            np.array([np.array(data1).reshape(1, -1) for data1 in data1]))
        data2 = main(hc_path)
        data2 = np.squeeze(
            np.array([np.array(data2).reshape(1, -1) for data2 in data2]))
        data = np.vstack([data1, data2])

        # validation data
        data_val = main(validation_path)
        data_val = np.squeeze(
            np.array([np.array(data_val).reshape(1, -1) for data_val in data_val]))

        # data in mask
        data_tr = data[:, mask]
        data_val = data_val[:, mask]

        # label_tr
        label_tr = np.hstack(
            [np.ones([len(data1), ]) - 1, np.ones([len(data2), ])])
        return label_tr, data_tr, data_val

    # training and test
    def tr_te():
        import lc_svc_rfe_cv_V2 as lsvc
        svc = lsvc.SVCRefCv(
            pca_n_component=0.9,
            show_results=1,
            show_roc=0,
            k=kfold)
        if if_training:
            results = svc.svc_rfe_cv(data_tr, label_tr)
        return results

    # run
    data_tr, label_tr = datasets.make_classification(n_samples=200, n_classes=2,
                                                     n_informative=50, n_redundant=3,
                                                     n_features=100, random_state=1)

    label_tr, data_tr, data_val = load_nii_and_gen_label(
        patients_path, hc_path, mask)

    selector, weight = rfeCV(data_tr, label_tr, step=0.1, cv=kfold, n_jobs=1,
                             permutation=0)

    results = tr_te()
    results = results.__dict__
    y_pred = selector.predict(x)
