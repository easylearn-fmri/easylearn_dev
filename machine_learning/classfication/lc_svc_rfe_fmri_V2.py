# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:07:21 2019
@author: lenovo
"""
import sys
import os
cpwd = __file__
root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# root = r'D:/My_Codes/LC_Machine_Learning/lc_rsfmri_tools/lc_rsfmri_tools_python'
sys.path.append(root)
# sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Machine_learning\classfication')

import nibabel as nib
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn.utils as ut

from Utils.lc_niiProcessor import NiiProcessor
from Machine_learning.classfication import lc_svc_rfe_cv_V3 as lsvc


class SvcRfeFmri():
    """
    Input is fmri image, such as .nii or .img
    """
    def __init__(sel):
        # =========================================================
        sel.patients_path = r'D:\workstation_b\豪哥\results\Patients'
        sel.hc_path = r'D:\workstation_b\豪哥\results\HC'
        sel.suffix = '.nii'
        sel.mask = r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'
        sel.out_path = r'D:\workstation_b\豪哥\results'  # 结果保存路径
        # =========================================================
        
        sel.is_save_resutls = 1  # save all results
        sel.is_save_weight_to_nii = 1
        sel.k = 5  # outer k-fold
        sel.pca_n_component= 0.8
        sel.show_results = 1
        sel.show_roc = 1
        sel.is_train = 1
        # Mask
        sel.mask, sel.mask_obj = NiiProcessor().read_sigle_nii(sel.mask)
        sel.orig_shape = sel.mask.shape
        sel.mask = sel.mask >= 0.2
        sel.mask = np.array(sel.mask).reshape(-1,)

    def load_nii_and_gen_label(sel):
        """
        Load nii and generate label
        """
        data1, _ = NiiProcessor().main(sel.patients_path, sel.suffix)
        data1 = np.squeeze(
            np.array([np.array(data1).reshape(1, -1) for data1 in data1]))

        data2, _ = NiiProcessor().main(sel.hc_path, sel.suffix)
        data2 = np.squeeze(
            np.array([np.array(data2).reshape(1, -1) for data2 in data2]))

        sel.data = np.vstack([data1, data2])

        # data in sel.mask
        sel.data_in_mask = sel.data[:, sel.mask]
        # label
        sel.label = np.hstack(
            [np.ones([len(data1), ]), np.ones([len(data2), ])-1])
        return sel

    def tr_te(sel):
        """ 
        Training and test
        """
        svc = lsvc.SVCRfeCv(
                            outer_k=sel.k,
                            pca_n_component=sel.pca_n_component,
                            show_results=sel.show_results,
                            show_roc=sel.show_roc)
        if sel.is_train:
#            sel.label = ut.shuffle(sel.label)
            sel.results = svc.svc_rfe_cv(sel.data_in_mask, sel.label)
        return sel

    def weight2nii(sel, results):
        """
        Transfer weight matrix to nii file
        I used the mask file as reference to generate the nii file
        """
        weight = np.squeeze(results['weight_all'])
        weight_mean = np.mean(weight, axis=0)

        # to orignal space
        weight_mean_orig = np.zeros(sel.orig_shape)
        mask_orig = np.reshape(sel.mask, sel.orig_shape)
        weight_mean_orig[mask_orig] = weight_mean
        # save to nii
        weight_nii = nib.Nifti1Image(weight_mean_orig, affine=sel.mask_obj.affine)
        weight_nii.to_filename(os.path.join(sel.out_path, 'weight.nii'))

    def save_results(sel):
        import time
        now = time.strftime("%Y%m%d%H%M%S", time.localtime())
        with open(os.path.join(sel.out_path, "".join(["results_", now, "_.pkl"])), "wb") as file:
            pickle.dump(sel.results.__dict__, file, True)

#        # load pkl file
#        with open("".join(["results_",now,"_.pkl"]),"rb") as file:
#            results = pickle.load(file)

    def run(sel):
        """run"""
        sel.load_nii_and_gen_label()
        sel.tr_te()
        results = sel.results.__dict__

        # save all results
        if sel.is_save_resutls:
            sel.save_results()

        # save weight
        if sel.is_save_weight_to_nii:
            sel.weight2nii(results)

        return results


if __name__ == "__main__":
    sel = SvcRfeFmri()
    results = sel.run()
    print(results.keys())
    print(results.items())
