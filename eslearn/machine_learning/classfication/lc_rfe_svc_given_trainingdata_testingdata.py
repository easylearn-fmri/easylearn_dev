# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:25:57 2019
@author: LI Chao
"""
import sys
import numpy as np

from eslearn.utils.lc_niiProcessor import NiiProcessor
from lc_svc_rfe_cv_V2 import SVCRefCv
from eslearn.utils.lc_evaluation_model_performances import eval_performance

class SvcForGivenTrAndTe(SVCRefCv):
    """
    Training model on given training data.
    Then apply this mode to another testing data.
    Last, evaluate the performance
    If you encounter any problem, please contact lichao19870617@gmail.com
    """
    def __init__(self,
                 # =====================================================================
                 # all inputs are follows
                 patients_path=r'D:\workstation_b\xiaowei\ToLC\training\BD_label1',  # 训练组病人
                 hc_path=r'D:\workstation_b\xiaowei\ToLC\training\MDD__label0',  # 训练组正常人
                 val_path=r'D:\workstation_b\xiaowei\ToLC\testing',  # 验证集数据
                 val_label=r'D:\workstation_b\xiaowei\ToLC\testing_label.txt',  # 验证数据的label文件
                 suffix='.nii',  #图像文件的后缀
                 mask=r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii',
                 k=5  # 训练集内部进行RFE时，用的kfold CV
                 # =====================================================================
                 ):
        
        super().__init__()
        self.patients_path=patients_path
        self.hc_path=hc_path
        self.val_path=val_path
        self.val_label=val_label
        self.suffix=suffix
        self.mask=mask
        self.k=k
        print("SvcForGivenTrAndTe initiated")
        
    def _load_data_infolder(self):
        """load training data and validation data and generate label for training data"""
        print("loading...")
        # train data
        data1, _ = NiiProcessor().read_multi_nii(self.patients_path, self.suffix)
        data1 = np.squeeze(np.array([np.array(data1).reshape(1,-1) for data1 in data1]))
        data2,_ = NiiProcessor().read_multi_nii(self.hc_path, self.suffix)
        data2 = np.squeeze(np.array([np.array(data2).reshape(1,-1) for data2 in data2]))
        data = np.vstack([data1,data2])
        
        # validation data
        data_validation,self.name_val=NiiProcessor().read_multi_nii(self.val_path, self.suffix)
        data_validation=np.squeeze(np.array([np.array(data_validation).reshape(1,-1) for data_validation in data_validation]))
        
        # data in mask
        mask, _ = NiiProcessor().read_sigle_nii(self.mask)
        mask=mask>=0.2
        mask=np.array(mask).reshape(-1,)
        
        self.data_train=data[:,mask]
        self.data_validation=data_validation[:,mask]
        
        # label_tr
        self.label_tr=np.hstack([np.ones([len(data1),]),np.ones([len(data2),]) - 1])
        print("loaded")
        return self

    def tr_te_ev(self):
        """
        训练，测试，评估
        """
        
        # scale
        data_train,data_validation=self.scaler(self.data_train,self.data_validation,self.scale_method)
        
        # reduce dim
        if 0<self.pca_n_component<1:
            data_train,data_validation,trained_pca=self.dimReduction(data_train,data_validation,self.pca_n_component)
        else:
            pass
        
        # training
        print("training...\nYou need to wait for a while")
        model,weight=self.training(data_train,self.label_tr,\
                 step=self.step, cv=self.k,n_jobs=self.num_jobs,\
                 permutation=self.permutation)

        # fetch orignal weight
        if 0 < self.pca_n_component< 1:
            weight=trained_pca.inverse_transform(weight)
        self.weight_all=weight
        
        # testing
        print("testing...")
        self.predict,self.decision=self.testing(model,data_validation)

        # eval performances
        self.val_label=np.loadtxt(self.val_label)
        acc, sens, spec, auc = eval_performance(
            self.val_label,self.predict,self.decision, 
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
            verbose=1, is_showfig=False,
        )

    
    def main(self):
        self._load_data_infolder()
        self.tr_te_ev()
    
if __name__=="__main__":
    svc=SvcForGivenTrAndTe()
    svc.main()
    print("Done!\n")