# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:25:57 2019
@author: LI Chao
"""

import numpy as np
import nibabel as nib
import os
import time

from eslearn.model_evaluator import ModelEvaluator
from eslearn.utils.lc_niiProcessor import NiiProcessor
from eslearn.base import BaseMachineLearning
from eslearn.machine_learning.classfication._base_classificaition import BaseClassification, PipelineSearch_


class ClassificationXiaowei(BaseMachineLearning, PipelineSearch_):
    """
    Training model on given training data.
    Then apply this mode to another testing data.
    Last, evaluate the performance
    If you encounter any problem, please contact lichao19870617@gmail.com
    """
    def __init__(self,
                 # =====================================================================
                 patients_path=r'D:\workstation_b\xiaowei\TOLC_20200811\training\BD_label1',  # 训练组病人
                 hc_path=r'D:\workstation_b\xiaowei\TOLC_20200811\training\MDD_label0',  # 训练组正常人
                 val_path=r'D:\workstation_b\xiaowei\TOLC_20200811\mix6',  # 验证集数据
                 val_label=r'D:\workstation_b\xiaowei\TOLC_20200811\mix6_label.txt',  # 验证数据的label文件
                 suffix='.nii',
                 mask=r'D:\workstation_b\xiaowei\TOLC3\dFC\TRA\MDD\StdzFC_ROI1_01367_resting7000.nii',
                 k=3,
                 save_directory = r'D:\workstation_b\xiaowei\TOLC_20200811'
                 # =====================================================================
                 ):
        
        super(BaseMachineLearning, self).__init__()
        super(PipelineSearch_, self).__init__()
        self.search_strategy = 'grid'
        self.n_jobs = 2
        self.k=k
        self.verbose=False
        
        self.patients_path=patients_path
        self.hc_path=hc_path
        self.val_path=val_path
        self.val_label=val_label
        self.suffix=suffix
        self.mask=mask
        self.save_directory = save_directory
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
        data_validation, self.name_val = NiiProcessor().read_multi_nii(self.val_path, self.suffix)
        data_validation = np.squeeze(np.array([np.array(data_validation).reshape(1,-1) for data_validation in data_validation]))
        
        # data in mask
        mask, self.mask_obj =  NiiProcessor().read_sigle_nii(self.mask)
        self.mask_orig = mask>=0.1
        self.mask_1d = np.array(self.mask_orig).reshape(-1,)
        
        self.data_train = data[:,self.mask_1d]
        self.data_validation = data_validation[:,self.mask_1d]
        
        # label_tr
        self.label_train=np.hstack([np.ones([len(data1),]),np.ones([len(data2),]) - 1])
        self.label_validation = np.loadtxt(self.val_label)
        print("loaded")
        return self

    def pipeline_grid(self, 
                       method_feature_preprocessing=None, 
                       param_feature_preprocessing=None,
                       method_dim_reduction=None,
                       param_dim_reduction=None,
                       method_feature_selection=None,
                       param_feature_selection=None,
                       method_machine_learning=None,
                       param_machine_learning=None
    ):

        self.make_pipeline_(
            method_feature_preprocessing=method_feature_preprocessing, 
            param_feature_preprocessing=param_feature_preprocessing, 
            method_dim_reduction=method_dim_reduction, 
            param_dim_reduction=param_dim_reduction, 
            method_feature_selection=method_feature_selection,
            param_feature_selection=param_feature_selection,
            method_machine_learning=method_machine_learning, 
            param_machine_learning=param_machine_learning
        )

        print(self.param_search_)
        # Train
        self.fit_pipeline_(self.data_train, self.label_train)
        
        # Get weights
        self.get_weights_(self.data_train, self.label_train)
        self._weight2nii(self.weights_)
        
        # Predict
        pred_train, dec_train = self.predict(self.data_train)
        self.predict_validation, self.decision = self.predict(self.data_validation)
        
        # Eval performances
        print("Evaluating training data...")
        bi_evaluator = ModelEvaluator().binary_evaluator
        acc, sens, spec, auc = bi_evaluator(
            self.label_train,pred_train,dec_train, 
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
            verbose=1, is_showfig=False,is_savefig=True, out_name=os.path.join(self.save_directory, "performances_train.pdf")
        )
        
        print("Evaluating test data...")
        self.val_label=np.loadtxt(self.val_label)
        acc, sens, spec, auc = bi_evaluator(
            self.val_label, self.predict_validation ,self.decision, 
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
            verbose=1, is_showfig=False, is_savefig=True, out_name=os.path.join(self.save_directory, "performances_test.pdf")
        )
    
    def _weight2nii(self, weights, dimension_nii_data=(61, 73, 61)):
        """
        Transfer weight matrix to nii file
        I used the mask file as reference to generate the nii file
        """

        weight = np.squeeze(weights)
        # weight_mean = np.mean(weight, axis=0)

        # to orignal space
        weight_mean_orig = np.zeros(dimension_nii_data)
        weight_mean_orig[self.mask_orig] = weight
        # save to nii
        weight_nii = nib.Nifti1Image(weight_mean_orig, affine=self.mask_obj.affine)
        weight_nii.to_filename(os.path.join(self.save_directory, "weight.nii"))

    
if __name__=="__main__":
    time_start = time.time()
    clf = ClassificationXiaowei()
    clf.get_configuration_(configuration_file=r'D:\workstation_b\xiaowei\TOLC_20200811\configuration_file(200728)(1).json')
    clf.get_preprocessing_parameters()
    clf.get_dimension_reduction_parameters()
    clf.get_feature_selection_parameters()
    clf.get_unbalance_treatment_parameters()
    clf.get_machine_learning_parameters()
    clf.get_model_evaluation_parameters()
    
    method_feature_preprocessing = clf.method_feature_preprocessing
    param_feature_preprocessing= clf.param_feature_preprocessing

    method_dim_reduction = clf.method_dim_reduction
    param_dim_reduction = clf.param_dim_reduction

    method_feature_selection = clf.method_feature_selection
    param_feature_selection = clf.param_feature_selection

    method_machine_learning = clf.method_machine_learning
    param_machine_learning = clf.param_machine_learning
    
    clf._load_data_infolder()
    clf.pipeline_grid(
        method_feature_preprocessing=method_feature_preprocessing, 
        param_feature_preprocessing=param_feature_preprocessing,
        method_dim_reduction=method_dim_reduction,
        param_dim_reduction=param_dim_reduction,
        method_feature_selection=method_feature_selection,
        param_feature_selection=param_feature_selection, 
        method_machine_learning=method_machine_learning, 
        param_machine_learning=param_machine_learning,
    )
    time_end = time.time()
    print(f"Running time = {time_end-time_start}\n")