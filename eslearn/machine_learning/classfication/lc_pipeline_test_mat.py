# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:25:57 2019
@author: LI Chao
"""

import numpy as np
import nibabel as nib
import scipy.io as sio
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from eslearn.model_evaluator import ModelEvaluator
from eslearn.utils.lc_niiProcessor import NiiProcessor
from eslearn.base import BaseMachineLearning
from eslearn.machine_learning.classfication._base_classificaition import BaseClassification, PipelineSearch_


class ClassificationYueYing(BaseMachineLearning, PipelineSearch_):
    """

    """

    def __init__(self,
                 # =====================================================================
                 # all inputs are follows
                 patients_path=r'D:\悦影科技\数据处理业务1\data_variance_22_30_z\dFCD_var_22\zemci',  # 训练组病人
                 hc_path=r'D:\悦影科技\数据处理业务1\data_variance_22_30_z\dFCD_var_22\znc',  # 训练组正常人
                 suffix='.nii',
                 mask=r'D:\悦影科技\数据处理业务1\data_variance_22_30_z\GreyMask_02_61x73x61.img',
                 k=10,
                 legend1='EMCI', 
                 legend2='LMCI', 
                 performances_save_name=r'D:\悦影科技\数据处理业务1\data_variance_22_30_z\dFCD_var_22\lmciVSemci.pdf',
                 save_wei_name = r'D:\悦影科技\数据处理业务1\data_variance_22_30_z\dFCD_var_22\lmciVSemci.nii'
                 # =====================================================================
                 ):
        
        super(BaseMachineLearning, self).__init__()
        super(PipelineSearch_, self).__init__()
        self.search_strategy = 'grid'
        self.n_jobs = 2
        self.k=k
        self.verbose=True
        
        self.patients_path=patients_path
        self.hc_path=hc_path
        self.suffix=suffix
        self.mask=mask
        self.legend1 = legend1
        self.legend2 = legend2
        self.performances_save_name = performances_save_name
        self.save_wei_name = save_wei_name
        print("SvcForGivenTrAndTe initiated")
        
    def _load_data_infolder(self):
        """load training data and validation data and generate label for training data"""
        print("loading...")
        # train data
        data1 = sio.loadmat(self.patients_path)['data']
        data2 = sio.loadmat(self.hc_path)['data']
        self.data = np.vstack([data1,data2])
        
        # label_tr
        self.label = np.hstack([np.ones([len(data1),]),np.ones([len(data2),]) - 1])
        print("loaded")
        return self

    def loop(self):
        self.get_configuration_(
            configuration_file=r'D:\My_Codes\easylearn\eslearn\GUI\test\configuration_file.json')
        self.get_preprocessing_parameters()
        self.get_dimension_reduction_parameters()
        self.get_feature_selection_parameters()
        self.get_unbalance_treatment_parameters()
        self.get_machine_learning_parameters()
        self.get_model_evaluation_parameters()

        method_feature_preprocessing = self.method_feature_preprocessing
        param_feature_preprocessing = self.param_feature_preprocessing

        method_dim_reduction = self.method_dim_reduction
        param_dim_reduction = self.param_dim_reduction

        method_feature_selection = self.method_feature_selection
        param_feature_selection = self.param_feature_selection

        method_machine_learning = self.method_machine_learning
        param_machine_learning = self.param_machine_learning

        # Load
        self._load_data_infolder()

        # Split data into training and test datasets
        accuracy = []
        sensitivity = []
        specificity = []
        auc = []
        pred_test = []
        decision = []
        weights = []
        label_test_all = []
        cv = StratifiedKFold(n_splits=self.k, random_state=0)
        for train_index, test_index in cv.split(self.data, self.label):
            data_train = self.data[train_index, :]
            data_test = self.data[test_index, :]
            label_train = self.label[train_index]
            label_test = self.label[test_index]
            label_test_all.extend(label_test)

            # Resample
            ros = RandomOverSampler(random_state=0)
            data_train, label_train = ros.fit_resample(data_train, label_train)

            print(f"After re-sampling, the sample size are: {sorted(Counter(label_train).items())}")

            acc, sens, spec, auc_, pred_test_, dec, wei = self.pipeline_grid(
                method_feature_preprocessing=method_feature_preprocessing,
                param_feature_preprocessing=param_feature_preprocessing,
                method_dim_reduction=method_dim_reduction,
                param_dim_reduction=param_dim_reduction,
                method_feature_selection=method_feature_selection,
                param_feature_selection=param_feature_selection,
                method_machine_learning=method_machine_learning,
                param_machine_learning=param_machine_learning,
                data_train=data_train, data_test=data_test, label_train=label_train, label_test=label_test
            )

            accuracy.append(acc)
            sensitivity.append(sens)
            specificity.append(spec)
            auc.append(auc_)
            pred_test.extend(pred_test_)
            decision.extend(dec)
            weights.append(wei)
            
        # Eval performances
        evaluator = ModelEvaluator()
        acc, sens, spec, auc_ = evaluator.binary_evaluator(
            label_test_all, pred_test, decision,
            accuracy_kfold=accuracy, sensitivity_kfold=sensitivity, specificity_kfold=specificity, AUC_kfold=auc,
            verbose=1, is_showfig=True, legend1=self.legend1, legend2=self.legend2, is_savefig=False, out_name=self.performances_save_name
        )

        # save weight to nii
        # self._weight2nii(weights)
        return accuracy, sensitivity, specificity, auc, weights

    def pipeline_grid(self, 
                       method_feature_preprocessing=None, 
                       param_feature_preprocessing=None,
                       method_dim_reduction=None,
                       param_dim_reduction=None,
                       method_feature_selection=None,
                       param_feature_selection=None,
                       method_machine_learning=None,
                       param_machine_learning=None,
                       data_train=None, data_test=None, label_train=None, label_test=None
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
        self.fit_pipeline_(data_train, label_train)
        
        # Get weights
        self.get_weights_(data_train, label_train)
        
        # Predict
        pred_test, dec_test = self.predict(data_test)
        
        # Eval performances
        evaluator = ModelEvaluator()
        acc, sens, spec, auc = evaluator.binary_evaluator(
            label_test, pred_test, dec_test,
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
            verbose=1, is_showfig=False,
        )
        return acc, sens, spec, auc, pred_test, dec_test, self.weights_


    def _weight2nii(self, weights, dimension_nii_data=(61, 73, 61)):
        """
        Transfer weight matrix to nii file
        I used the mask file as reference to generate the nii file
        """

        weight = np.squeeze(weights)
        weight_mean = np.mean(weight, axis=0)

        # to orignal space
        weight_mean_orig = np.zeros(dimension_nii_data)
        weight_mean_orig[self.mask_orig] = weight_mean
        # save to nii
        weight_nii = nib.Nifti1Image(weight_mean_orig, affine=self.mask_obj.affine)
        weight_nii.to_filename(self.save_wei_name)
        

if __name__=="__main__":
    time_start = time.time()
    clf = ClassificationYueYing()
    accuracy, sensitivity, specificity, auc, weights = clf.loop()
    print(np.mean(accuracy), np.mean(sensitivity), np.mean(specificity), np.mean(auc))
    print(np.std(accuracy)/np.power(len(accuracy), 0.5), 
        np.std(sensitivity)/np.power(len(sensitivity), 0.5), 
        np.std(specificity)/np.power(len(specificity), 0.5), 
        np.std(auc)/np.power(len(auc), 0.5))
    time_end = time.time()
    print(f"Running time = {time_end-time_start}\n")

    # best_model = clf.model_.best_estimator_
    # feature_selection =  best_model.get_params().get('feature_selection', None)
    