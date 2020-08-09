# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:25:57 2019
@author: LI Chao
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split

from eslearn.model_evaluation.el_evaluation_model_performances import eval_performance
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
                 # all inputs are follows
                 patients_path=r'D:\悦影科技\数据处理业务1\data_variance_22_30\dFCD_var_22\AD',  # 训练组病人
                 hc_path=r'D:\悦影科技\数据处理业务1\data_variance_22_30\dFCD_var_22\NC',  # 训练组正常人
                 suffix='.nii',
                 mask=r'D:\悦影科技\数据处理业务1\data_variance_22_30\GreyMask_02_61x73x61.img',
                 k=3
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

        # data in mask
        mask, _ = NiiProcessor().read_sigle_nii(self.mask)
        self.mask_orig = mask>=0.1
        self.mask_1d = np.array(self.mask_orig).reshape(-1,)
        
        self.data = data[:,self.mask_1d]
        
        # label_tr
        self.label = np.hstack([np.ones([len(data1),]),np.ones([len(data2),]) - 1])

        # Split data into training and test datasets
        self.data_train, self.data_test, self.label_train, self.label_test = train_test_split(self.data, self.label, test_size=0.4, random_state=42, shuffle=True)
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
        
        # Predict
        pred_test, dec_test = self.predict(self.data_test)
        
         # Eval performances
        acc, sens, spec, auc = eval_performance(
            self.label_test, pred_test, dec_test,
            accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
            verbose=1, is_showfig=True,
        )


if __name__=="__main__":
    time_start = time.time()
    clf = ClassificationXiaowei()
    clf.get_configuration_(configuration_file=r'D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\GUI\test\configuration_file.json')
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
    
    
    best_model = clf.model_.best_estimator_

    feature_selection =  best_model.get_params().get('feature_selection', None)
    