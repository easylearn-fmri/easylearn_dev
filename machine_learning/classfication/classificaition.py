#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, RFE,RFECV, VarianceThreshold, mutual_info_classif
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from sklearn.svm import LinearSVC, SVC  # NOTE. If using SVC, then search C will very slow.
from sklearn.linear_model import LogisticRegression, Lasso, ridge_regression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import datasets

from eslearn.base import BaseMachineLearning
from eslearn.machine_learning.classfication._base_classificaition import PipelineSearch_

x, y = datasets.make_classification(n_samples=200, n_classes=3,
                                    n_informative=50, n_redundant=3,
                                    n_features=100, random_state=1)

time_start = time.time()

class Classification(BaseMachineLearning):
    
    def __init__(self):
        super(Classification, self).__init__

    def classification(self, 
                       method_feature_preprocessing=None, 
                       param_feature_preprocessing=None,
                       method_dim_reduction=None,
                       param_dim_reduction=None,
                       method_feature_selection=None,
                       param_feature_selection=None,
                       method_machine_learning=None,
                       param_machine_learning=None,
                       x=None, 
                       y=None):
        
        pipeline = PipelineSearch_(search_strategy='grid', n_jobs=2)
        pipeline.make_pipeline_(
            method_feature_preprocessing=[method_feature_preprocessing], 
            param_feature_preprocessing=param_feature_preprocessing, 
            method_dim_reduction=[method_dim_reduction], 
            param_dim_reduction=param_dim_reduction, 
            method_feature_selection=[method_feature_selection],
            param_feature_selection=param_feature_selection,
            method_machine_learning=[method_machine_learning], 
            param_machine_learning=param_machine_learning
        )

        pipeline.fit_pipeline_(x, y)
        pipeline.get_weights_(x, y)
        yhat, y_porb = pipeline.predict(x)

        time_end = time.time()

        print(f"Running time = {time_end-time_start}\n")
        print("="*50)


if __name__ == "__main__":
    clf = Classification()

    clf.get_configuration_(configuration_file=r'F:\Python378\Lib\site-packages\eslearn\GUI\test\configuration_file.json')
    clf.get_preprocessing_parameters()
    clf.get_dimension_reduction_parameters()
    clf.get_feature_selection_parameters()
    clf.get_unbalance_treatment_parameters()
    clf.get_machine_learning_parameters()
    clf.get_model_evaluation_parameters()
    
    print(clf.method_feature_preprocessing)
    print(clf.param_feature_preprocessing)
    
    print(clf.method_dim_reduction)
    print(clf.param_dim_reduction)
    
    print(clf.method_feature_selection)
    print(clf.param_feature_selection)
    
    print(clf.method_unbalance_treatment)
    print(clf.param_unbalance_treatment)
    
    print(clf.method_machine_learning)
    print(clf.param_machine_learning)

    print(clf.method_model_evaluation)
    print(clf.param_model_evaluation)


    
    method_feature_preprocessing = clf.method_feature_preprocessing
    param_feature_preprocessing= clf.param_feature_preprocessing

    method_dim_reduction = clf.method_dim_reduction
    param_dim_reduction = clf.param_dim_reduction

    method_feature_selection = clf.method_feature_selection
    param_feature_selection = clf.param_feature_selection

    method_machine_learning = clf.method_machine_learning
    param_machine_learning = clf.param_machine_learning
    
    clf.classification(
        method_feature_preprocessing, 
        param_feature_preprocessing,
        method_dim_reduction,
        param_dim_reduction,
        method_feature_selection,
        param_feature_selection, 
        method_machine_learning, 
        param_machine_learning,
        x, y
    )