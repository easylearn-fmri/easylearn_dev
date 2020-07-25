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
        pass

    def classification(self, 
                       method_feature_preprocessing=None, 
                       param_feature_preprocessing=None,
                       method_dim_reduction=None,
                       param_dim_reduction=None,
                       method_feature_selection=None,
                       param_feature_selection=None,
                       type_estimator=None,
                       param_estimator=None,
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
            type_estimator=[type_estimator], 
            param_estimator=param_estimator
        )

        pipeline.fit_pipeline_(x, y)
        pipeline.get_weights_(x, y)
        yhat, y_porb = pipeline.predict(x)

        time_end = time.time()

        print(f"Running time = {time_end-time_start}\n")
        print("="*50)


if __name__ == "__main__":
    clf = Classification()
    
    method_feature_preprocessing = eval("MinMaxScaler()")
    param_feature_preprocessing= {"feature_preprocessing__feature_range":[(0,1)]}

    method_dim_reduction=eval("PCA()")
    param_dim_reduction={'dim_reduction__n_components':[0.5, 0.9]}

    method_feature_selection = eval("RFECV(estimator=LinearSVC())")
    param_feature_selection={'feature_selection__estimator': [BayesianRidge()], 'feature_selection__step': [0.2]}

    type_estimator = eval("SVC()")
    param_estimator={
        'estimator__C':[1,2,10]
    }
    
    clf.classification(
        method_feature_preprocessing, 
        param_feature_preprocessing,
        method_dim_reduction,
        param_dim_reduction,
        method_feature_selection,
        param_feature_selection, 
        type_estimator, 
        param_estimator,
        x, y)