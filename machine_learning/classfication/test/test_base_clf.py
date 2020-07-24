#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import time
from sklearn.preprocessing import StandardScaler # 标准化数据
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from sklearn.svm import LinearSVC  # NOTE. If using SVC, then search C will very slow.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

from eslearn.machine_learning.classfication._base_clf import _Pipeline

x, y = datasets.make_classification(n_samples=200, n_classes=3,
                                    n_informative=50, n_redundant=3,
                                    n_features=100, random_state=1)

time_start = time.time()

pipeline = _Pipeline(search_strategy='grid', n_jobs=2)
pipeline._make_pipeline(
    # method_unbalanced_treatment=[RandomOverSampler()], 
    # param_unbalanced_treatment={},
    method_normalization=[StandardScaler()], 
    parm_normalization={"normalization__with_mean":[True]},
    method_unbalanced_treatment=[RandomOverSampler()], 
    param_unbalanced_treatment=None,
    method_normalization=[StandardScaler()], 
    parm_normalization=None,
    method_dim_reduction=[PCA()], 
    param_dim_reduction={'dim_reduction__n_components':[0.5,0.9]}, 
    method_feature_selection=[SelectKBest(f_classif)],
    param_feature_selection={'feature_selection__k': [1,2]},
    type_estimator=[LogisticRegression()], 
    param_estimator={'estimator__solver':['saga'], 'estimator__penalty': ['elasticnet'], 'estimator__l1_ratio':[0.5,0.9]}
)

pipeline.fit_pipeline(x, y)
pipeline.get_weights(x, y)
time_end = time.time()


print(f"Running time = {time_end-time_start}\n")
print("="*10)
