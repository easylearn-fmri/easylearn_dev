#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import time
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from sklearn.svm import LinearSVC  # NOTE. If using SVC, then search C will very slow.
from sklearn import datasets

from eslearn.machine_learning.classfication._base_clf import _Pipeline

x, y = datasets.make_classification(n_samples=500, n_classes=3,
                                    n_informative=50, n_redundant=3,
 
                                    n_features=1000, random_state=1)

time_start = time.time()

pipeline = _Pipeline(search_strategy='grid', n_jobs=2)
pipeline._make_pipeline(
                    method_dim_reduction=[PCA()], 
                    param_dim_reduction={'dim_reduction__n_components':[0.5,0.9]}, 
                    method_feature_selection=[SelectKBest(f_classif)],
                    param_feature_selection={'feature_selection__k': [1,2]},
                    type_estimator=[LinearSVC()], 
                    param_estimator={'estimator__C': [1,10]}
)

pipeline._fit_pipeline(x, y)
pipeline.get_weights()
time_end = time.time()

print("="*100)
print(f"Running time = {time_end-time_start}\n")
print("="*100)