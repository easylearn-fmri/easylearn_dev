#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
This class is the base class for classification
"""


import sys
import os
import numpy as np
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC  # NOTE. If using SVC, then search C will very slow.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import RandomOverSampler
from joblib import Memory
from shutil import rmtree
import nibabel as nib


class Clf():
    """Base class for classification"""

    def __init__(self):
        pass
        
class _Pipeline(Clf):
    """Make pipeline"""

    def __init__(self):
        super().__init__()

    def _make_pipeline(self, method_dim_reduction=None, param_dim_reduction=None, 
                        method_feature_selection=None, param_feature_selection=None, 
                        type_estimator=None, param_estimator=None):
        """Construct pipeline

        Currently, the pipeline only supports one specific method for corresponding method, 
        e.g., only supports one dimension reduction method for dimension reduction.
        In the next version, the pipeline will support multiple methods for each corresponding method.

        Parameters:
        ----------
            method_dim_reduction: [list of] sklearn module, such as [PCA()]
                method of dimension reduction.

            param_dim_reduction: dictionary [or list of dictionaries], {'reduce_dim__n_components':[0.5,0.9]}, 
                parameters of dimension reduction, such as components of PCA.

            method_feature_selection: [list of] sklearn module, such as [LinearSVC()]
                method of feature selection.

            param_feature_selection: dictionary [or list of dictionaries], {'feature_selection__k': [0.5,0.9]},
                parameters of feature selection, such as How many features to be kept.

            type_estimator: [list of] sklearn module, such as [LinearSVC()]
                method of feature selection.

            param_estimator: dictionary [or list of dictionaries], such as 
            {'estimator__penalty': ['l1', 'l2'], 'estimator__C': [10]}
                parameters of feature selection.

        """

        location = 'cachedir'
        self.memory = Memory(location=location, verbose=10)
        self.pipe = Pipeline(steps=[
                ('data_normalization', 'passthrough'),
                ('dim_reduction', 'passthrough'),
                ('feature_selection', 'passthrough'),
                ('estimator', 'passthrough'),
            ], 
            memory=self.memory
        )

        # Set parameters of gridCV
        print("Setting parameters of gridCV...\n")
        
        self.param_grid = {}

        if method_dim_reduction:
            self.param_grid.update({'dim_reduction':method_dim_reduction})
            self.param_grid.update(param_dim_reduction)
        if method_feature_selection:
            self.param_grid.update({'feature_selection': method_feature_selection})
            self.param_grid.update(param_feature_selection)
        if type_estimator:
            self.param_grid.update({'estimator': type_estimator})
            self.param_grid.update(param_estimator)
        
    def _fit_pipeline(self):
        """Fit the pipeline"""

        cv = StratifiedKFold(n_splits=self.k)
        if self.search_strategy == 'grid':
            model = GridSearchCV(
                self.pipe, n_jobs=self.n_jobs, param_grid=self.self.param_grid, cv=cv, 
                scoring = make_scorer(self.metric), refit=True
            )
            # print(f"GridSearchCV fitting (about {iteration_num} times iteration)...\n")

        elif self.search_strategy == 'random':
            model = RandomizedSearchCV(
                self.pipe, n_jobs=self.n_jobs, param_distributions=self.param_grid, cv=cv, 
                scoring = make_scorer(self.metric), refit=True, n_iter=self.n_iter_of_randomedsearch,
            )
        
            # print(f"RandomizedSearchCV fitting (about {iteration_num} times iteration)...\n")
        else:
            print(f"Please specify which search strategy!\n")
            return

        print("Fitting...")
        model.fit(x_train, y_train)

        # Delete the temporary cache before exiting
        self.memory.clear(warn=False)
        rmtree(location)

        return model

if __name__ == "__main__":
    pipeline = _Pipeline()
    pipeline._make_pipeline(method_dim_reduction=[PCA()], 
                        param_dim_reduction={'reduce_dim__n_components':[0.5,0.9]}, 
                        method_feature_selection=[SelectKBest(f_classif)],
                        param_feature_selection={'feature_selection__k': [0.5,0.9]},
                        type_estimator=[LinearSVC()], 
                        param_estimator={'estimator__C': [1,10]}
    )