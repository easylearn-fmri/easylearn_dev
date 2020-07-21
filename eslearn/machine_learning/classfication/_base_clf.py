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
from abc import abstractmethod, ABCMeta

class Clf(metaclass=ABCMeta):
    """Base class for classification"""

    def __init__(self):
        pass
    
    @abstractmethod
    def get_weights(self):
        """
        If the model is linear model, the weights are coefficients.
        If the model is not the linear model, the weights are calculated by occlusion test <Transfer learning improves resting-state functional
        connectivity pattern analysis using convolutional neural networks>.
        """
       
    
class _Pipeline(Clf):
    """Make pipeline"""

    def __init__(self,search_strategy='random', k=5, 
        metric=accuracy_score, n_iter_of_randomedsearch=10, n_jobs=1,
        location='cachedir'):
        super().__init__()
        self.search_strategy = search_strategy
        self.k = k
        self.metric = metric
        self.n_iter_of_randomedsearch = n_iter_of_randomedsearch
        self.n_jobs = n_jobs
        self.location = location

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

        
        self.memory = Memory(location=self.location, verbose=10)
        self.pipe = Pipeline(steps=[
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
        
    def _fit_pipeline(self, x, y):
        """Fit the pipeline"""

        cv = StratifiedKFold(n_splits=self.k)
        if self.search_strategy == 'grid':
            self.model = GridSearchCV(
                self.pipe, n_jobs=self.n_jobs, param_grid=self.param_grid, cv=cv, 
                scoring = make_scorer(self.metric), refit=True
            )
            # print(f"GridSearchCV fitting (about {iteration_num} times iteration)...\n")

        elif self.search_strategy == 'random':
            self.model = RandomizedSearchCV(
                self.pipe, n_jobs=self.n_jobs, param_distributions=self.param_grid, cv=cv, 
                scoring = make_scorer(self.metric), refit=True, n_iter=self.n_iter_of_randomedsearch,
            )
        
            # print(f"RandomizedSearchCV fitting (about {iteration_num} times iteration)...\n")
        else:
            print(f"Please specify which search strategy!\n")
            return

        print("Fitting...")
        self.model.fit(x, y)

        # Delete the temporary cache before exiting
        self.memory.clear(warn=False)
        rmtree(self.location)

    def get_weights(self, x=None, y=None):
        """
        If the model is linear model, the weights are coefficients.
        If the model is not the linear model, the weights are calculated by occlusion test <Transfer learning improves resting-state functional
        connectivity pattern analysis using convolutional neural networks>.
        """
        
        best_model = self.model.best_estimator_
        estimator =  best_model['estimator']
        dim_reduction = best_model['dim_reduction']
        feature_selection =  best_model['feature_selection']

        weight = np.zeros(np.size(feature_selection.get_support()))
        
        # Check if is linear model, namely have coef_
        estimator_dict = dir(estimator)
        if "coef_" in estimator_dict:
            weight = estimator.coef_
        else:
            y_hat = self.model.predict(x)

    def _predict(self,x):
        pass
