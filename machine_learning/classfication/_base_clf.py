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
from abc import abstractmethod, ABCMeta

from sklearn.utils.testing import ignore_warnings
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


class Clf(metaclass=ABCMeta):
    """Base class for classification"""

    def __init__(self):
        self.weights_ = None
        self.weights_norm_ = None
        self.predict_proba_ = None
        self.predict_ = None
    
    def get_weights(self, x=None, y=None):
        """
        If the model is linear model, the weights are coefficients.
        If the model is not the linear model, the weights are calculated by occlusion test <Transfer learning improves resting-state functional
        connectivity pattern analysis using convolutional neural networks>.
        """
        
        best_model = self.model.best_estimator_
        estimator =  best_model['estimator']
        if "dim_reduction" in self.param_grid.keys():
            dim_reduction = best_model['dim_reduction']
        if "feature_selection" in self.param_grid.keys():
            feature_selection =  best_model['feature_selection']


        # Get weight according to model type: linear model or nonlinear model
        estimator_dict = dir(estimator)
        if "coef_" in estimator_dict:
            if "feature_selection" in self.param_grid.keys():
                self.weights_ = [np.zeros(np.size(feature_selection.get_support())) for i in range(len(estimator.coef_))]
            else:
                self.weights_ = [[] for i in range(len(estimator.coef_))]
                
            for i in range(len(estimator.coef_)):
                if "feature_selection" in self.param_grid.keys():
                    self.weights_[i][feature_selection.get_support()] = estimator.coef_[i] 
                else:
                    self.weights_[i] = estimator.coef_[i] 

                if "dim_reduction" in self.param_grid.keys():
                    self.weights_[i] = dim_reduction.inverse_transform(self.weights_[i])
        else:
            self.weights_ = []
            y_hat = self.model.predict(x)
            score_true = self.metric(y, y_hat)
            len_feature = np.shape(x)[1]
            for ifeature in range(len_feature):
                x_ = np.array(x).copy()
                x_[:,ifeature] = 0
                y_hat = self.model.predict(x_)
                self.weights_.append(score_true-self.metric(y, y_hat))
                
        # Normalize weights
        self.weights_norm_ = [wei/np.sum(np.power(np.e,wei)) for wei in self.weights_]
        
    
class _Pipeline(Clf):
    """Make pipeline"""

    def __init__(self,
        search_strategy='random', 
        k=5, 
        metric=accuracy_score, 
        n_iter_of_randomedsearch=10, 
        n_jobs=1,
        location='cachedir'):

        super().__init__()
        self.search_strategy = search_strategy
        self.k = k
        self.metric = metric
        self.n_iter_of_randomedsearch = n_iter_of_randomedsearch
        self.n_jobs = n_jobs
        self.location = location

    def _make_pipeline(self, 
                        method_unbalanced_treatment=None, 
                        param_unbalanced_treatment=None,
                        method_normalization=None, 
                        parm_normalization=None,
                        method_dim_reduction=None, 
                        param_dim_reduction=None, 
                        method_feature_selection=None, 
                        param_feature_selection=None, 
                        type_estimator=None, 
                        param_estimator=None):
        """Construct pipeline

        Currently, the pipeline only supports one specific method for corresponding method, 
        e.g., only supports one dimension reduction method for dimension reduction.
        In the next version, the pipeline will support multiple methods for each corresponding method.

        Parameters:
        ----------
            method_unbalanced_treatment: [list of] imblearn module, such as [RandomOverSampler()].
                Method of how to treatment unbalanced samples.

            param_unbalanced_treatment: dictionary [or list of dictionaries], such as {'unbalanced_treatment__n_components':[0.5,0.9]}, 
                parameters of treatment unbalanced samples.

            method_normalization: [list of] sklearn module, such as [PCA()]
                    method of dimension reduction.

            parm_normalization: dictionary [or list of dictionaries], {'reduce_dim__n_components':[0.5,0.9]}, 
                parameters of dimension reduction, such as components of PCA.

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

        
        self.memory = Memory(location=self.location, verbose=0)

        self.pipe = Pipeline(steps=[
                ('unbalanced_treatment', 'passthrough'),
                ('normalization','passthrough'),
                ('dim_reduction', 'passthrough'),
                ('feature_selection', 'passthrough'),
                ('estimator', 'passthrough'),
            ], 
            memory=self.memory
        )

        # Set parameters of gridCV
        print("Setting parameters of gridCV...\n")
        
        self.param_grid = {}

        if method_unbalanced_treatment:
            self.param_grid.update({'unbalanced_treatment':method_unbalanced_treatment})
            self.param_grid.update(param_unbalanced_treatment)
        if method_normalization:
            self.param_grid.update({'normalization':method_normalization})
            self.param_grid.update(parm_normalization)
        if method_dim_reduction:
            self.param_grid.update({'dim_reduction':method_dim_reduction})
            self.param_grid.update(param_dim_reduction)
        if method_feature_selection:
            self.param_grid.update({'feature_selection': method_feature_selection})
            self.param_grid.update(param_feature_selection)
        if type_estimator:
            self.param_grid.update({'estimator': type_estimator})
            self.param_grid.update(param_estimator)
    
    def fit_pipeline(self, x=None, y=None):
        """Fit the pipeline"""

        cv = StratifiedKFold(n_splits=self.k)  # Default is StratifiedKFold
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

    def _predict(self,x):
        pass
