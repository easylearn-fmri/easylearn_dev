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


class BaseClassification(metaclass=ABCMeta):
    """Base class for classification"""

    def __init__(self):
        self.weights_ = None
        self.weights_norm_ = None
    
    def get_weights_(self, x=None, y=None):
        """
        If the model is linear model, the weights are coefficients.
        If the model is not the linear model, the weights are calculated by occlusion test <Transfer learning improves resting-state functional
        connectivity pattern analysis using convolutional neural networks>.
        """
        
        best_model = self.model.best_estimator_
        estimator =  best_model['estimator']
        dim_reduction = best_model.get_params().get('dim_reduction',None)
        feature_selection =  best_model.get_params().get('feature_selection', None)

        # Get weight according to model type: linear model or nonlinear model
        coef_ =  estimator.__dict__.get("coef_", None)
        if coef_.any():  # Linear model
            if feature_selection:
                self.weights_ = [np.zeros(np.size(feature_selection.get_support())) for i in range(len(coef_))]
            else:
                self.weights_ = [[] for i in range(len(coef_))]
                
            for i, coef__ in enumerate(coef_):
                if feature_selection:
                    self.weights_[i][feature_selection.get_support()] = coef__
                else:
                    self.weights_[i] = coef__

                if dim_reduction:
                    self.weights_[i] = dim_reduction.inverse_transform(self.weights_[i])
        else:  # Nonlinear model
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
        
    
class PipelineSearch_(BaseClassification):
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

    def make_pipeline_(self, 
                        method_feature_preprocessing=None, 
                        param_feature_preprocessing=None,
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
            method_feature_preprocessing: [list of] sklearn module, such as [PCA()]
                    method of dimension reduction.

            param_feature_preprocessing: dictionary [or list of dictionaries], {'reduce_dim__n_components':[0.5,0.9]}, 
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
                ('feature_preprocessing','passthrough'),
                ('dim_reduction', 'passthrough'),
                ('feature_selection', 'passthrough'),
                ('estimator', 'passthrough'),
            ], 
            memory=self.memory
        )

        # Set parameters of gridCV
        print("Setting parameters of gridCV...\n")
        
        self.param_search = {}

        if method_feature_preprocessing:
            self.param_search.update({'feature_preprocessing':method_feature_preprocessing})
        if param_feature_preprocessing:          
            self.param_search.update(param_feature_preprocessing)
            
        if method_dim_reduction:
            self.param_search.update({'dim_reduction':method_dim_reduction})
        if param_dim_reduction:
            self.param_search.update(param_dim_reduction)
            # Get number of features that pass the dimension reduction
            # pca = method_dim_reduction[0]
            # xx = pca.fit_transform(x)
                
        if method_feature_selection:
            self.param_search.update({'feature_selection': method_feature_selection})
        if param_feature_selection:
            self.param_search.update(param_feature_selection)
            
        if type_estimator:
            self.param_search.update({'estimator': type_estimator})
        if param_estimator:
            self.param_search.update(param_estimator)
        
        return self
    
    def fit_pipeline_(self, x=None, y=None):
        """Fit the pipeline"""

        cv = StratifiedKFold(n_splits=self.k)  # Default is StratifiedKFold
        if self.search_strategy == 'grid':
            self.model = GridSearchCV(
                self.pipe, n_jobs=self.n_jobs, param_grid=self.param_search, cv=cv, 
                scoring = make_scorer(self.metric), refit=True
            )
            # print(f"GridSearchCV fitting (about {iteration_num} times iteration)...\n")

        elif self.search_strategy == 'random':
            self.model = RandomizedSearchCV(
                self.pipe, n_jobs=self.n_jobs, param_distributions=self.param_search, cv=cv, 
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

        return self

    def predict(self, x):
        y_hat = self.model.predict(x)
        
        # TODO
        try:
            y_prob = self.model.predict_proba(x)
        except:
            y_prob = []
                
        return y_hat, y_prob

