#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
This class is the base class for regression
Author: Mengshi dong <dongmengshi1990@163.com>
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from joblib import Memory
from abc import abstractmethod, ABCMeta
import warnings
from sklearn.exceptions import ConvergenceWarning

from eslearn.machine_learning.base import AbstractSupervisedMachineLearningBase
from eslearn.utils.timer import timer

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


class BaseRegression(AbstractSupervisedMachineLearningBase):
    """Base class for regression

    Parameters
    ----------
    None

    Attributes
    ----------
    model_: Fited model object, default None

    weights_: ndarray of shape(n_class, n_features) if the model is linear model, else shape(1,n_features), default None
        Feature weights of the fited model

    weights_norm_: ndarray of shape(n_class, n_features) if the model is linear model, else shape(1,n_features), default None
        Normalized feature weights. Using StandardScaler (z-score) to get the normalized feature weights.

    """

    def __init__(self, 
                metric=mean_squared_error,
                search_strategy='random', 
                k=5, 
                n_iter_of_randomedsearch=10, 
                n_jobs=1, 
                location='cachedir',
                verbose=False):
        
        self.metric = metric
        self.model_ = None
        self.weights_ = None
        self.weights_norm_ = None

        self.search_strategy = search_strategy
        self.k = k
        self.n_iter_of_randomedsearch = n_iter_of_randomedsearch
        self.n_jobs = n_jobs
        self.location = location
        self.verbose = verbose
        
    @timer
    def fit_(self, x=None, y=None):
        """Fit the pipeline_"""
        
        # TODO: Extending to other CV methods
        cv = KFold(n_splits=self.k)  # Default is StratifiedKFold
        if self.is_search:
            if self.search_strategy == 'grid':
                self.model_ = GridSearchCV(
                    self.pipeline_, n_jobs=self.n_jobs, param_grid=self.param_search_, cv=cv, 
                    scoring = make_scorer(self.metric), refit=True
                )
                # print(f"GridSearchCV fitting (about {iteration_num} times iteration)...\n")

            elif self.search_strategy == 'random':
                self.model_ = RandomizedSearchCV(
                    self.pipeline_, n_jobs=self.n_jobs, param_distributions=self.param_search_, cv=cv, 
                    scoring = make_scorer(self.metric), refit=True, n_iter=self.n_iter_of_randomedsearch,
                )
            
                # print(f"RandomizedSearchCV fitting (about {iteration_num} times iteration)...\n")
            else:
                print("Please specify which search strategy!\n")
                return
        else:
            self.model_ = self.pipeline_

        # Fit
        self.model_.fit(x, y)

        # Delete the temporary cache before exiting
        self.memory.clear(warn=False)
        return self

    def predict_(self, x):
        y_hat = self.model_.predict(x)
        return y_hat

    def get_weights_(self, x=None, y=None):
        """
        If the model is linear model, the weights are coefficients.
        If the model is not the linear model, the weights are calculated by occlusion test <Transfer learning improves resting-state functional
        connectivity pattern analysis using convolutional neural networks>.
        """
        
        if self.is_search:
            best_model = self.model_.best_estimator_
        else:
            best_model = self.model_
            
        feature_preprocessing = best_model['feature_preprocessing']
        dim_reduction = best_model.get_params().get('dim_reduction',None)
        feature_selection =  best_model.get_params().get('feature_selection', None)
        estimator =  best_model['estimator']

        # Get weight according to model type: linear model or nonlinear model
        if hasattr(estimator, "coef_"):
            coef =  estimator.coef_.reshape(1,-1)                
            if feature_selection and (feature_selection != "passthrough"):
                self.weights_ = feature_selection.inverse_transform(coef)
            else:
                self.weights_ = coef

            if dim_reduction and (dim_reduction != "passthrough"):
                self.weights_ = dim_reduction.inverse_transform(self.weights_)
            
        else:  # Nonlinear model
        # TODO: Consider the problem of slow speed caused by a large number of features
            x_reduced_selected = x.copy()
            if feature_preprocessing and (feature_preprocessing != "passthrough"):
                x_reduced_selected = feature_preprocessing.fit_transform(x_reduced_selected)
            if dim_reduction and (dim_reduction != "passthrough"):
                x_reduced_selected = dim_reduction.fit_transform(x_reduced_selected)
            if feature_selection and (feature_selection != "passthrough"):
                x_reduced_selected = feature_selection.fit_transform(x_reduced_selected, y)

            y_hat = self.model_.predict(x)
            score_true = self.metric(y, y_hat)
            len_feature = np.shape(x_reduced_selected)[1]
            
            if len_feature > 1000:
                 print("***There are {len_feature} features, it may take a long time to get the weight!***\n")
                 print("***I suggest that you reduce the dimension of features***\n")
            
            self.weights_ = np.zeros([1,len_feature])
            for ifeature in range(len_feature):
                print(f"Getting weight for the {ifeature+1}th feature...\n")
                x_ = np.array(x_reduced_selected).copy()
                x_[:,ifeature] = 0
                y_hat = estimator.predict(x_)
                self.weights_[0, ifeature] = score_true - self.metric(y, y_hat)
            
            # Back to original space
            self.weights_ = np.reshape(self.weights_, [1, -1])
            if feature_selection and (feature_selection != "passthrough"):
                self.weights_ = feature_selection.inverse_transform(self.weights_)
            if dim_reduction and (dim_reduction != "passthrough"):
                self.weights_  = dim_reduction.inverse_transform(self.weights_)            
                
        # Normalize weights using z-score method
        self.weights_norm_ = StandardScaler().fit_transform(self.weights_.T).T
    

if __name__ == "__main__":
    basereg = BaseRegression()

