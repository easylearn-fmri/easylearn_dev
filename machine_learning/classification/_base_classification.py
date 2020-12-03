#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
This class is the base class for classification
"""

import numpy as np

import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import make_scorer, accuracy_score, auc, f1_score
from sklearn.pipeline import Pipeline
from joblib import Memory
from shutil import rmtree
import warnings
from sklearn.exceptions import ConvergenceWarning

from eslearn.machine_learning.base import AbstractSupervisedMachineLearningBase
from eslearn.utils.timer import  timer


warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


class BaseClassification():
    """Base class for classification

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
                search_strategy='grid', 
                gridcv_k=3, 
                metric=accuracy_score, 
                n_iter_of_randomedsearch=10, 
                n_jobs=2):

        self.search_strategy = search_strategy
        self.gridcv_k = gridcv_k
        self.metric = metric
        self.n_iter_of_randomedsearch = n_iter_of_randomedsearch
        self.n_jobs = n_jobs
        
        self.model_ = None
        self.weights_ = None
        self.weights_norm_ = None

    @timer 
    def fit_sklearn_search_model(self, pipeline, x=None, y=None):
        """Fit the scikit-learn search or pipeline model
        """
        
        self.model_.fit(x, y)
        # Delete the temporary cache before exiting
        self.memory.clear(warn=False)
        return self
    
    def predict_(self, x):
        # TODO: extend this to multiple-class classification

        y_pred = self.model_.predict(x)
        if hasattr(self.model_, 'predict_proba'):
            y_prob = self.model_.predict_proba(x)
        elif hasattr(self.model_, 'decision_function'):
            y_prob = self.model_.decision_function(x)
        else:
            y_prob = y_pred
                
        return y_pred, y_prob
    
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
        if hasattr(estimator, "coef_"):  # Linear model
            coef =  estimator.coef_
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
            
            y_pred = self.model_.predict(x)
            score_true = self.metric(y, y_pred)
            len_feature = x_reduced_selected.shape[1]
            self.weights_ = np.zeros([1,len_feature])
            
            if len_feature > 1000:
                 print(f"***There are {len_feature} features, it may take a long time to get the weight!***\n")
                 print("***I suggest that you reduce the dimension of features***\n")
                 
            for ifeature in range(len_feature):
                print(f"Getting weight for the {ifeature+1}th feature...\n")
                x_ = x_reduced_selected.copy()
                x_[:,ifeature] = 0
                y_pred = estimator.predict(x_)
                self.weights_[0, ifeature] = score_true - self.metric(y, y_pred)
            
            # Back to original space
            if feature_selection and (feature_selection != "passthrough"):
                self.weights_ = feature_selection.inverse_transform(self.weights_)
            if dim_reduction and (dim_reduction != "passthrough"):
                self.weights_  = dim_reduction.inverse_transform(self.weights_)            
                
        # Normalize weights
        self.weights_norm_ = StandardScaler().fit_transform(self.weights_.T).T
        
        
if __name__=="__main__":
    baseclf = BaseClassification()